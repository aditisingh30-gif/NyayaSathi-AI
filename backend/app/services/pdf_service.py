import re
import uuid
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pymupdf  # PyMuPDF


# ---------------------------------------------------------------------------
# Pattern library — ordered from most-specific to least-specific.
# Each entry is (compiled_regex, label_template, group_for_number).
# ---------------------------------------------------------------------------

# Article/Section/Clause with inline em-dash title (Indian statute format)
# e.g. "21. Protection of life and personal liberty.—No person..."
# e.g. "Section 73. Compensation for loss..."
INLINE_ARTICLE_PATTERN = re.compile(
    r"(?m)^\s*(?:Article\s+)?(\d{1,3}[A-Z]?)\.\s+([A-Z][^\n—–-]{3,100}?)(?:[\.—–-])",
)

# Named patterns for section headers in legal documents
LEGAL_HEADER_PATTERNS = [
    re.compile(r"^\s*(Section\s+\d+[A-Za-z]?[\.:\-]?\s+[^\n]{3,80})", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(Article\s+\d+[A-Za-z]?[\.:\-]?\s+[^\n]{3,80})", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(Clause\s+\d+(\.\d+)*[\.:\-]?\s+[^\n]{3,80})", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(CHAPTER\s+[IVXLCDM\d]+[\.:\-]?\s+[^\n]{3,80})", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*([0-9]+\.\s+[A-Z][A-Za-z\s]{3,60})", re.MULTILINE),
    re.compile(r"^\s*([A-Z\s]{4,40}:)", re.MULTILINE),
]

# Detects Indian statute and constitutional article headers:
# e.g., "21. Protection of life and personal liberty.—"
# e.g., "2[21A. Right to education.—"
# e.g., "Article 21. Protection of life..."
# e.g., "35. (1) A complaint, in relation to..." (Gazette statutory format)
# e.g., "36. (1) Every proceeding..."
# e.g., "Section 35. Manner in which complaint shall be made"
CONSTITUTION_ARTICLE_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+\[)?(?:(?:Section|Article|Clause)\s+)?(\d{1,3}[A-Za-z]?)\.\s*(?:([A-Z][A-Za-z ,\[\]()0-9]{2,100}?)\.?[—\-]|\(([0-9a-zA-Z]+)\)|([A-Z][A-Za-z ,\[\]()0-9]{2,100}))"
)

# Schedules / Appendix headings — to tag chunks as non-mainbody
SCHEDULE_RE = re.compile(
    r"(?i)(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+schedule",
)
APPENDIX_RE = re.compile(r"(?i)(appendix|annexure|schedule)\s+[A-Z\d]")


def _is_schedule_text(text: str) -> bool:
    """Return True if the text belongs to a Schedule or Appendix (not main body)."""
    return bool(SCHEDULE_RE.search(text) or APPENDIX_RE.search(text))


def _extract_article_number(header: str) -> Optional[str]:
    """
    Pull the bare article/section number from a header string.
    E.g. "21. Protection of life..." -> "21"
         "Article 21A. Right to..." -> "21A"
         "Section 73. Compensation" -> "73"
    Returns lowercase string or None.
    """
    m = re.match(r"^\s*(?:article|section|clause)?\s*(\d{1,3}[A-Za-z]?)\b", header, re.I)
    return m.group(1).lower() if m else None


class PDFProcessingService:
    """
    Handles PDF parsing, page extraction, legal hierarchy detection,
    and semantic chunking using PyMuPDF.

    Key improvements over v1:
    - Constitution-aware article boundary splitting (bare "21. Title—" format)
    - Per-article chunks with accurate section_header and article_number metadata
    - Schedule/Appendix tagging to prevent non-mainbody chunks polluting results
    - Page numbers always reflect actual PDF page (1-indexed)
    """

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_document(
        self,
        file_source: Any,
        filename: str = "document.pdf",
        doc_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extract text, structure, and metadata from a PDF file or bytes.
        Returns:
            metadata: dict with summary info
            pages_data: list of dicts per page
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]

        if isinstance(file_source, (str, Path)):
            doc = pymupdf.open(str(file_source))
        elif isinstance(file_source, bytes):
            doc = pymupdf.open(stream=file_source, filetype="pdf")
        else:
            raise ValueError("file_source must be file path or bytes")

        pages_data = []
        full_text = []

        try:
            num_pages = len(doc)
            raw_title = doc.metadata.get("title") if doc.metadata else None
            # Reject raw printer/software artifacts (e.g., '2606GI.p65', 'word.docx', 'untitled')
            # and fall back to the clean uploaded filename stem
            is_artifact = (
                not raw_title or len(raw_title.strip()) < 3
                or any(raw_title.lower().endswith(ext) for ext in (".p65", ".docx", ".pdf", ".qxd", ".txt", ".rtf"))
                or raw_title.lower().startswith("microsoft word")
                or raw_title.lower() in ("untitled", "default", "document")
                or bool(re.match(r"^\d+[a-zA-Z0-9_\-\.]*$", raw_title.strip()))
            )
            if is_artifact:
                title = Path(filename).stem.replace("_", " ").title()
            else:
                title = raw_title.strip()

            for page_index in range(num_pages):
                page = doc[page_index]
                page_num = page_index + 1  # 1-indexed, matches actual PDF page
                text = page.get_text("text")

                detected_sections = self._detect_legal_sections(text)
                primary_section = detected_sections[0] if detected_sections else ""

                # Detect if page is part of a Schedule/Appendix
                is_schedule = _is_schedule_text(text)

                pages_data.append({
                    "page_number": page_num,
                    "text": text.strip(),
                    "detected_sections": detected_sections,
                    "primary_section": primary_section,
                    "char_count": len(text),
                    "is_schedule": is_schedule,
                })
                full_text.append(text)

            combined_text = "\n".join(full_text)
            hasher = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

            doc_metadata = {
                "doc_id": doc_id,
                "filename": filename,
                "title": title,
                "num_pages": num_pages,
                "total_chars": len(combined_text),
                "sha256": hasher,
            }

            return doc_metadata, pages_data

        finally:
            doc.close()

    def _detect_legal_sections(self, text: str) -> List[str]:
        """Scan text for statutory section headings or contract clauses."""
        sections = []
        for pattern in LEGAL_HEADER_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                header = m[0] if isinstance(m, tuple) else m
                header = header.strip()
                if header and header not in sections:
                    sections.append(header)
        return sections

    def chunk_document(
        self,
        doc_metadata: Dict[str, Any],
        pages_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Split extracted pages into legal-oriented semantic chunks.

        Strategy:
        1. Try to split each page on article/section boundaries (Constitution style).
        2. Where article boundaries are found, each article gets its own chunk with
           accurate section_header and article_number metadata.
        3. For pages without article markers, fall back to paragraph-based chunking.
        4. Schedule/Appendix pages are tagged in metadata (is_schedule=True).
        5. Page numbers always reflect actual PDF page number.
        """
        doc_id = doc_metadata["doc_id"]
        doc_title = doc_metadata.get("title", "Legal Document")
        chunks = []
        chunk_counter = 1

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]
            is_schedule = page.get("is_schedule", False)

            if not page_text:
                continue

            # Try article-boundary splitting first
            article_chunks = self._split_by_article_boundaries(page_text, doc_title=doc_title)

            if article_chunks:
                # Each article_chunk = (article_number, header, text)
                for art_num, art_header, art_text in article_chunks:
                    if not art_text.strip():
                        continue
                    # Long article text: further split if needed
                    sub_chunks = self._split_text_by_size(art_text)
                    for sub_idx, sub_text in enumerate(sub_chunks):
                        if not sub_text.strip():
                            continue
                        chunk_id = f"{doc_id}_p{page_num}_c{chunk_counter}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "page_number": page_num,
                            "section_header": art_header,
                            "article_number": art_num,       # e.g. "21", "21a", "73"
                            "is_schedule": is_schedule,
                            "text": sub_text.strip(),
                            "metadata": {
                                "document_id": doc_id,
                                "document_title": doc_title,
                                "page_number": page_num,
                                "section_header": art_header,
                                "article_number": art_num,
                                "is_schedule": str(is_schedule),
                                "chunk_id": chunk_id,
                            }
                        })
                        chunk_counter += 1
            else:
                # Fallback: paragraph-based chunking (for pages without article markers)
                current_section = page.get("primary_section", "")
                raw_blocks = page_text.split("\n\n")
                paragraphs = []
                for blk in raw_blocks:
                    sub_lines = blk.split("\n")
                    curr = []
                    for line in sub_lines:
                        is_header = any(pat.search(line) for pat in LEGAL_HEADER_PATTERNS)
                        if is_header and curr:
                            paragraphs.append("\n".join(curr).strip())
                            curr = [line]
                        else:
                            curr.append(line)
                    if curr:
                        paragraphs.append("\n".join(curr).strip())

                paragraphs = [p for p in paragraphs if p]
                if not paragraphs:
                    paragraphs = [page_text]

                current_chunk_text = ""
                current_chunk_section = current_section

                for para in paragraphs:
                    for pattern in LEGAL_HEADER_PATTERNS:
                        match = pattern.search(para)
                        if match:
                            current_chunk_section = match.group(1).strip()
                            break

                    if len(current_chunk_text) + len(para) > self.chunk_size and current_chunk_text:
                        chunk_id = f"{doc_id}_p{page_num}_c{chunk_counter}"
                        chunks.append(self._make_chunk(
                            chunk_id, doc_id, doc_title, page_num,
                            current_chunk_section, current_chunk_text,
                            is_schedule=is_schedule
                        ))
                        chunk_counter += 1
                        overlap = current_chunk_text[-self.chunk_overlap:] if len(current_chunk_text) > self.chunk_overlap else ""
                        current_chunk_text = overlap + " " + para
                    else:
                        current_chunk_text = (current_chunk_text + "\n\n" + para).strip() if current_chunk_text else para

                if current_chunk_text.strip():
                    chunk_id = f"{doc_id}_p{page_num}_c{chunk_counter}"
                    chunks.append(self._make_chunk(
                        chunk_id, doc_id, doc_title, page_num,
                        current_chunk_section, current_chunk_text,
                        is_schedule=is_schedule
                    ))
                    chunk_counter += 1

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_by_article_boundaries(
        self, page_text: str, doc_title: str = ""
    ) -> List[Tuple[str, str, str]]:
        """
        Detect article/section boundaries within a page's text.

        Looks for the Indian statute pattern: "21. Title text.—body..." or "2[21A. Title.—..."
        Returns list of (article_number_str, full_header_str, body_text).
        If leading text exists before the first article boundary, it is returned with
        article_number="" so it is preserved without contaminating the first article.
        Returns [] if no article markers found.
        """
        # Find all article-start positions
        matches = list(CONSTITUTION_ARTICLE_RE.finditer(page_text))
        if not matches:
            return []

        results = []

        # Determine default provision label: Constitution uses Article, statutes use Section, contracts use Clause
        doc_lower = doc_title.lower() if doc_title else ""
        if "constitution" in doc_lower:
            default_prefix = "Article"
        elif any(k in doc_lower for k in ("contract", "agreement", "msa", "nda", "terms")):
            default_prefix = "Clause"
        else:
            default_prefix = "Section"

        # If there is content before the first article boundary on this page, capture it
        if matches[0].start() > 0:
            pre_text = page_text[:matches[0].start()].strip()
            if pre_text and len(pre_text) > 30:
                results.append(("", "", pre_text))

        for i, m in enumerate(matches):
            art_num = m.group(1).lower()          # "20", "21", "21a", "22", "35"
            title_with_dash = m.group(2)
            title_bare = m.group(4)
            art_title = (title_with_dash or title_bare or "").strip()

            # Inspect the exact matched prefix text to preserve statutory terminology
            match_str = m.group(0).lower()
            if "article" in match_str:
                prefix = "Article"
            elif "section" in match_str:
                prefix = "Section"
            elif "clause" in match_str:
                prefix = "Clause"
            else:
                prefix = default_prefix

            if art_title:
                art_header = f"{prefix} {m.group(1)}. {art_title}"
            else:
                art_header = f"{prefix} {m.group(1)}"

            # Text runs from match start to next match start (or end of page)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
            art_text = page_text[start:end].strip()

            results.append((art_num, art_header, art_text))

        return results

    def _split_text_by_size(self, text: str) -> List[str]:
        """Split long text into chunks of at most chunk_size chars with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        pos = 0
        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            chunks.append(text[pos:end])
            pos += self.chunk_size - self.chunk_overlap
        return chunks

    def _make_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        doc_title: str,
        page_num: int,
        section_header: str,
        text: str,
        is_schedule: bool = False,
    ) -> Dict[str, Any]:
        art_num = _extract_article_number(section_header) or ""
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "page_number": page_num,
            "section_header": section_header,
            "article_number": art_num,
            "is_schedule": is_schedule,
            "text": text.strip(),
            "metadata": {
                "document_id": doc_id,
                "document_title": doc_title,
                "page_number": page_num,
                "section_header": section_header,
                "article_number": art_num,
                "is_schedule": str(is_schedule),
                "chunk_id": chunk_id,
            }
        }

    @staticmethod
    def generate_legal_pdf(
        title: str,
        sections: List[Tuple[str, str]],
        output_path: str,
    ) -> str:
        """
        Utility to create genuine, well-structured legal benchmark PDFs
        with PyMuPDF for testing and out-of-the-box system demo.
        """
        doc = pymupdf.open()
        doc.set_metadata({"title": title, "author": "NyayaSathi Legal AI"})
        page = doc.new_page(width=595, height=842)  # A4 size

        rect = pymupdf.Rect(50, 40, 545, 100)
        page.insert_textbox(rect, title.upper(), fontsize=16, fontname="helv", color=(0.1, 0.2, 0.4), align=1)

        y = 110
        for sec_title, sec_content in sections:
            if y > 720:
                page = doc.new_page(width=595, height=842)
                y = 50

            header_rect = pymupdf.Rect(50, y, 545, y + 25)
            page.insert_textbox(header_rect, sec_title, fontsize=12, fontname="helv", color=(0.15, 0.15, 0.2))
            y += 26

            body_text = sec_content.strip()
            estimated_lines = len(body_text) // 70 + body_text.count("\n") + 1
            body_height = max(40, estimated_lines * 14)

            if y + body_height > 780:
                page = doc.new_page(width=595, height=842)
                y = 50

            body_rect = pymupdf.Rect(50, y, 545, y + body_height + 10)
            page.insert_textbox(body_rect, body_text, fontsize=10, fontname="helv", color=(0.25, 0.25, 0.25))
            y += body_height + 25

        doc.save(output_path)
        doc.close()
        return output_path


# Global instance
pdf_service = PDFProcessingService()
