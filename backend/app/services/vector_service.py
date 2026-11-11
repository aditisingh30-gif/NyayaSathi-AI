import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from app.config import settings

# Common English and legal filler words that must not distort semantic vector dot-products
LEGAL_STOP_WORDS = {
    "what", "is", "of", "the", "in", "and", "or", "a", "an", "to", "for", "with",
    "by", "as", "at", "on", "from", "this", "that", "it", "are", "be", "any", "all",
    "shall", "which", "who", "whom", "whose", "does", "do", "did", "how", "where",
    "when", "can", "could", "would", "should", "under", "about", "into", "their",
    "there", "then", "them", "these", "those", "such", "may", "been", "have", "has",
    "had", "having", "having", "between", "through", "during", "before", "after", "above",
    "below", "other", "some", "only", "own", "same", "so", "than", "too", "very", "s"
}


def extract_substantive_tokens(text: str) -> Set[str]:
    """Extract substantive legal terms, numbers, and identifiers, discarding stop words."""
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return {w for w in words if w not in LEGAL_STOP_WORDS and len(w) > 1}


def extract_provision_target(text: str) -> Optional[str]:
    """
    Extract statutory section, article, or clause target from text.
    Handles 'Article 21', 'Article 21A', 'Section 73', 'Clause 9',
    preserving alphanumeric suffixes like '21a' without confusing with '21'.
    """
    matches = re.findall(r"\b(?:section|article|clause)\s+(\d+[a-z]?)\b", text, re.I)
    return matches[0].lower() if matches else None


def extract_header_provision(header: str) -> Optional[str]:
    """
    Extract statutory provision number from a section header string.
    Handles explicit prefixes like 'Article 21. ...', 'Section 73. ...'
    as well as bare number headers like '21. Protection of life...', '11. In this Part', '110. Power...'
    """
    # 1. Explicit prefix
    m = re.findall(r"\b(?:section|article|clause)\s+(\d+[a-z]?)\b", header, re.I)
    if m:
        return m[0].lower()
    # 2. Bare number prefix at start of header e.g. "21.", "21A.", "110."
    m2 = re.match(r"^\s*(?:\d+\[)?(\d+[a-z]?)\.", header, re.I)
    if m2:
        return m2.group(1).lower()
    return None


class LegalSemanticEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Substantive legal vector embedding function.
    Filters out common stop words so only substantive statutory concepts,
    section numbers, and act titles contribute to dense vector representations.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for text in input:
            vec = np.zeros(self.dimension, dtype=np.float32)
            substantive = extract_substantive_tokens(text)

            # Also generate word pairs (bigrams) for legal phrases (e.g. data_breach, personal_data)
            tokens = list(substantive)
            raw_words = [w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()) if w not in LEGAL_STOP_WORDS]
            for i in range(len(raw_words) - 1):
                tokens.append(f"{raw_words[i]}_{raw_words[i+1]}")

            for token in tokens:
                h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % self.dimension
                sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
                vec[idx] += sign

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings


class ChromaVectorService:
    """
    Manages vector storage, indexing, and similarity retrieval
    for legal document chunks using ChromaDB with hybrid lexical-semantic filtering.
    """

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.embedding_function = LegalSemanticEmbeddingFunction(dimension=384)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "NyayaSathi Legal Document Corpus", "hnsw:space": "cosine"}
        )

        self.registry_file = Path(self.persist_dir) / "documents_registry.json"
        self._doc_registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_registry(self) -> None:
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._doc_registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to persist document registry: {e}")

    def add_document_chunks(
        self,
        doc_metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> int:
        if not chunks:
            return 0

        doc_id = doc_metadata["doc_id"]
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = []

        for c in chunks:
            meta = {
                "document_id": str(doc_id),
                "document_title": str(doc_metadata.get("title", "Legal Document")),
                "category": str(doc_metadata.get("category", "General Legal")),
                "page_number": int(c.get("page_number", 1)),
                "section_header": str(c.get("section_header", "")),
                # New fields for precise article-number matching
                "article_number": str(c.get("article_number", "")),
                "is_schedule": str(c.get("is_schedule", False)),
                "chunk_id": str(c["chunk_id"]),
            }
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        self._doc_registry[doc_id] = {
            "doc_id": doc_id,
            "filename": doc_metadata.get("filename", "document.pdf"),
            "title": doc_metadata.get("title", "Legal Document"),
            "category": doc_metadata.get("category", "General Legal"),
            "num_pages": doc_metadata.get("num_pages", 1),
            "num_chunks": len(chunks),
            "file_size_bytes": doc_metadata.get("total_chars", 0),
            "uploaded_at": doc_metadata.get("uploaded_at", ""),
            "summary": doc_metadata.get("summary", ""),
        }
        self._save_registry()
        return len(chunks)

    def query_similar(
        self,
        query_text: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        min_relevance_threshold: float = 0.35,
        original_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid vector and lexical relevance retrieval.
        Applies substantive keyword overlap and strict thresholding to eliminate
        unrelated false positives.

        Args:
            query_text: The (optionally expanded) query used for vector retrieval.
            original_query: The raw user query (pre-expansion) used for section-number
                            mismatch detection. When provided, section guard uses this
                            instead of query_text to avoid false positives from expansion.
        """
        where_filter = None
        if doc_ids and len(doc_ids) == 1:
            where_filter = {"document_id": doc_ids[0]}
        elif doc_ids and len(doc_ids) > 1:
            where_filter = {"document_id": {"$in": doc_ids}}
        elif category:
            where_filter = {"category": category}

        total_count = self.collection.count()
        if total_count == 0:
            return []

        # Query vector store for candidates
        candidate_k = min(max(top_k * 3, 10), total_count)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=candidate_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        query_substantive = extract_substantive_tokens(query_text)

        # Use the original (pre-expansion) query for section-number mismatch detection.
        # This prevents "Section 420" injected by query expansion from blocking unrelated
        # but genuinely relevant chunks (e.g. BNS Section 318 for a "cheating" query).
        section_guard_query = original_query if original_query else query_text

        # Extract target article/section number from original query
        target_number = extract_provision_target(section_guard_query)

        # Also check if query is specifically about a schedule (e.g. "Second Schedule")
        query_asks_schedule = bool(re.search(
            r"(?i)(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+schedule",
            section_guard_query
        ))

        # --- Exact Metadata Matching Stage ---
        # When user explicitly queries for an article/section (e.g., 'Article 21', 'Article 21A'),
        # directly query ChromaDB metadata for exact article_number matches.
        exact_match_chunks: List[Dict[str, Any]] = []
        if target_number:
            exact_where: Dict[str, Any] = {"article_number": target_number}
            if doc_ids and len(doc_ids) == 1:
                exact_where = {"$and": [{"article_number": target_number}, {"document_id": doc_ids[0]}]}
            elif doc_ids and len(doc_ids) > 1:
                exact_where = {"$and": [{"article_number": target_number}, {"document_id": {"$in": doc_ids}}]}

            try:
                exact_res = self.collection.get(
                    where=exact_where,
                    include=["documents", "metadatas"]
                )
                if exact_res and exact_res.get("ids"):
                    for e_id, e_doc, e_meta in zip(exact_res["ids"], exact_res["documents"], exact_res["metadatas"]):
                        # Ensure schedule chunk is not returned unless specifically asked
                        is_sched = str(e_meta.get("is_schedule", "False")).lower() in ("true", "1", "yes")
                        if is_sched and not query_asks_schedule:
                            continue
                        exact_match_chunks.append({
                            "chunk_id": e_id,
                            "text": e_doc,
                            "metadata": e_meta,
                            "document_id": e_meta.get("document_id", ""),
                            "document_title": e_meta.get("document_title", ""),
                            "page_number": int(e_meta.get("page_number", 1)),
                            "section_header": e_meta.get("section_header", ""),
                            "article_number": e_meta.get("article_number", ""),
                            "category": e_meta.get("category", ""),
                            "relevance_score": 1.0,  # Highest confidence for exact metadata match
                        })
            except Exception as e:
                print(f"Notice: exact metadata match query skipped: {e}")

        # If exact metadata matches found, prioritize them
        if exact_match_chunks:
            return exact_match_chunks[:top_k]

        scored_chunks = []

        for cid, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # 1. Base vector similarity from cosine distance
            vector_sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))

            # 2. Substantive lexical token overlap
            chunk_tokens = extract_substantive_tokens(
                doc_text + " " + meta.get("section_header", "") + " " + meta.get("document_title", "")
            )
            overlap_tokens = query_substantive.intersection(chunk_tokens)
            lexical_ratio = (len(overlap_tokens) / len(query_substantive)) if query_substantive else 0.0

            # 3. Guardrail: If substantive terms have zero overlap, the chunk is unrelated!
            if len(overlap_tokens) == 0:
                continue

            # 4. Schedule/header-only penalty: query is for a main article but chunk is from a Schedule/Appendix
            is_schedule_chunk = str(meta.get("is_schedule", "False")).lower() in ("true", "1", "yes")
            if is_schedule_chunk and not query_asks_schedule and target_number:
                continue

            # Also check text content for schedule headings
            if not query_asks_schedule and target_number and len(doc_text.strip()) < 120 and "schedule" in doc_text.lower():
                continue

            # 5. Provision mismatch guardrail using article_number metadata and header extraction
            if target_number:
                chunk_article_num = meta.get("article_number", "").lower().strip()
                chunk_sec_header = meta.get("section_header", "").lower().strip()

                if chunk_article_num:
                    # Exact article_number metadata mismatch check
                    # Guarantees e.g. "21a" is never confused with "21", and "22" never matches "21"
                    if chunk_article_num != target_number and target_number not in chunk_tokens:
                        continue
                else:
                    # Fallback: extract provision from section_header using bare-number and prefix patterns
                    header_prov = extract_header_provision(chunk_sec_header)
                    if header_prov and header_prov != target_number and target_number not in chunk_tokens:
                        continue
                    # If target_number is specified (e.g. "999", "21") but chunk has neither an article_number
                    # nor a header provision matching target_number AND target_number is NOT in chunk_tokens,
                    # this chunk is an unrelated fragment (e.g., page header or preamble fragment) -> reject!
                    if not header_prov and target_number not in chunk_tokens:
                        continue

            # 6. Hybrid blended score (55% lexical grounding + 45% vector similarity)
            hybrid_score = (0.55 * lexical_ratio) + (0.45 * vector_sim)

            # Bonus: exact target number matches article_number metadata field
            chunk_article_num = meta.get("article_number", "").lower().strip()
            if target_number and chunk_article_num == target_number:
                hybrid_score = min(1.0, hybrid_score + 0.35)  # strong boost for exact match

            # Bonus: target number found in section_header text
            elif target_number and target_number in meta.get("section_header", "").lower():
                hybrid_score = min(1.0, hybrid_score + 0.25)

            if hybrid_score < min_relevance_threshold:
                continue

            scored_chunks.append({
                "chunk_id": cid,
                "text": doc_text,
                "metadata": meta,
                "document_id": meta.get("document_id", ""),
                "document_title": meta.get("document_title", ""),
                "page_number": int(meta.get("page_number", 1)),
                "section_header": meta.get("section_header", ""),
                "article_number": meta.get("article_number", ""),
                "category": meta.get("category", ""),
                "relevance_score": round(hybrid_score, 4),
            })

        # Rank descending by score
        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_chunks[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        try:
            self.collection.delete(where={"document_id": doc_id})
            if doc_id in self._doc_registry:
                del self._doc_registry[doc_id]
                self._save_registry()
            return True
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self._doc_registry.values())

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._doc_registry.get(doc_id)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "total_documents": len(self._doc_registry),
            "persist_dir": self.persist_dir,
        }

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "NyayaSathi Legal Document Corpus", "hnsw:space": "cosine"}
        )
        self._doc_registry = {}
        self._save_registry()


# Global instance
vector_service = ChromaVectorService()
