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

GENERIC_LEGAL_FRAMEWORK_WORDS = {
    "procedure", "manner", "filing", "file", "filed", "process", "steps",
    "section", "sections", "article", "articles", "clause", "clauses",
    "act", "provision", "provisions", "law", "statute", "rule", "rules",
    "mention", "relevant", "state", "under", "court", "matter", "case",
    "india", "indian", "national", "central", "say", "says", "tell", "explain",
    "provide", "provides", "about", "what", "does"
}


def extract_substantive_tokens(text: str) -> Set[str]:
    """Extract substantive legal terms, numbers, and identifiers, discarding stop words."""
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return {w for w in words if w not in LEGAL_STOP_WORDS and len(w) > 1}


def extract_target_provision_details(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract statutory prefix ('section', 'article', 'clause') and provision number
    from query text, preserving alphanumeric suffixes (e.g. '21a', '35').
    """
    m = re.findall(r"\b(section|article|clause)\s+(\d+[a-z]?)\b", text, re.I)
    if m:
        return m[0][0].lower(), m[0][1].lower()
    return None, None


def extract_provision_target(text: str) -> Optional[str]:
    """
    Extract statutory section, article, or clause target number from text.
    Handles 'Article 21', 'Article 21A', 'Section 73', 'Clause 9'.
    """
    _, num = extract_target_provision_details(text)
    return num


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


def detect_legal_intent(query_str: str) -> Dict[str, bool]:
    """
    Classify semantic legal intent of a query (procedural, definition, authority, etc.).
    """
    q = query_str.lower()

    # Procedural intent: filing, procedure, manner, steps, where to file, how to complain
    is_procedural = bool(re.search(
        r"\b(procedure|manner|filing|file|how\s+to\s+file|how\s+to\s+make|make\s+a\s+complaint|how\s+to\s+complain|steps|process|e-daakhil|prescribed\s+fee|admissibility|admission|where\s+to\s+file)\b",
        q
    ))

    # Fine-grained procedural stages:
    # 1. Manner of filing (Section 35)
    is_filing_manner = bool(re.search(
        r"\b(manner\s+in\s+which|manner\s+of\s+filing|how\s+to\s+file|who\s+can\s+file|who\s+may\s+file|prescribed\s+fee|e-daakhil|electronic\s+filing|make\s+a\s+complaint)\b",
        q
    ))

    # 2. Admissibility and 21-day timeline (Section 36)
    is_admissibility = bool(re.search(
        r"\b(admissibility|twenty-one\s+days|21\s+days|deemed\s+admitted|deemed\s+admission|admit\s+the\s+complaint|reject\s+the\s+same|opportunity\s+of\s+being\s+heard|admitted\s+within)\b",
        q
    ))

    # 3. Post-admission procedure & opposite party notice/reply (Section 38)
    is_post_admission = bool(re.search(
        r"\b(after\s+admission|procedure\s+on\s+admission|post\s+admission|opposite\s+party|thirty\s+days|30\s+days|version\s+of\s+the\s+case|refer\s+a\s+copy|settle\s+the\s+consumer\s+dispute)\b",
        q
    ))

    # Definition intent: what is, definition, define, meaning, defined as, who is
    is_definition = bool(re.search(
        r"\b(definition|definitions|define|defined|meaning|what\s+is\s+a\s+consumer|who\s+is\s+a\s+consumer|what\s+is\s+a\s+complaint|what\s+constitutes|what\s+is\s+meant\s+by)\b",
        q
    ))

    # Institutional / Regulatory Authority: CCPA, Central Authority, establishment
    is_authority = bool(re.search(
        r"\b(central\s+authority|central\s+consumer\s+protection\s+authority|ccpa|regulator|establishment\s+of\s+(?:the\s+)?central)\b",
        q
    ))

    return {
        "is_procedural": is_procedural,
        "is_filing_manner": is_filing_manner,
        "is_admissibility": is_admissibility,
        "is_post_admission": is_post_admission,
        "is_definition": is_definition,
        "is_authority": is_authority,
    }


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

        ids = results["ids"][0] if (results and results.get("ids") and results["ids"][0]) else []
        docs = results["documents"][0] if (results and results.get("documents") and results["documents"][0]) else []
        metas = results["metadatas"][0] if (results and results.get("metadatas") and results["metadatas"][0]) else []
        distances = results["distances"][0] if (results and results.get("distances") and results["distances"][0]) else [0.0] * len(ids)

        query_substantive = extract_substantive_tokens(query_text)

        # Use the original (pre-expansion) query for section-number mismatch detection.
        # This prevents "Section 420" injected by query expansion from blocking unrelated
        # but genuinely relevant chunks (e.g. BNS Section 318 for a "cheating" query).
        section_guard_query = original_query if original_query else query_text
        guard_substantive = extract_substantive_tokens(section_guard_query)
        combined_substantive = query_substantive.union(guard_substantive)

        # Extract statutory prefix ('section', 'article') and number ('35', '21')
        target_prefix, target_number = extract_target_provision_details(section_guard_query)
        if not target_number:
            target_number = extract_provision_target(section_guard_query)

        # Also check if query is specifically about a schedule (e.g. "Second Schedule")
        query_asks_schedule = bool(re.search(
            r"(?i)(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+schedule",
            section_guard_query
        ))

        # Detect legal subject domain from query text
        is_consumer_query = bool(re.search(
            r"\b(consumer|deficiency|complaint|complainant|district\s+commission|state\s+commission|national\s+commission|ccpa|e-daakhil|cpa)\b",
            section_guard_query, re.I
        ))
        is_constitution_query = bool(re.search(
            r"\b(constitution|fundamental\s+right|writs?|habeas|mandamus|article\s+\d+|preamble)\b",
            section_guard_query, re.I
        ))
        is_contract_query = bool(re.search(
            r"\b(contract|agreement|consideration|breach\s+of\s+contract|voidable|indemnity|guarantee)\b",
            section_guard_query, re.I
        ))
        is_criminal_query = bool(re.search(
            r"\b(penal|crime|criminal|bns|ipc|bnss|crpc|cheating|theft|murder|arrest|bail)\b",
            section_guard_query, re.I
        ))
        is_data_privacy_query = bool(re.search(
            r"\b(dpdp|data\s+protection|privacy|data\s+fiduciary|data\s+principal)\b",
            section_guard_query, re.I
        ))
        has_statute_domain = (
            is_consumer_query or is_constitution_query or is_contract_query
            or is_criminal_query or is_data_privacy_query
        )

        # Substantive subject-matter terms excluding generic legal framework terminology and provision numbers
        query_subject_terms = {
            t for t in guard_substantive
            if t not in GENERIC_LEGAL_FRAMEWORK_WORDS and not t.isdigit() and (not target_number or t != target_number)
        }

        # Ambiguous bare provision query guard:
        # If the user asks only e.g. "What does Section 35 say?" without any statute domain or subject terms,
        # we must NOT arbitrarily guess an unrelated document; return empty so RAG reports Source Not Available.
        if target_number and len(query_subject_terms) == 0 and not has_statute_domain:
            return []

        # Collect candidate chunks: combine vector results with exact metadata matches
        candidate_map: Dict[str, Tuple[str, Dict[str, Any], float]] = {}
        for cid, doc_text, meta, dist in zip(ids, docs, metas, distances):
            candidate_map[cid] = (doc_text, meta, float(dist))

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
                        if e_id not in candidate_map:
                            candidate_map[e_id] = (e_doc, e_meta, 0.0)
            except Exception as e:
                print(f"Notice: exact metadata match query skipped: {e}")

        if not candidate_map:
            return []

        # Classify legal intent of query
        query_intent = detect_legal_intent(section_guard_query)
        if not any(query_intent.values()):
            query_intent = detect_legal_intent(query_text)

        scored_chunks = []

        for cid, (doc_text, meta, dist) in candidate_map.items():
            doc_title = meta.get("document_title", "").lower()
            category = meta.get("category", "").lower()
            chunk_sec_header = meta.get("section_header", "").lower().strip()
            chunk_article_num = meta.get("article_number", "").lower().strip()
            chunk_text_lower = doc_text.lower()

            chunk_is_consumer = any(w in doc_title or w in category for w in ("consumer", "cpa"))
            chunk_is_constitution = any(w in doc_title or w in category for w in ("constitution",)) or "the constitution of india" in doc_title
            chunk_is_contract = any(w in doc_title or w in category for w in ("contract",))
            chunk_is_criminal = any(w in doc_title or w in category for w in ("penal", "bns", "bnss", "criminal"))
            chunk_is_data_privacy = any(w in doc_title or w in category for w in ("dpdp", "data protection"))

            # Guard 1: Legal Subject Domain Mismatch Guardrails
            # A consumer query must NEVER return Constitution, Contract, or Criminal documents.
            # A Constitution query must NEVER return Consumer or Contract statutes.
            if is_consumer_query and not chunk_is_consumer:
                continue
            if is_constitution_query and not chunk_is_constitution:
                continue
            if is_contract_query and not chunk_is_contract:
                continue
            if is_criminal_query and not chunk_is_criminal:
                continue
            if is_data_privacy_query and not chunk_is_data_privacy:
                continue

            # Guard 2: Statutory Prefix Mismatch Guardrail ('Section' vs 'Article')
            # Statutory Acts have Sections, while the Constitution of India has Articles.
            # Never treat Constitution Article 35 as Section 35 of an Act.
            if target_prefix == "section":
                if chunk_is_constitution or chunk_sec_header.startswith("article "):
                    continue
            elif target_prefix == "article":
                if (chunk_is_consumer or chunk_is_contract or chunk_is_criminal or
                        chunk_sec_header.startswith("section ")):
                    continue

            # 1. Base vector similarity from cosine distance
            vector_sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))

            # 2. Substantive lexical token overlap
            chunk_tokens = extract_substantive_tokens(
                doc_text + " " + meta.get("section_header", "") + " " + meta.get("document_title", "")
            )
            overlap_tokens = combined_substantive.intersection(chunk_tokens)
            lexical_ratio = (len(overlap_tokens) / len(combined_substantive)) if combined_substantive else 0.0

            # 3. Guardrail: If substantive terms have zero overlap, the chunk is unrelated!
            if len(overlap_tokens) == 0:
                continue

            # 3b. Subject matter grounding: If user query contains specific subject/domain terms,
            # ensure at least one subject term matches the chunk rather than only generic procedural words.
            if query_subject_terms and len(query_subject_terms.intersection(chunk_tokens)) == 0:
                continue

            # 4. Schedule/header-only penalty: query is for a main article but chunk is from a Schedule/Appendix
            is_schedule_chunk = str(meta.get("is_schedule", "False")).lower() in ("true", "1", "yes")
            if is_schedule_chunk and not query_asks_schedule and target_number:
                continue

            # Also check text content for schedule headings
            if not query_asks_schedule and target_number and len(doc_text.strip()) < 120 and "schedule" in doc_text.lower():
                continue

            # 5. Provision mismatch guardrail using article_number metadata and header extraction
            header_prov = extract_header_provision(chunk_sec_header)
            matches_target_number = (
                (chunk_article_num and chunk_article_num == target_number)
                or (header_prov and header_prov == target_number)
                or (target_number and target_number in chunk_tokens)
            )

            if target_number:
                if not matches_target_number:
                    # Procedural multi-section allowance: For procedural consumer queries,
                    # related procedural sections (35, 36, 38) are necessary to answer the procedural flow.
                    if query_intent["is_procedural"] and chunk_is_consumer:
                        is_cpa_procedural = (
                            chunk_article_num in ("35", "36", "38")
                            or any(s in chunk_sec_header for s in ("section 35", "section 36", "section 38", "manner", "proceedings before district commission", "procedure on admission"))
                        )
                        if not is_cpa_procedural:
                            continue
                    else:
                        continue

            # 6. Hybrid blended score (55% lexical grounding + 45% vector similarity)
            hybrid_score = (0.55 * lexical_ratio) + (0.45 * vector_sim)

            # Bonus: exact target number matches article_number metadata field
            if target_number and chunk_article_num == target_number:
                hybrid_score = min(1.0, hybrid_score + 0.35)  # strong boost for exact match
            # Bonus: target number found in section_header text
            elif target_number and target_number in chunk_sec_header:
                hybrid_score = min(1.0, hybrid_score + 0.25)

            # 7. Intent-aware semantic reranking
            chunk_is_procedural = (
                any(p in chunk_sec_header for p in ("manner", "procedure", "proceedings", "filing", "admission"))
                or any(p in chunk_text_lower for p in ("manner in which complaint shall be made", "may be filed with a district commission", "electronic filing of complaints", "e-daakhil", "accompanied with such fee", "admissibility of the complaint", "refer a copy of the admitted complaint", "opposite party"))
            )
            chunk_is_definition = (
                "definition" in chunk_sec_header
                or chunk_article_num == "2"
                or chunk_sec_header.startswith("section 2")
                or "unless the context otherwise requires" in chunk_text_lower[:120]
            )
            chunk_is_authority = (
                "central authority" in chunk_sec_header
                or "central consumer protection authority" in chunk_sec_header
                or chunk_article_num == "10"
                or "establishment of central consumer protection authority" in chunk_text_lower
            )

            # Section-specific markers for Consumer Protection Act
            is_sec35 = chunk_article_num == "35" or "section 35" in chunk_sec_header or "manner in which complaint shall be made" in chunk_sec_header
            is_sec36 = chunk_article_num == "36" or "section 36" in chunk_sec_header or "proceedings before district commission" in chunk_sec_header or "admissibility" in chunk_text_lower
            is_sec38 = chunk_article_num == "38" or "section 38" in chunk_sec_header or "procedure on admission" in chunk_sec_header or "opposite party" in chunk_text_lower

            if query_intent["is_procedural"]:
                if chunk_is_procedural:
                    hybrid_score = min(1.0, hybrid_score + 0.35)

                    # Fine-grained sub-intent weighting
                    if query_intent["is_filing_manner"] and is_sec35:
                        hybrid_score = min(1.0, hybrid_score + 0.20)
                    elif query_intent["is_admissibility"] and is_sec36:
                        hybrid_score = min(1.0, hybrid_score + 0.25)
                    elif query_intent["is_post_admission"] and is_sec38:
                        hybrid_score = min(1.0, hybrid_score + 0.25)
                    elif is_sec35:
                        hybrid_score = min(1.0, hybrid_score + 0.15)
                    elif is_sec36:
                        hybrid_score = min(1.0, hybrid_score + 0.10)
                    elif is_sec38:
                        hybrid_score = min(1.0, hybrid_score + 0.08)

                elif chunk_is_definition and not query_intent["is_definition"]:
                    hybrid_score = max(0.0, hybrid_score - 0.30)
                elif chunk_is_authority and not query_intent["is_authority"]:
                    hybrid_score = max(0.0, hybrid_score - 0.25)

            elif query_intent["is_definition"]:
                if chunk_is_definition:
                    hybrid_score = min(1.0, hybrid_score + 0.35)
                elif chunk_is_procedural:
                    hybrid_score = max(0.0, hybrid_score - 0.20)

            elif query_intent["is_authority"]:
                if chunk_is_authority:
                    hybrid_score = min(1.0, hybrid_score + 0.35)
                elif chunk_is_definition or chunk_is_procedural:
                    hybrid_score = max(0.0, hybrid_score - 0.20)

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
