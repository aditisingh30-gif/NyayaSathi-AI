import time
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.vector_service import vector_service, ChromaVectorService
from app.services.gemini_service import gemini_service, GeminiLegalService
from app.models.schemas import QueryRequest, QueryResponse, Citation

# Legal concept map for query expansion (Indian & Commercial Law)
LEGAL_EXPANSIONS = {
    "cheating": "Section 420 Indian Penal Code / Section 318 Bharatiya Nyaya Sanhita (BNS) fraud dishonestly inducing delivery of property",
    "fraud": "Section 17 Indian Contract Act / Section 318 Bharatiya Nyaya Sanhita deceit misrepresentation",
    "data breach": "Section 8 Digital Personal Data Protection Act 2023 security safeguards breach notification",
    "privacy": "Digital Personal Data Protection Act 2023 Data Principal rights consent notice",
    "contract breach": "Section 73 Section 74 Indian Contract Act 1872 compensation for loss damages",
    "termination": "Termination for convenience material breach cure period notice rights",
    "indemnity": "Section 124 Indian Contract Act 1872 contract of indemnity hold harmless liability cap",
    "non-compete": "Section 27 Indian Contract Act 1872 agreement in restraint of trade void",
    "consumer": "Consumer Protection Act 2019 deficiency of service unfair trade practice right to refund",
    "arrest": "Bharatiya Nagarik Suraksha Sanhita (BNSS) Section 35 arrest rights bail grounds",
}


class RAGLegalService:
    """
    Orchestrates legal query expansion, vector retrieval from ChromaDB,
    statutory context synthesis via Gemini, and citation formatting.
    """

    def __init__(
        self,
        v_service: Optional[ChromaVectorService] = None,
        g_service: Optional[GeminiLegalService] = None,
    ):
        self.vector_service = v_service or vector_service
        self.gemini_service = g_service or gemini_service

    def expand_query(self, raw_query: str) -> str:
        """Enrich user question with formal legal terminology if relevant keywords exist."""
        lower_q = raw_query.lower()
        expansions = []
        for term, legal_kw in LEGAL_EXPANSIONS.items():
            if term in lower_q:
                expansions.append(legal_kw)

        if expansions:
            return f"{raw_query} {' '.join(expansions[:2])}"
        return raw_query

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute full RAG retrieval and legal response generation."""
        start_time = time.time()

        # 1. Expand query
        expanded_q = self.expand_query(request.question)

        # 2. Vector retrieval from ChromaDB with strict relevance guardrails
        chunks = self.vector_service.query_similar(
            query_text=expanded_q,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
            category=request.category,
            min_relevance_threshold=0.35,
            original_query=request.question,  # Pre-expansion query for section-number guard
        )

        # 3. If no chunks passed the guardrails, return a clear "source unavailable" message
        #    instead of letting Gemini hallucinate citations from an empty context.
        if not chunks:
            latency = round(time.time() - start_time, 3)
            source_unavailable_answer = (
                "⚖️ **Source Not Available in Knowledge Base**\n\n"
                f"The NyayaSathi knowledge base does not currently contain the relevant legal provisions "
                f"to answer your question about *\"{request.question}\"*.\n\n"
                "**What this means**: No matching statutes, sections, or clauses were found that are "
                "substantively related to your query. The system will never fabricate or guess citations.\n\n"
                "**To get an answer**, please upload the relevant Act or document "
                "(e.g., Constitution of India PDF) via the **Document Library** tab, then ask again.\n\n"
                "**Currently indexed documents**: Digital Personal Data Protection Act 2023 · "
                "Indian Contract Act 1872 · Bharatiya Nyaya Sanhita 2023 · Enterprise Cloud AI MSA\n\n"
                "_NyayaSathi is committed to accuracy: only verified provisions from indexed documents are presented._"
            )
            return QueryResponse(
                answer=source_unavailable_answer,
                citations=[],
                confidence_score=0.0,
                query=request.question,
                documents_consulted=[],
                disclaimer=settings.LEGAL_DISCLAIMER if request.include_disclaimer else "",
                latency_seconds=latency,
            )

        # 4. Gemini LLM Synthesis (only reached when relevant chunks exist)
        synth_result = self.gemini_service.generate_rag_response(
            question=request.question,
            retrieved_chunks=chunks,
        )

        latency = round(time.time() - start_time, 3)

        # 5. Extract distinct documents consulted
        docs_consulted = list(dict.fromkeys(
            c.get("document_title") for c in chunks if c.get("document_title")
        ))

        citations = [Citation(**c) for c in synth_result.get("citations", [])]

        return QueryResponse(
            answer=synth_result.get("answer", ""),
            citations=citations,
            confidence_score=synth_result.get("confidence_score", 0.0),
            query=request.question,
            documents_consulted=docs_consulted,
            disclaimer=settings.LEGAL_DISCLAIMER if request.include_disclaimer else "",
            latency_seconds=latency,
        )


# Global instance
rag_service = RAGLegalService()
