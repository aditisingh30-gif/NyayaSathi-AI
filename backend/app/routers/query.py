from fastapi import APIRouter, HTTPException
from typing import List
from app.services.rag_service import rag_service
from app.models.schemas import QueryRequest, QueryResponse, ChatRequest, ChatMessage, Citation

router = APIRouter(prefix="/api/query", tags=["Legal Query & RAG"])


@router.post("", response_model=QueryResponse)
def query_legal_corpus(request: QueryRequest):
    """
    RAG-powered legal retrieval and synthesis.
    Queries ChromaDB vectors, retrieves grounded statutory/contractual context,
    and calls Gemini LLM with citation grounding.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Query question cannot be empty.")

    try:
        response = rag_service.query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legal query processing failed: {str(e)}")


@router.post("/chat")
def chat_with_legal_assistant(request: ChatRequest):
    """
    Conversational multi-turn legal assistant endpoint.
    Maintains dialogue history while retrieving context for the latest user prompt.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Message list cannot be empty.")

    latest_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not latest_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in dialogue.")

    # Execute RAG on latest message
    query_req = QueryRequest(
        question=latest_user_msg,
        doc_ids=request.doc_ids,
        top_k=request.top_k,
    )
    rag_resp = rag_service.query(query_req)

    return {
        "reply": rag_resp.answer,
        "citations": rag_resp.citations,
        "confidence_score": rag_resp.confidence_score,
        "disclaimer": rag_resp.disclaimer,
    }
