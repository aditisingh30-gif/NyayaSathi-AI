from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== DOCUMENT SCHEMAS ====================

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_number: int
    section_header: str = ""
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    title: str
    category: str = "General Legal"  # Act, Statutory Policy, Contract, Court Judgment, Compliance
    num_pages: int
    num_chunks: int
    file_size_bytes: int
    uploaded_at: str
    summary: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    title: str
    num_pages: int
    num_chunks: int
    status: str = "success"
    message: str


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentMetadata]


# ==================== RAG & QUERY SCHEMAS ====================

class Citation(BaseModel):
    document_id: str
    document_title: str
    page_number: int
    section_header: str = ""
    text_snippet: str
    relevance_score: float = 0.0


class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    category: Optional[str] = None
    top_k: int = 5
    include_disclaimer: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_score: float
    query: str
    documents_consulted: List[str]
    disclaimer: str
    latency_seconds: float = 0.0


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    citations: Optional[List[Citation]] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    doc_ids: Optional[List[str]] = None
    top_k: int = 5


# ==================== CLAUSE & POLICY ANALYSIS SCHEMAS ====================

class ClauseAnalysisRequest(BaseModel):
    clause_text: str
    contract_type: str = "Commercial Contract"  # SaaS, Employment, Lease, NDA, Vendor, General
    context: Optional[str] = None


class ClauseRiskItem(BaseModel):
    risk_title: str
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    risk_summary: str
    legal_issue: str
    potential_consequence: str
    recommended_revision: str
    statutory_reference: Optional[str] = None


class ClauseAnalysisResponse(BaseModel):
    original_clause: str
    contract_type: str
    overall_risk_score: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_factors: List[ClauseRiskItem]
    improved_clause_draft: str
    compliance_notes: List[str]
    disclaimer: str


class PolicyComplianceRequest(BaseModel):
    doc_id: Optional[str] = None
    policy_text: Optional[str] = None
    framework: str = "DPDP_ACT_2023"  # DPDP_ACT_2023, IT_ACT_2000, CONSUMER_PROTECTION_2019, LABOUR_CODES


class ComplianceCheckItem(BaseModel):
    framework_requirement: str
    status: str  # COMPLIANT, NON_COMPLIANT, PARTIAL_RISK, NOT_APPLICABLE
    finding: str
    section_reference: str
    remediation_step: str


class PolicyComplianceResponse(BaseModel):
    framework: str
    title: str
    overall_compliance_percentage: int
    checks: List[ComplianceCheckItem]
    executive_summary: str
    disclaimer: str


# ==================== SYSTEM & HEALTH SCHEMAS ====================

class SystemStats(BaseModel):
    status: str = "healthy"
    app_name: str
    version: str
    total_documents: int
    total_chunks: int
    chroma_collection: str
    gemini_configured: bool
    active_model: str
    uptime_seconds: float = 0.0
