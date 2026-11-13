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
    indexing_status: str = "indexed"  # "indexed", "failed", "processing"
    error_message: Optional[str] = None
    storage_path: Optional[str] = None
    source: str = "local"  # "supabase", "local", "benchmark"


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


class DocumentSyncResponse(BaseModel):
    status: str
    synced_count: int
    failed_count: int
    already_indexed_count: int
    total_supabase_files: int
    message: str
    errors: Optional[List[str]] = None


class SyncStatusResponse(BaseModel):
    is_syncing: bool
    last_sync_at: Optional[str] = None
    last_sync_result: Optional[Dict[str, Any]] = None


class DocumentRetryResponse(BaseModel):
    status: str
    doc_id: str
    title: Optional[str] = None
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None
    message: str


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
    control_name: Optional[str] = None
    framework_requirement: str
    status: str  # COMPLIANT, PARTIALLY_COMPLIANT, NON_COMPLIANT
    finding: str
    evidence_extracted: Optional[str] = "No evidence found in submitted policy."
    missing_requirements: Optional[str] = "None identified."
    section_reference: str
    commencement_status: Optional[str] = (
        "Enacted (Sections 3–5, 6(1)–(8), 6(10), and 7–17 scheduled to commence on 13 May 2027; "
        "Section 6(9) scheduled to commence tomorrow, 13 November 2026, under G.S.R. 843(E))"
    )
    remediation_step: str
    points_awarded: Optional[int] = 0
    max_points: Optional[int] = 100


class PolicyComplianceResponse(BaseModel):
    framework: str
    title: str
    overall_compliance_percentage: int
    controls_assessed: Optional[int] = 4
    commencement_notice: Optional[str] = None
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
