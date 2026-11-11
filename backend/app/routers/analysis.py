from fastapi import APIRouter, HTTPException
from app.config import settings
from app.services.gemini_service import gemini_service
from app.services.vector_service import vector_service
from app.models.schemas import (
    ClauseAnalysisRequest,
    ClauseAnalysisResponse,
    PolicyComplianceRequest,
    PolicyComplianceResponse,
    ClauseRiskItem,
    ComplianceCheckItem,
)

router = APIRouter(prefix="/api/analysis", tags=["Contract & Policy Analysis"])


@router.post("/clause", response_model=ClauseAnalysisResponse)
def analyze_contract_clause(request: ClauseAnalysisRequest):
    """
    Evaluates a specific contract clause for legal risks (indemnity, termination, liability caps, non-compete, etc.)
    and suggests revised draft language.
    """
    if not request.clause_text.strip():
        raise HTTPException(status_code=400, detail="Clause text cannot be empty.")

    try:
        raw_result = gemini_service.analyze_clause(
            clause_text=request.clause_text,
            contract_type=request.contract_type,
        )

        risk_factors = [
            ClauseRiskItem(**rf) for rf in raw_result.get("risk_factors", [])
        ]

        return ClauseAnalysisResponse(
            original_clause=request.clause_text,
            contract_type=request.contract_type,
            overall_risk_score=raw_result.get("overall_risk_score", "MEDIUM"),
            risk_factors=risk_factors,
            improved_clause_draft=raw_result.get("improved_clause_draft", ""),
            compliance_notes=raw_result.get("compliance_notes", []),
            disclaimer=settings.LEGAL_DISCLAIMER,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clause analysis failed: {str(e)}")


@router.post("/policy", response_model=PolicyComplianceResponse)
def audit_policy_compliance(request: PolicyComplianceRequest):
    """
    Audits a policy or agreement against Indian statutory frameworks
    (e.g., DPDP Act 2023, IT Act 2000, Consumer Protection Rules).
    """
    text_to_audit = request.policy_text

    # If doc_id provided, fetch text from registered document
    if not text_to_audit and request.doc_id:
        doc = vector_service.get_document_by_id(request.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Specified document not found.")
        # Retrieve chunks for this doc
        chunks = vector_service.query_similar("", top_k=20, doc_ids=[request.doc_id])
        text_to_audit = "\n\n".join(c["text"] for c in chunks)

    if not text_to_audit or not text_to_audit.strip():
        raise HTTPException(status_code=400, detail="Please provide either policy_text or a valid doc_id.")

    try:
        audit_result = gemini_service.audit_policy(
            text=text_to_audit,
            framework=request.framework,
        )

        checks = [
            ComplianceCheckItem(**c) for c in audit_result.get("checks", [])
        ]

        return PolicyComplianceResponse(
            framework=audit_result.get("framework", request.framework),
            title=audit_result.get("title", f"{request.framework} Compliance Audit"),
            overall_compliance_percentage=audit_result.get("overall_compliance_percentage", 50),
            checks=checks,
            executive_summary=audit_result.get("executive_summary", ""),
            disclaimer=audit_result.get("disclaimer", settings.LEGAL_DISCLAIMER),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy compliance audit failed: {str(e)}")
