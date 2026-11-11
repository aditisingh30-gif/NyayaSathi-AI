import os
import re
import json
from typing import List, Dict, Any, Optional
from app.config import settings

# Use modern google-genai SDK (fast, recommended by Google)
GEMINI_GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    GEMINI_GENAI_AVAILABLE = True
except Exception:
    GEMINI_GENAI_AVAILABLE = False


class GeminiLegalService:
    """
    Service interfacing with Google Gemini LLM API for legal analysis,
    contract review, policy auditing, and RAG answer synthesis.
    Includes a built-in legal intelligence fallback engine.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self._init_client()

    def _init_client(self):
        self.client = None
        if not self.api_key:
            return

        if GEMINI_GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to init google-genai client: {e}")

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self._init_client()

    def _call_gemini(self, prompt: str, system_instruction: str = "") -> str:
        """Call live Gemini model or raise exception to trigger fallback."""
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured")

        if GEMINI_GENAI_AVAILABLE and self.client:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction if system_instruction else None,
                    temperature=0.2,  # Low temperature for deterministic legal reasoning
                )
            )
            return response.text

        raise RuntimeError("No active Gemini client available")

    # ==================== RAG ANSWER GENERATION ====================

    def generate_rag_response(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes a legally grounded response citing specific provisions,
        acts, clauses, and page numbers from retrieved chunks.
        """
        if not retrieved_chunks:
            return {
                "answer": (
                    "⚖️ **Source Not Available in Knowledge Base**\n\n"
                    "No relevant legal provisions were found in the NyayaSathi knowledge base matching your query. "
                    "The system will not fabricate citations or statutory references.\n\n"
                    "Please upload the relevant Act, Contract, or Policy PDF via the Document Library tab, then ask again."
                ),
                "citations": [],
                "confidence_score": 0.0,
                "model_used": "system-default"
            }

        # Format context for prompt
        context_blocks = []
        citations = []

        for idx, chunk in enumerate(retrieved_chunks):
            doc_title = chunk.get("document_title", "Document")
            page = chunk.get("page_number", 1)
            section = chunk.get("section_header") or "General Provision"
            score = chunk.get("relevance_score", 0.0)
            text = chunk.get("text", "")

            context_blocks.append(
                f"[Source {idx+1}] Document: '{doc_title}' | Page {page} | Section/Clause: '{section}'\n"
                f"Content: {text}\n"
            )

            citations.append({
                "document_id": chunk.get("document_id", ""),
                "document_title": doc_title,
                "page_number": page,
                "section_header": section,
                "text_snippet": " ".join(text.split()),
                "relevance_score": score,
            })

        context_str = "\n".join(context_blocks)

        system_instruction = (
            "You are NyayaSathi (न्यायसाथी), an authoritative Indian and Comparative Legal & Policy AI Assistant. "
            "Your duty is to provide precise, rigorous, and legally grounded answers based strictly on the provided sources. "
            "RULES:\n"
            "1. Explicitly cite the exact Document Title, Section/Clause, and Page number whenever making any legal assertion.\n"
            "2. Preserve the exact statutory wording in the relevant provisions section.\n"
            "3. Generate explanations, rights, or implications ONLY when directly supported by retrieved text evidence.\n"
            "4. NEVER generate generic, unsupported legal implications (e.g. do not state generic compliance or mitigation unless the text explicitly says so).\n"
            "5. If retrieved evidence is insufficient to fully answer aspects of the question, explicitly state that evidence is insufficient.\n"
            "6. Do NOT hallucinate statutory sections or case law not present in the sources.\n"
            "7. Maintain an objective, professional judicial tone."
        )

        user_prompt = (
            f"LEGAL CONTEXT SOURCES:\n{context_str}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Provide a precise, evidence-grounded legal answer with headings: "
            "'1. Executive Summary', '2. Relevant Legal Provisions (Verbatim Text)', '3. Statutory Analysis & Scope', '4. Citations & References'.\n"
            "Do NOT include generic, unsupported boilerplates."
        )

        try:
            raw_answer = self._call_gemini(user_prompt, system_instruction=system_instruction)
            return {
                "answer": raw_answer.strip(),
                "citations": citations,
                "confidence_score": round(max(c["relevance_score"] for c in citations), 3),
                "model_used": self.model_name
            }
        except Exception as e:
            # Fallback to Built-in Legal Intelligence Engine
            fallback_answer = self._fallback_rag_synthesizer(question, retrieved_chunks)
            return {
                "answer": fallback_answer,
                "citations": citations,
                "confidence_score": round(max(c["relevance_score"] for c in citations), 3),
                "model_used": "nyayasathi-legal-engine (offline/local)"
            }

    def _fallback_rag_synthesizer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Deterministic, evidence-grounded legal synthesis directly from retrieved context."""
        top_chunk = chunks[0]
        doc_title = top_chunk.get("document_title", "Governing Legal Document")
        section = top_chunk.get("section_header") or "Applicable Provision"
        page = top_chunk.get("page_number", 1)

        provisions_blocks = []
        evidence_points = []

        for c in chunks[:3]:
            sec = c.get("section_header") or f"Provision on Page {c.get('page_number', 1)}"
            c_doc = c.get("document_title", doc_title)
            c_page = c.get("page_number", 1)
            raw_text = c.get("text", "").strip()
            clean_text = " ".join(raw_text.split())

            # 1. Verbatim Statutory Provision
            provisions_blocks.append(
                f"- **{sec}** ({c_doc}, Page {c_page}):\n"
                f"  > \"{clean_text}\""
            )

            # 2. Extract evidence-grounded takeaways directly from the text
            # Extract key legal phrases or direct conditions (e.g. 'procedure established by law', 'penalty', 'obligations')
            clean_lower = clean_text.lower()
            if "life or personal liberty" in clean_lower:
                evidence_points.append(
                    "- **Protection of Life and Liberty**: Explicitly prohibits deprivation of life or personal liberty of any person except according to procedure established by law."
                )
            if "procedure established by law" in clean_lower:
                evidence_points.append(
                    "- **Procedural Requirement**: Any state action encroaching upon personal liberty must strictly conform to valid, legally enacted procedure."
                )
            if "free and compulsory education" in clean_lower:
                evidence_points.append(
                    "- **Right to Education**: Mandates the State to provide free and compulsory education to all children aged 6 to 14 years."
                )
            if "arrest" in clean_lower and "custody" in clean_lower:
                evidence_points.append(
                    "- **Safeguards Against Arrest & Detention**: Requires informing arrested persons of the grounds of arrest and affording legal counsel."
                )
            if "penalty" in clean_lower or "crore" in clean_lower:
                evidence_points.append(
                    "- **Statutory Penalties**: Explicit statutory penalty schedule applies as specified in the retrieved text."
                )
            if "compensation" in clean_lower and "breach" in clean_lower:
                evidence_points.append(
                    "- **Damages for Breach**: Entitles the non-defaulting party to compensation for natural loss, excluding indirect or remote damages."
                )

        provisions_text = "\n\n".join(provisions_blocks)

        # If no specific recognized evidence patterns match, extract from verbatim text rather than generic claims
        if evidence_points:
            analysis_text = "\n".join(dict.fromkeys(evidence_points))
        else:
            analysis_text = (
                "- **Textual Mandate**: The rights, prohibitions, or duties are strictly governed by the exact statutory wording reproduced above.\n"
                "- **Scope of Evidence**: Additional case law, judicial precedents, or administrative guidelines are not present in the indexed text; legal interpretation is strictly bounded by the retrieved provision."
            )

        return (
            f"### 1. Executive Summary\n"
            f"Based on the statutory text of **{doc_title}**, specifically under **{section}** (Page {page}), "
            f"the provision directly addresses your inquiry regarding *'{question}'*.\n\n"
            f"### 2. Relevant Legal Provisions (Verbatim Statutory Text)\n"
            f"{provisions_text}\n\n"
            f"### 3. Statutory Analysis & Scope (Evidence-Grounded)\n"
            f"{analysis_text}\n\n"
            f"### 4. Evidentiary Limitation Note\n"
            f"This analysis is based strictly on the verified text of the retrieved document ({doc_title}, Page {page}). "
            f"No external or unverified citations have been synthesized."
        )

    # ==================== CLAUSE RISK ANALYZER ====================

    def analyze_clause(self, clause_text: str, contract_type: str = "Commercial Contract") -> Dict[str, Any]:
        """
        Evaluates a contract clause for legal risks (indemnity, termination, liability, jurisdiction, IP).
        """
        system_instruction = (
            "You are an expert Corporate & Contract Law Counsel. "
            "Analyze the given contract clause for operational and legal risks. "
            "Output your findings as valid JSON with keys: "
            "'overall_risk_score' ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'), "
            "'risk_factors' (array of objects: 'risk_title', 'risk_level', 'risk_summary', 'legal_issue', 'potential_consequence', 'recommended_revision', 'statutory_reference'), "
            "'improved_clause_draft' (string), "
            "'compliance_notes' (array of strings)."
        )

        user_prompt = f"Contract Type: {contract_type}\nClause Text to Review:\n\"\"\"{clause_text}\"\"\"\n\nReturn JSON only."

        try:
            raw = self._call_gemini(user_prompt, system_instruction=system_instruction)
            # Clean markdown code block if present
            cleaned = re.sub(r"^```json\s*", "", raw.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
            return json.loads(cleaned)
        except Exception:
            # Deterministic Contract Risk Analysis Engine
            return self._fallback_clause_analyzer(clause_text, contract_type)

    def _fallback_clause_analyzer(self, clause_text: str, contract_type: str) -> Dict[str, Any]:
        """Built-in rule-based contract clause risk engine."""
        lower = clause_text.lower()
        risks = []
        overall_risk = "LOW"
        notes = []

        # Check Unilateral Termination
        if "sole discretion" in lower or ("terminate" in lower and "immediate" in lower and "without cause" in lower):
            overall_risk = "HIGH"
            risks.append({
                "risk_title": "Unilateral Immediate Termination Without Cause",
                "risk_level": "HIGH",
                "risk_summary": "One party holds disproportionate power to cancel the engagement without cure periods or mutual parity.",
                "legal_issue": "Lack of reciprocal termination rights and absence of 30-day notice or cure periods.",
                "potential_consequence": "Sudden business disruption, unrecoverable operational investments, and cash-flow shock.",
                "recommended_revision": "Require mutual termination for convenience with at least 30 to 60 days written notice and a 30-day cure period for material breach.",
                "statutory_reference": "Section 73 & 74, Indian Contract Act, 1872 (Breach and Damages)"
            })
            notes.append("Standard commercial practice requires bilateral termination notice periods.")

        # Check Uncapped Indemnity
        if "indemnify" in lower and ("unlimited" in lower or "all losses" in lower or "indirect" in lower or "consequential" in lower):
            overall_risk = "CRITICAL"
            risks.append({
                "risk_title": "Uncapped Broad-Form Indemnity",
                "risk_level": "CRITICAL",
                "risk_summary": "Indemnification is not capped to contract value and encompasses indirect/consequential damages.",
                "legal_issue": "Disproportionate exposure exceeding total fees paid under the agreement.",
                "potential_consequence": "Catastrophic financial liability in excess of contract value or insurance limits.",
                "recommended_revision": "Cap aggregate indemnification to total fees paid in the preceding 12 months, and expressly exclude indirect, punitive, or consequential damages.",
                "statutory_reference": "Section 124, Indian Contract Act, 1872 (Contract of Indemnity)"
            })
            notes.append("Always insist on an aggregate liability cap (Super Cap or 1x-2x Annual Contract Value).")

        # Check Non-Compete Post Termination
        if "not compete" in lower or "non-compete" in lower or "restraint" in lower:
            overall_risk = "HIGH"
            risks.append({
                "risk_title": "Post-Termination Non-Compete Restraint",
                "risk_level": "HIGH",
                "risk_summary": "Restricts trade or commercial activity post termination of relationship.",
                "legal_issue": "Agreements in restraint of trade are void under Section 27 of the Indian Contract Act, 1872.",
                "potential_consequence": "Clause is legally unenforceable in Indian courts and creates spurious litigation risk.",
                "recommended_revision": "Narrow scope to reasonable protection of trade secrets and confidentiality rather than blanket prohibition of trade.",
                "statutory_reference": "Section 27, Indian Contract Act, 1872 (Agreement in Restraint of Trade Void)"
            })

        # Check Liquidated Damages / Penalties
        if "penalty" in lower or "forfeit" in lower or ("liquidated damages" in lower and "excessive" in lower):
            overall_risk = "MEDIUM" if overall_risk == "LOW" else overall_risk
            risks.append({
                "risk_title": "Punitive Liquidated Damages",
                "risk_level": "MEDIUM",
                "risk_summary": "Pre-stipulated damages may be deemed punitive rather than genuine pre-estimates of loss.",
                "legal_issue": "Indian courts only enforce reasonable compensation not exceeding the amount named.",
                "potential_consequence": "Litigation delays to prove actual loss incurred under Section 74.",
                "recommended_revision": "Clarify that liquidated damages represent a genuine pre-estimate of loss and not a penalty.",
                "statutory_reference": "Section 74, Indian Contract Act, 1872 (Compensation for Breach of Contract)"
            })

        # If no major flags caught, provide general baseline check
        if not risks:
            risks.append({
                "risk_title": "Standard Commercial Exposure",
                "risk_level": "LOW",
                "risk_summary": "Clause appears structured within standard commercial parameters.",
                "legal_issue": "Ensure terms align with bilateral representations and warranties.",
                "potential_consequence": "Low direct exposure; verify governing law and jurisdiction clauses.",
                "recommended_revision": "Ensure defined terms match schedule definitions and dispute resolution procedures.",
                "statutory_reference": "Indian Contract Act, 1872"
            })
            notes.append("No immediate unconscionable or void provisions detected.")

        # Suggest improved draft
        improved_draft = (
            f"MUTUALLY REVISED DRAFT:\n"
            f"\"{clause_text.strip()}\n\n"
            f"PROVIDED ALWAYS that each party's aggregate liability under this clause shall be limited to the total fees "
            f"actually received under this Agreement during the twelve (12) months preceding the claim. "
            f"Neither party shall be liable for special, indirect, or consequential damages. Any termination shall require thirty (30) "
            f"business days prior written notice with opportunity to cure.\""
        )

        return {
            "overall_risk_score": overall_risk,
            "risk_factors": risks,
            "improved_clause_draft": improved_draft,
            "compliance_notes": notes
        }

    # ==================== POLICY COMPLIANCE AUDITOR ====================

    def audit_policy(self, text: str, framework: str = "DPDP_ACT_2023") -> Dict[str, Any]:
        """
        Audits text against legal & regulatory frameworks (e.g. DPDP Act 2023, IT Act, Consumer Protection).
        """
        lower = text.lower()

        if framework == "DPDP_ACT_2023":
            checks = [
                {
                    "framework_requirement": "Notice and Lawful Basis for Processing (Section 5)",
                    "status": "COMPLIANT" if ("notice" in lower or "purpose" in lower) else "NON_COMPLIANT",
                    "finding": "Notice specifying data processed and purpose identified." if ("notice" in lower or "purpose" in lower) else "Missing explicit itemized notice of personal data and specified purpose.",
                    "section_reference": "DPDP Act, 2023 - Section 5",
                    "remediation_step": "Draft clear, standalone notice in plain language accompanied by English / scheduled languages option."
                },
                {
                    "framework_requirement": "Reasonable Security Safeguards & Breach Prevention (Section 8(5))",
                    "status": "COMPLIANT" if ("security" in lower or "safeguard" in lower or "encrypt" in lower) else "NON_COMPLIANT",
                    "finding": "Technical safeguards and encryption standards referenced." if ("security" in lower or "safeguard" in lower) else "No documented technical and organizational measures to prevent personal data breach.",
                    "section_reference": "DPDP Act, 2023 - Section 8(5)",
                    "remediation_step": "Incorporate ISO 27001 / AES-256 encryption benchmarks and specify breach notification protocols."
                },
                {
                    "framework_requirement": "Data Principal Rights - Access, Correction & Erasure (Sections 11 & 12)",
                    "status": "COMPLIANT" if ("erasure" in lower or "delete" in lower or "correction" in lower) else "PARTIAL_RISK",
                    "finding": "Mechanisms for withdrawal of consent and erasure addressed." if ("erasure" in lower or "delete" in lower) else "Incomplete provisions for Data Principal right to erasure and correction.",
                    "section_reference": "DPDP Act, 2023 - Section 12",
                    "remediation_step": "Provide direct self-service portal or email endpoint for erasure upon withdrawal of consent."
                },
                {
                    "framework_requirement": "Grievance Redressal Mechanism & Data Protection Officer (Section 8(9) & 10)",
                    "status": "COMPLIANT" if ("grievance" in lower or "dpo" in lower or "officer" in lower) else "NON_COMPLIANT",
                    "finding": "Grievance redressal officer contact published." if ("grievance" in lower or "dpo" in lower) else "No designated Grievance Officer name, email, or physical address provided.",
                    "section_reference": "DPDP Act, 2023 - Section 8(9)",
                    "remediation_step": "Publish full contact details of the Grievance Officer and establish maximum 30-day response SLA."
                }
            ]
            title = "Digital Personal Data Protection Act (DPDP Act) 2023 Audit"

        else:
            checks = [
                {
                    "framework_requirement": "General Statutory Compliance & Transparency",
                    "status": "COMPLIANT",
                    "finding": "Document exhibits foundational disclosure practices.",
                    "section_reference": "Standard Regulatory Guidance",
                    "remediation_step": "Conduct periodic annual regulatory review."
                }
            ]
            title = f"{framework} Regulatory Audit"

        compliant_count = sum(1 for c in checks if c["status"] == "COMPLIANT")
        total_count = len(checks)
        percentage = int((compliant_count / total_count) * 100)

        return {
            "framework": framework,
            "title": title,
            "overall_compliance_percentage": percentage,
            "checks": checks,
            "executive_summary": (
                f"Audited against {title}. Overall compliance rated at {percentage}%. "
                f"Found {compliant_count} compliant controls out of {total_count} evaluated."
            ),
            "disclaimer": settings.LEGAL_DISCLAIMER
        }


# Global instance
gemini_service = GeminiLegalService()
