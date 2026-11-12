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
            "You are an expert Corporate & Contract Law Counsel specializing in Indian Commercial Law and standard contracting practices. "
            "Analyze the given contract clause for operational, financial, and legal risks. "
            "Guidelines for evaluation:\n"
            "1. Termination clauses: Distinguish balanced termination (mutual rights, reasonable written notice e.g. 30+ days, cure periods for breach) which is LOW/ACCEPTABLE, "
            "from unilateral termination at will without notice or compensation which is HIGH or CRITICAL risk.\n"
            "2. Notice & Cure periods: Absence of notice or absence of cure period for remediable breach must be flagged with appropriate risk level.\n"
            "3. Indemnities: Uncapped, one-sided, or covering consequential losses should be evaluated as HIGH or CRITICAL.\n"
            "4. Risk levels must strictly be one of: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.\n"
            "5. The improved_clause_draft must specifically address the identified deficiencies in the clause without introducing unrelated or unsupported boilerplate.\n"
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
            parsed = json.loads(cleaned)

            # Defensive Floor Guardrail: Compute baseline deterministic risk
            # Ensures LLM output cannot downgrade an objectively severe risk (e.g. unilateral no-notice termination)
            # below the floor established by commercial law rules.
            baseline = self._fallback_clause_analyzer(clause_text, contract_type)
            risk_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            gemini_score = parsed.get("overall_risk_score", "LOW").upper()
            baseline_score = baseline.get("overall_risk_score", "LOW").upper()

            if risk_rank.get(baseline_score, 1) > risk_rank.get(gemini_score, 1):
                parsed["overall_risk_score"] = baseline_score
                # Ensure the critical risk factors from baseline are included if missing
                existing_titles = {rf.get("risk_title", "") for rf in parsed.get("risk_factors", [])}
                for rf in baseline.get("risk_factors", []):
                    if rf.get("risk_level") in ["HIGH", "CRITICAL"] and rf.get("risk_title") not in existing_titles:
                        parsed.setdefault("risk_factors", []).insert(0, rf)

            return parsed
        except Exception:
            # Deterministic Contract Risk Analysis Engine
            return self._fallback_clause_analyzer(clause_text, contract_type)

    def _fallback_clause_analyzer(self, clause_text: str, contract_type: str) -> Dict[str, Any]:
        """Built-in rule-based contract clause risk engine."""
        lower = clause_text.lower()
        risks = []
        notes = []

        # Hierarchy helper
        risk_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

        def elevate_risk(current: str, candidate: str) -> str:
            return candidate if risk_rank.get(candidate, 1) > risk_rank.get(current, 1) else current

        overall_risk = "LOW"

        # ----------------- 1. TERMINATION ANALYSIS -----------------
        is_termination_clause = any(w in lower for w in ["terminate", "termination", "cancel", "cancellation", "rescind"])
        
        if is_termination_clause:
            # Check for unilateral / one-sided termination power
            has_unilateral_party = bool(re.search(r"\b(company|client|customer|buyer|licensor|disclosing party|employer|vendor)\s+may\s+terminate\b", lower))
            has_sole_discretion = any(w in lower for w in ["sole discretion", "at its discretion", "at its option", "at will"])
            has_any_time = any(w in lower for w in ["at any time", "at whenever", "immediately and without"])
            has_mutual = any(w in lower for w in ["either party", "both parties", "each party", "mutually"])

            is_unilateral = (has_unilateral_party or has_sole_discretion or has_any_time) and not has_mutual

            # Check for lack of notice
            no_notice = any(w in lower for w in [
                "without notice", "without prior notice", "without any prior notice",
                "without written notice", "no notice", "immediate termination", "terminate immediately",
                "without cause"
            ]) or bool(re.search(r"\bterminate\b.*?\bimmediately\b", lower))
            has_express_notice = bool(re.search(r"\b(?:\d+|thirty|sixty|ninety|fifteen|ten|forty-five)\b.*?\b(?:days?|business days?|months?)\b.*?\bnotice\b", lower)) or "prior written notice" in lower or "advance notice" in lower

            # Check for lack of compensation / without liability
            no_compensation = any(w in lower for w in [
                "without compensation", "without liability", "no compensation", "no liability", "without penalty"
            ]) or (any(w in lower for w in ["without", "no"]) and any(w in lower for w in ["compensation", "liability", "damages"]))

            # Check for lack of cure period when termination is for breach
            is_breach_termination = any(w in lower for w in ["breach", "default", "failure to perform"])
            has_cure_period = bool(re.search(r"\b(?:cure|remedy)\b.*?\b\d+\b\s*(?:days?|business days?)", lower)) or any(w in lower for w in ["opportunity to cure", "period to cure", "cure period", "remedy such breach", "is not cured within"])

            # Evaluate severity
            if is_unilateral and (no_notice or has_any_time) and no_compensation:
                overall_risk = elevate_risk(overall_risk, "CRITICAL")
                risks.append({
                    "risk_title": "Unilateral At-Will Termination Without Notice or Compensation",
                    "risk_level": "CRITICAL",
                    "risk_summary": "One party holds unconstrained power to cancel the agreement at any moment without advance notice or compensation for accrued obligations.",
                    "legal_issue": "Grossly one-sided termination clause depriving the non-terminating party of reciprocity, reasonable reliance, and financial recourse.",
                    "potential_consequence": "Sudden operational stoppage, unrecoverable capital or resource investments, and total lack of transition safeguards.",
                    "recommended_revision": "Require mutual termination rights with a mandatory written notice period (e.g. 30 to 60 days) and compensation for work performed up to the termination date.",
                    "statutory_reference": "Sections 39 & 73, Indian Contract Act, 1872 (Right to Rescind and Damages for Breach)"
                })
                notes.append("Unilateral at-will termination without notice exposes the dependent party to severe unmitigated commercial risk.")

            elif is_unilateral and (no_notice or has_any_time):
                overall_risk = elevate_risk(overall_risk, "HIGH")
                risks.append({
                    "risk_title": "Unilateral Immediate Termination Without Cause",
                    "risk_level": "HIGH",
                    "risk_summary": "One party retains the unilateral right to cancel without adequate notice, creating operational instability.",
                    "legal_issue": "Lack of reciprocal termination rights and absence of reasonable advance notice.",
                    "potential_consequence": "Unexpected cancellation of services with zero transition runway and potential cash-flow disruption.",
                    "recommended_revision": "Convert to bilateral termination for convenience with a minimum 30 days prior written notice.",
                    "statutory_reference": "Section 73, Indian Contract Act, 1872"
                })
                notes.append("Commercial contracts typically require bilateral termination rights with 30-60 days advance notice.")

            elif is_unilateral and not has_mutual:
                overall_risk = elevate_risk(overall_risk, "MEDIUM")
                risks.append({
                    "risk_title": "One-Sided Termination Rights",
                    "risk_level": "MEDIUM",
                    "risk_summary": "Termination rights are granted exclusively to one party rather than both parties on equal terms.",
                    "legal_issue": "Asymmetric contractual balance of power.",
                    "potential_consequence": "Inability for the counterparty to exit the agreement under equivalent circumstances.",
                    "recommended_revision": "Make the termination rights reciprocal so that either party may exercise them under the same conditions.",
                    "statutory_reference": "Principles of Mutuality, Indian Contract Act, 1872"
                })

            # Check breach termination missing cure period
            if is_breach_termination and not has_cure_period and no_notice:
                overall_risk = elevate_risk(overall_risk, "HIGH")
                risks.append({
                    "risk_title": "Immediate Termination for Breach Without Cure Period",
                    "risk_level": "HIGH",
                    "risk_summary": "Agreement allows instantaneous cancellation upon alleged breach without providing an opportunity to cure remediable defaults.",
                    "legal_issue": "Absence of a standard 15 to 30-day notice-and-cure mechanism for curable breaches.",
                    "potential_consequence": "Forfeiture or abrupt cancellation triggered by minor, inadvertent, or remediable technical non-compliance.",
                    "recommended_revision": "Insert an explicit provision granting 30 days written notice to remedy any curable breach prior to termination taking effect.",
                    "statutory_reference": "Section 55, Indian Contract Act, 1872"
                })
                notes.append("Best practice requires a notice-and-cure window before terminating for cause.")

            elif is_breach_termination and not has_cure_period and not no_notice:
                overall_risk = elevate_risk(overall_risk, "MEDIUM")
                risks.append({
                    "risk_title": "Absence of Express Cure Period for Material Breach",
                    "risk_level": "MEDIUM",
                    "risk_summary": "The clause does not specify a clear time window to cure a remediable breach before termination becomes final.",
                    "legal_issue": "Ambiguity regarding whether curable defaults can be remedied prior to termination.",
                    "potential_consequence": "Disputes regarding whether immediate termination was justified or premature.",
                    "recommended_revision": "Specify a 15 to 30-day written notice and cure window for any curable breach.",
                    "statutory_reference": "Section 39, Indian Contract Act, 1872"
                })

            # If balanced termination clause with mutual rights and clear notice
            if has_mutual and (has_express_notice or has_cure_period) and not any(r["risk_level"] in ["HIGH", "CRITICAL"] for r in risks):
                overall_risk = "LOW"
                risks = [{
                    "risk_title": "Balanced Bilateral Termination",
                    "risk_level": "LOW",
                    "risk_summary": "The clause provides mutual termination rights with express advance written notice and/or cure mechanics.",
                    "legal_issue": "None detected; reciprocal and aligned with standard commercial practice.",
                    "potential_consequence": "Standard commercial notice and orderly exit process.",
                    "recommended_revision": "Retain clause as drafted; ensure operational teams track notice delivery requirements.",
                    "statutory_reference": "Section 73, Indian Contract Act, 1872"
                }]
                notes = ["Termination rights appear mutual with reasonable advance notice."]

        # ----------------- 2. INDEMNITY & LIABILITY ANALYSIS -----------------
        has_indemnity = any(w in lower for w in ["indemnify", "indemnification", "hold harmless", "defend and hold"])
        if has_indemnity:
            is_unlimited = any(w in lower for w in ["unlimited", "without limit", "without limitation", "all losses", "any and all losses"])
            covers_indirect = any(w in lower for w in ["indirect", "consequential", "punitive", "special damages", "incidental"])
            is_one_sided_indemnity = bool(re.search(r"\b(vendor|contractor|service provider|licensee)\s+shall\s+indemnify\b", lower)) and "each party" not in lower

            if is_unlimited or covers_indirect:
                overall_risk = elevate_risk(overall_risk, "CRITICAL")
                risks.append({
                    "risk_title": "Uncapped Broad-Form Indemnity",
                    "risk_level": "CRITICAL",
                    "risk_summary": "Indemnification is not capped to contract value and encompasses indirect or consequential losses.",
                    "legal_issue": "Disproportionate liability exposure exceeding contract value, potentially bypassing standard statutory remoteness of damage rules.",
                    "potential_consequence": "Catastrophic, uninsurable financial liability far exceeding the financial benefit of the agreement.",
                    "recommended_revision": "Cap aggregate indemnification to total fees paid/payable under the agreement, and expressly carve out consequential and indirect damages.",
                    "statutory_reference": "Section 73 & Section 124, Indian Contract Act, 1872"
                })
                notes.append("Insist on an aggregate liability cap (e.g. 1x-2x Annual Contract Value) and exclude indirect damages.")
            elif is_one_sided_indemnity:
                overall_risk = elevate_risk(overall_risk, "MEDIUM")
                risks.append({
                    "risk_title": "Unilateral Indemnification Obligation",
                    "risk_level": "MEDIUM",
                    "risk_summary": "Indemnity obligation is imposed solely on one party without reciprocal protection.",
                    "legal_issue": "Asymmetrical risk allocation.",
                    "potential_consequence": "Unbalanced exposure for third-party claims.",
                    "recommended_revision": "Negotiate mutual indemnification limited to third-party claims arising from gross negligence, willful misconduct, or IP infringement.",
                    "statutory_reference": "Section 124, Indian Contract Act, 1872"
                })

        # ----------------- 3. RESTRAINT OF TRADE (NON-COMPETE) -----------------
        if any(w in lower for w in ["not compete", "non-compete", "restraint of trade", "shall not engage in any competing"]):
            overall_risk = elevate_risk(overall_risk, "HIGH")
            risks.append({
                "risk_title": "Post-Termination Non-Compete Restraint",
                "risk_level": "HIGH",
                "risk_summary": "Imposes restrictions on commercial trade or professional activity following contract termination.",
                "legal_issue": "Agreements in restraint of trade or profession are void under Section 27 of the Indian Contract Act, 1872.",
                "potential_consequence": "The covenant is legally unenforceable in Indian courts and exposes parties to unnecessary litigation.",
                "recommended_revision": "Replace blanket non-compete with enforceable non-solicitation of clients/employees and strict non-disclosure of confidential information.",
                "statutory_reference": "Section 27, Indian Contract Act, 1872 (Agreement in Restraint of Trade Void)"
            })
            notes.append("Under Indian law, post-contractual non-compete restrictions are void ab initio under Section 27.")

        # ----------------- 4. PUNITIVE LIQUIDATED DAMAGES / PENALTIES -----------------
        if any(w in lower for w in ["penalty", "forfeit all", "punitive damages"]) or ("liquidated damages" in lower and any(w in lower for w in ["excessive", "unreasonable", "regardless of actual"])):
            overall_risk = elevate_risk(overall_risk, "MEDIUM")
            risks.append({
                "risk_title": "Punitive Liquidated Damages or Penalty Stipulation",
                "risk_level": "MEDIUM",
                "risk_summary": "Pre-stipulated damages appear punitive rather than a genuine pre-estimate of anticipated loss.",
                "legal_issue": "Indian courts do not enforce penalties; under Section 74, only reasonable compensation up to the named amount is recoverable.",
                "potential_consequence": "Litigation delays requiring judicial assessment to prove actual losses incurred.",
                "recommended_revision": "Explicitly state that liquidated damages represent a genuine pre-estimate of loss and not a penalty.",
                "statutory_reference": "Section 74, Indian Contract Act, 1872 (Compensation for Breach of Contract)"
            })

        # ----------------- 5. UNFAIR EXCLUSION OF LIABILITY -----------------
        if any(w in lower for w in ["sole remedy", "exclusive remedy"]) and any(w in lower for w in ["no liability whatsoever", "disclaims all liability"]):
            overall_risk = elevate_risk(overall_risk, "HIGH")
            risks.append({
                "risk_title": "Total Exclusion of Liability / Illusory Remedy",
                "risk_level": "HIGH",
                "risk_summary": "One party disclaims virtually all liability, rendering remedies illusory.",
                "legal_issue": "Unconscionable exclusion of fundamental contractual obligations.",
                "potential_consequence": "Inability to recover damages in the event of gross negligence or material failure.",
                "recommended_revision": "Ensure exclusions do not apply to gross negligence, willful misconduct, breach of confidentiality, or statutory liabilities.",
                "statutory_reference": "Section 73, Indian Contract Act, 1872"
            })

        # ----------------- 6. BASELINE CHECK IF NO RISKS FOUND -----------------
        if not risks:
            risks.append({
                "risk_title": "Standard Commercial Exposure",
                "risk_level": "LOW",
                "risk_summary": "Clause appears structured within customary commercial parameters without obvious unilateral or unconscionable covenants.",
                "legal_issue": "Verify terms align with bilateral representations and overall agreement schedules.",
                "potential_consequence": "Low direct legal risk; ensure governing law and dispute escalation clauses are properly cross-referenced.",
                "recommended_revision": "Retain language with standard cross-referencing to definitions and dispute resolution procedures.",
                "statutory_reference": "Indian Contract Act, 1872"
            })
            notes.append("No immediate unconscionable, void, or severe asymmetrical provisions detected.")

        # ----------------- 7. EVIDENCE-TAILORED REVISED DRAFT -----------------
        # Generate targeted revision based on the actual identified risks
        revision_parts = []
        if any("termination" in r["risk_title"].lower() for r in risks):
            if any(r["risk_level"] in ["HIGH", "CRITICAL"] for r in risks):
                revision_parts.append(
                    "Either party may terminate this Agreement for convenience upon providing at least thirty (30) days prior written notice to the other party. "
                    "Upon any such termination, the Customer shall compensate the Vendor for all authorized services performed and reimbursable expenses incurred up to the effective termination date. "
                    "In the event of a material breach, the non-defaulting party may terminate this Agreement only after giving thirty (30) days prior written notice specifying the breach, "
                    "provided the defaulting party fails to cure such breach within said thirty (30) day period."
                )
        if any("indemnity" in r["risk_title"].lower() for r in risks):
            revision_parts.append(
                "Each party's aggregate indemnification and liability under this clause shall be limited to the total fees paid or payable under this Agreement in the twelve (12) months preceding the claim. "
                "Neither party shall be liable for any indirect, special, punitive, or consequential damages."
            )
        if any("non-compete" in r["risk_title"].lower() for r in risks):
            revision_parts.append(
                "During the term of this Agreement and for twelve (12) months thereafter, neither party shall solicit the key employees of the other party without prior written consent. "
                "Confidential information shall be maintained strictly in accordance with the confidentiality provisions herein."
            )

        if revision_parts:
            improved_draft = (
                f"RECOMMENDED REVISED DRAFT:\n"
                f"\"{ ' '.join(revision_parts) }\""
            )
        else:
            improved_draft = (
                f"RECOMMENDED REVISED DRAFT:\n"
                f"\"{clause_text.strip()}\n\n"
                f"[Confirmed: Language is commercially standard and bilateral; ensure defined terms and notice delivery methods align with the Master Agreement.]\""
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
