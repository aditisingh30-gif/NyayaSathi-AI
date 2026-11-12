import pytest
from app.services.gemini_service import gemini_service


class TestClauseRiskAnalyzer:
    """
    Test suite for Clause Risk Analyzer covering unilateral, bilateral,
    indemnity, non-compete, and breach cure scenarios.
    """

    def test_unilateral_at_will_no_notice_no_compensation_is_critical(self):
        """
        A clause allowing one party to terminate at any time without prior notice
        or compensation must be classified as CRITICAL (or HIGH) risk, never LOW.
        """
        clause = "The Company may terminate this Agreement at any time without prior notice or compensation."
        result = gemini_service._fallback_clause_analyzer(clause, "Service Agreement")
        
        assert result["overall_risk_score"] in ["CRITICAL", "HIGH"]
        assert any(
            rf["risk_level"] in ["CRITICAL", "HIGH"] and "termination" in rf["risk_title"].lower()
            for rf in result["risk_factors"]
        )
        assert len(result["improved_clause_draft"]) > 30
        assert "notice" in result["improved_clause_draft"].lower()
        assert "compensate" in result["improved_clause_draft"].lower()

    def test_unilateral_immediate_without_cause_is_high(self):
        """
        Unilateral immediate termination without cause creates operational volatility
        and must be rated HIGH risk.
        """
        clause = "Licensor may terminate this Agreement immediately without cause upon written email."
        result = gemini_service._fallback_clause_analyzer(clause, "Software License Agreement")
        
        assert result["overall_risk_score"] == "HIGH"
        assert any("Unilateral" in rf["risk_title"] for rf in result["risk_factors"])

    def test_unilateral_termination_with_notice_is_medium(self):
        """
        A termination clause with notice but granted exclusively to one party
        is asymmetric (MEDIUM risk), not CRITICAL and not LOW.
        """
        clause = "Client may terminate this Agreement by providing thirty (30) days advance written notice."
        result = gemini_service._fallback_clause_analyzer(clause, "Consulting Agreement")
        
        assert result["overall_risk_score"] == "MEDIUM"
        assert any("One-Sided" in rf["risk_title"] for rf in result["risk_factors"])

    def test_balanced_bilateral_termination_is_low(self):
        """
        Mutual termination rights with 30-day notice must be classified as LOW risk.
        We do not flag every termination clause as high risk.
        """
        clause = "Either party may terminate this Agreement for convenience upon providing at least thirty (30) days prior written notice to the other party."
        result = gemini_service._fallback_clause_analyzer(clause, "Vendor Agreement")
        
        assert result["overall_risk_score"] == "LOW"
        assert len(result["risk_factors"]) == 1
        assert result["risk_factors"][0]["risk_level"] == "LOW"
        assert "Bilateral" in result["risk_factors"][0]["risk_title"]

    def test_balanced_breach_with_cure_period_is_low(self):
        """
        Mutual termination for cause with a 30-day cure period is standard and LOW risk.
        """
        clause = "Either party may terminate this Agreement immediately for material breach if such breach is not cured within 30 days of written notice."
        result = gemini_service._fallback_clause_analyzer(clause, "Commercial MSA")
        
        assert result["overall_risk_score"] == "LOW"
        assert result["risk_factors"][0]["risk_level"] == "LOW"

    def test_immediate_breach_without_cure_period_is_high(self):
        """
        Termination for breach without providing any opportunity to cure must be flagged.
        """
        clause = "Customer may terminate this Agreement immediately upon any breach by Vendor without any cure period."
        result = gemini_service._fallback_clause_analyzer(clause, "IT Services Contract")
        
        assert result["overall_risk_score"] in ["HIGH", "CRITICAL"]

    def test_uncapped_indemnity_is_critical(self):
        """
        Uncapped indemnification for all losses direct or indirect must be CRITICAL.
        """
        clause = "Vendor shall defend, indemnify, and hold harmless Company from any and all losses, including indirect and consequential damages, without limit."
        result = gemini_service._fallback_clause_analyzer(clause, "Procurement Contract")
        
        assert result["overall_risk_score"] == "CRITICAL"
        assert any("Uncapped" in rf["risk_title"] for rf in result["risk_factors"])

    def test_post_termination_non_compete_is_high(self):
        """
        Post-termination non-compete restraint is void under Section 27 and must be HIGH risk.
        """
        clause = "Employee shall not engage in any competing business or enter into employment with a competitor for two years following termination."
        result = gemini_service._fallback_clause_analyzer(clause, "Employment Agreement")
        
        assert result["overall_risk_score"] == "HIGH"
        assert any("Section 27" in rf["statutory_reference"] for rf in result["risk_factors"])

    def test_standard_commercial_clause_is_low(self):
        """
        Commercially balanced boilerplate without aggressive covenants remains LOW.
        """
        clause = "This Agreement shall be governed by and construed in accordance with the laws of India. Any disputes shall be subject to the exclusive jurisdiction of the courts in New Delhi."
        result = gemini_service._fallback_clause_analyzer(clause, "Commercial Contract")
        
        assert result["overall_risk_score"] == "LOW"

    def test_gemini_output_cannot_downgrade_critical_risk(self, monkeypatch):
        """
        Verifies that even if an external LLM hallucinates and returns 'LOW' for a critical clause,
        the risk floor guardrail prevents the downgrade.
        """
        clause = "The Company may terminate this Agreement at any time without prior notice or compensation."
        
        # Simulate LLM returning a flawed LOW evaluation
        flawed_llm_response = (
            '{"overall_risk_score": "LOW", "risk_factors": [], "improved_clause_draft": "ok", "compliance_notes": []}'
        )
        monkeypatch.setattr(gemini_service, "_call_gemini", lambda *args, **kwargs: flawed_llm_response)
        
        result = gemini_service.analyze_clause(clause, "Commercial Agreement")
        assert result["overall_risk_score"] in ["CRITICAL", "HIGH"]
        assert any("Termination" in rf["risk_title"] for rf in result["risk_factors"])
