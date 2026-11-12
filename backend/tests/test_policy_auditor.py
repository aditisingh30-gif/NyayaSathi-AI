import pytest
from app.services.gemini_service import gemini_service


class TestPolicyAuditorLegalAccuracy:
    """
    Comprehensive legal-accuracy test suite for Policy Auditor (DPDP Act 2023):
    1. Correct statutory section references
    2. Commencement-date awareness (G.S.R. 843(E) dated 13 November 2025 -> 13 May 2027)
    3. Breach notification versus reasonable security safeguards (Section 8(5) vs 8(6))
    4. Grievance redressal under correct statutory sections (Section 8(9), 8(10) & 13)
    5. Consent withdrawal under Section 6(4)–(6)
    6. Score rounding (62.5% strictly rounded to 63%)
    7. Empty policy returns 0%
    """

    def test_correct_statutory_section_references(self):
        """
        Verify every control cites its exact statutory section without confusion:
        - Notice: Section 5
        - Consent Withdrawal: Section 6(4)–(6)
        - Security Safeguards: Section 8(5)
        - Breach Notification: Section 8(6)
        - Access: Section 11
        - Correction and Erasure: Section 12
        - Grievance Redressal: Section 8(9), 8(10) & Section 13
        """
        sample = (
            "We collect names and emails for order processing. "
            "Data is encrypted using TLS. "
            "Users may request erasure of data. "
            "Grievance Officer: Mr. Sharma (grievance@corp.in), Tech Park, Bengaluru."
        )
        res = gemini_service._fallback_policy_auditor(sample, "DPDP_ACT_2023")
        refs = {c["control_name"]: c["section_reference"] for c in res["checks"]}

        assert "Section 5" in refs["Itemized Notice & Specified Purpose"]
        assert "Section 6(4)" in refs["Itemized Notice & Specified Purpose"]
        assert "Section 8(5)" in refs["Reasonable Security Safeguards & Breach Notification"]
        assert "Section 8(6)" in refs["Reasonable Security Safeguards & Breach Notification"]
        assert "Section 11" in refs["Data Principal Rights: Access, Correction, Erasure & Withdrawal"]
        assert "Section 12" in refs["Data Principal Rights: Access, Correction, Erasure & Withdrawal"]
        assert "Section 8(9)" in refs["Grievance Redressal Mechanism & Officer Details"]
        assert "Section 8(10)" in refs["Grievance Redressal Mechanism & Officer Details"]
        assert "Section 13" in refs["Grievance Redressal Mechanism & Officer Details"]

    def test_commencement_date_awareness(self):
        """
        Verify audit includes commencement awareness noting G.S.R. 843(E) dated 13 Nov 2025:
        - Section 6(9) scheduled to commence tomorrow (13 November 2026 as of 12 November 2026).
        - Sections 3–5, 6(1)–(8), 6(10), and 7–17 scheduled to commence on 13 May 2027.
        """
        sample = "We collect names for billing. Security controls in place."
        res = gemini_service._fallback_policy_auditor(sample, "DPDP_ACT_2023")

        assert "commencement_notice" in res
        assert "G.S.R. 843(E)" in res["commencement_notice"]
        assert "13 May 2027" in res["commencement_notice"]
        assert "scheduled to commence tomorrow" in res["commencement_notice"].lower()
        assert "13 november 2026" in res["commencement_notice"].lower()

        for c in res["checks"]:
            assert "13 May 2027" in c["commencement_status"]
            assert "tomorrow" in c["commencement_status"].lower()

    def test_reasonable_security_vs_breach_notification(self):
        """
        Verify:
        - Section 8(5) does not mandate AES-256 universally (statutory standard is 'reasonable security safeguards').
        - Mentioning TLS only without breach notification under Section 8(6) is PARTIALLY_COMPLIANT (50 pts).
        """
        tls_only = "Security: All transmission is secured via TLS encryption."
        res = gemini_service._fallback_policy_auditor(tls_only, "DPDP_ACT_2023")
        sec_check = next(c for c in res["checks"] if "Security" in c["control_name"])

        assert sec_check["status"] == "PARTIALLY_COMPLIANT"
        assert sec_check["points_awarded"] == 50
        assert "breach notification" in sec_check["missing_requirements"].lower()
        # Ensure remediation does not present AES-256 as a rigid statutory mandate
        assert "appropriate to the nature of personal data" in sec_check["remediation_step"].lower() or "reasonable" in sec_check["remediation_step"].lower()

    def test_grievance_redressal_statutory_mapping(self):
        """
        Verify:
        - Section 8(9) assigns duty to publish contact coordinates.
        - Section 8(10) assigns exhaustion requirement before approaching Board.
        - Section 13 assigns Data Principal right of grievance redressal.
        """
        policy_with_email_only = (
            "We value privacy. For grievances, contact us at grievance@company.com."
        )
        res = gemini_service._fallback_policy_auditor(policy_with_email_only, "DPDP_ACT_2023")
        gr_check = next(c for c in res["checks"] if "Grievance" in c["control_name"])

        # Email only lacks officer name and physical coordinates under Section 8(9)
        assert gr_check["status"] == "PARTIALLY_COMPLIANT"
        assert gr_check["points_awarded"] == 50
        assert "Section 8(9)" in gr_check["section_reference"]
        assert "Section 8(10)" in gr_check["section_reference"]
        assert "Section 13" in gr_check["section_reference"]

    def test_consent_withdrawal_under_section_6(self):
        """
        Verify that consent withdrawal is evaluated under Section 6(4)–(6),
        and policy with itemized data + purpose but lacking withdrawal is PARTIALLY_COMPLIANT.
        """
        policy_no_withdrawal = (
            "Notice: We collect personal data such as names and phone numbers for the specified purpose of delivery."
        )
        res = gemini_service._fallback_policy_auditor(policy_no_withdrawal, "DPDP_ACT_2023")
        n_check = next(c for c in res["checks"] if "Notice" in c["control_name"])

        assert n_check["status"] == "PARTIALLY_COMPLIANT"
        assert n_check["points_awarded"] == 50
        assert "Section 6(4)" in n_check["missing_requirements"]

    def test_score_rounding_62_5_to_63(self):
        """
        Standard rounding verification:
        When 3 controls are PARTIALLY_COMPLIANT (50, 50, 50) and 1 is COMPLIANT (100):
        Total = 250 / 400 = 62.5% -> Strictly rounds to 63%.
        """
        sample = (
            "Privacy & Data Notice (Sample Corp):\n"
            "We collect personal data such as names, phone numbers, and email addresses when you register.\n"
            "Data is processed for internal analytics and order processing.\n"
            "Security safeguards: All data in transit is encrypted using TLS.\n"
            "Users may contact privacy@samplecorp.in to request erasure of their records.\n"
            "Grievance Officer: Mr. Rahul Sharma (grievance@samplecorp.in), 4th Floor, Tech Park, Bengaluru."
        )
        res = gemini_service._fallback_policy_auditor(sample, "DPDP_ACT_2023")

        # Breakdown: 50 + 50 + 50 + 100 = 250 / 4 = 62.5 -> 63
        assert res["overall_compliance_percentage"] == 63
        assert res["controls_assessed"] == 4
        assert "63%" in res["executive_summary"]
        assert "250/400 points" in res["executive_summary"]

    def test_complete_compliant_policy_achieves_100(self):
        """
        Verify fully compliant policy achieves 100% across all 4 controls.
        """
        policy = (
            "Comprehensive Data Notice:\n"
            "1. We collect personal data such as names, email, and phone numbers. "
            "Data is processed for specified purpose of account verification. "
            "Data Principals may withdraw consent at any time with comparable ease.\n"
            "2. We maintain reasonable security safeguards and technical measures to prevent personal data breach. "
            "In case of breach, we intimate the Data Protection Board and affected Data Principals without delay.\n"
            "3. Users have the right to access a summary of personal data, request correction or updating, and request erasure of personal data.\n"
            "4. Grievance Officer: Mr. Rahul Sharma (grievance@corp.in), 4th Floor, Tech Park, Bengaluru. "
            "Grievances must be submitted to the Data Fiduciary before approaching the Board."
        )
        res = gemini_service._fallback_policy_auditor(policy, "DPDP_ACT_2023")

        assert res["overall_compliance_percentage"] == 100
        for c in res["checks"]:
            assert c["status"] == "COMPLIANT"
            assert c["points_awarded"] == 100

    def test_empty_policy_returns_zero(self):
        """
        An empty policy or whitespace-only input must return 0% compliance.
        """
        res = gemini_service.audit_policy("   ", "DPDP_ACT_2023")
        assert res["overall_compliance_percentage"] == 0
        assert len(res["checks"]) == 4
        for c in res["checks"]:
            assert c["status"] == "NON_COMPLIANT"
            assert c["points_awarded"] == 0
