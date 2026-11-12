import React, { useState } from "react";
import { ShieldCheck, CheckCircle2, XCircle, AlertCircle, Sparkles, FileCheck2, ArrowRight } from "lucide-react";
import { auditPolicy } from "../services/api";

const SAMPLE_POLICY = `Privacy & Data Notice (Sample Corp):
We collect personal data such as names, phone numbers, and email addresses when you register.
Data is processed for internal analytics and order processing.
Security safeguards: All data in transit is encrypted using TLS.
Users may contact privacy@samplecorp.in to request erasure of their records.
Grievance Officer: Mr. Rahul Sharma (grievance@samplecorp.in), 4th Floor, Tech Park, Bengaluru.`;

export default function PolicyAuditor({ documents = [] }) {
  const [framework, setFramework] = useState("DPDP_ACT_2023");
  const [selectedDocId, setSelectedDocId] = useState("");
  const [policyText, setPolicyText] = useState(SAMPLE_POLICY);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAudit = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await auditPolicy(
        selectedDocId ? null : policyText,
        framework,
        selectedDocId || null
      );
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "COMPLIANT":
        return <span className="badge badge-low"><CheckCircle2 size={12} /> Compliant</span>;
      case "NON_COMPLIANT":
        return <span className="badge badge-critical"><XCircle size={12} /> Non-Compliant</span>;
      case "PARTIALLY_COMPLIANT":
      case "PARTIAL_RISK":
        return <span className="badge badge-medium"><AlertCircle size={12} /> Partially Compliant</span>;
      default:
        return <span className="badge badge-low">{status}</span>;
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* Control Card */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <ShieldCheck size={20} color="var(--gold-accent)" />
            <span>Statutory Policy & Regulatory Compliance Auditor</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Target Regulatory Framework:</span>
            <select
              value={framework}
              onChange={(e) => setFramework(e.target.value)}
              style={{
                background: "#081220",
                color: "#E2E8F0",
                border: "1px solid var(--navy-border)",
                padding: "0.4rem 0.8rem",
                borderRadius: "6px",
                fontSize: "0.82rem",
                outline: "none"
              }}
            >
              <option value="DPDP_ACT_2023">Digital Personal Data Protection Act (DPDP Act) 2023</option>
              <option value="IT_ACT_2000">Information Technology Act (IT Act) 2000 & SPDI Rules</option>
              <option value="CONSUMER_PROTECTION_2019">Consumer Protection (E-Commerce) Rules 2020</option>
            </select>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem", marginBottom: "1rem" }}>
          <div>
            <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Audit from Document Library (Optional)
            </label>
            <select
              value={selectedDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
              style={{
                width: "100%",
                background: "#081220",
                border: "1px solid var(--navy-border)",
                padding: "0.65rem 0.85rem",
                borderRadius: "6px",
                color: "#E2E8F0",
                fontSize: "0.84rem"
              }}
            >
              <option value="">-- Or audit custom text below --</option>
              {documents.map((d) => (
                <option key={d.doc_id} value={d.doc_id}>
                  {d.title} ({d.category})
                </option>
              ))}
            </select>
            <span style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginTop: "0.3rem", display: "block" }}>
              Choosing a library document pulls all indexed vector chunks for auditing.
            </span>
          </div>

          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
              <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)" }}>
                Policy / Clause Text to Audit
              </label>
              <button
                style={{ background: "transparent", border: "none", color: "var(--gold-accent)", fontSize: "0.74rem", cursor: "pointer" }}
                onClick={() => {
                  setSelectedDocId("");
                  setPolicyText(SAMPLE_POLICY);
                }}
              >
                Reset Sample Policy
              </button>
            </div>
            <textarea
              rows={4}
              value={policyText}
              disabled={Boolean(selectedDocId)}
              onChange={(e) => setPolicyText(e.target.value)}
              placeholder="Paste policy text or terms of service here..."
              style={{
                width: "100%",
                background: selectedDocId ? "#050b14" : "#081220",
                border: "1px solid var(--navy-border)",
                padding: "0.75rem",
                borderRadius: "6px",
                color: "#E2E8F0",
                fontSize: "0.84rem",
                fontFamily: "inherit",
                resize: "vertical"
              }}
            />
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            className="btn-primary"
            onClick={handleAudit}
            disabled={loading || (!selectedDocId && !policyText.trim())}
          >
            <Sparkles size={16} />
            <span>{loading ? "Auditing Compliance..." : "Audit Framework Compliance"}</span>
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: "1rem", background: "var(--danger-bg)", border: "1px solid var(--danger-border)", borderRadius: "8px", color: "var(--danger-text)", fontSize: "0.88rem" }}>
          Audit Error: {error}
        </div>
      )}

      {/* Audit Findings */}
      {result && (
        <div className="card">
          <div className="card-header">
            <div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#FFFFFF" }}>{result.title}</h3>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                {result.executive_summary}
              </p>
              {result.commencement_notice && (
                <div style={{ marginTop: "0.5rem", padding: "0.5rem 0.75rem", background: "rgba(59, 130, 246, 0.1)", border: "1px solid rgba(59, 130, 246, 0.2)", borderRadius: "6px", fontSize: "0.74rem", color: "#93C5FD" }}>
                  ⚖️ {result.commencement_notice}
                </div>
              )}
            </div>

            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: "1.8rem", fontWeight: 800, color: result.overall_compliance_percentage >= 75 ? "#10B981" : result.overall_compliance_percentage >= 50 ? "#F59E0B" : "#EF4444" }}>
                {result.overall_compliance_percentage}%
              </div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-dim)", textTransform: "uppercase" }}>Overall Score</div>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            {result.checks.map((c, idx) => (
              <div
                key={idx}
                style={{
                  background: "#0A1627",
                  border: "1px solid var(--navy-border)",
                  borderRadius: "8px",
                  padding: "1rem",
                  display: "grid",
                  gridTemplateColumns: "1fr auto",
                  gap: "1rem"
                }}
              >
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.3rem" }}>
                    <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "#FFFFFF" }}>
                      {c.control_name || c.framework_requirement}
                    </span>
                    {c.control_name && (
                      <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
                        ({c.framework_requirement})
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
                    {c.finding}
                  </div>

                  {c.evidence_extracted && c.evidence_extracted !== "No evidence found in submitted policy." && (
                    <div style={{ background: "rgba(16, 185, 129, 0.1)", border: "1px solid rgba(16, 185, 129, 0.2)", padding: "0.4rem 0.6rem", borderRadius: "5px", fontSize: "0.75rem", marginBottom: "0.4rem", color: "#6EE7B7" }}>
                      <strong>Evidence Extracted: </strong>
                      <span>{c.evidence_extracted}</span>
                    </div>
                  )}

                  {c.missing_requirements && c.missing_requirements !== "None identified." && c.missing_requirements !== "None. Satisfies baseline Section 5 notice mandates." && (
                    <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", padding: "0.4rem 0.6rem", borderRadius: "5px", fontSize: "0.75rem", marginBottom: "0.5rem", color: "#FCA5A5" }}>
                      <strong>Missing Requirements: </strong>
                      <span>{c.missing_requirements}</span>
                    </div>
                  )}

                  <div style={{ background: "rgba(0,0,0,0.3)", padding: "0.5rem 0.75rem", borderRadius: "6px", fontSize: "0.76rem", display: "flex", flexWrap: "wrap", gap: "1rem" }}>
                    <div>
                      <strong style={{ color: "var(--gold-accent)" }}>Statutory Ref: </strong>
                      <span>{c.section_reference}</span>
                    </div>
                    <div>
                      <strong style={{ color: "#6EE7B7" }}>Remediation: </strong>
                      <span>{c.remediation_step}</span>
                    </div>
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "flex-start" }}>
                  {getStatusBadge(c.status)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
