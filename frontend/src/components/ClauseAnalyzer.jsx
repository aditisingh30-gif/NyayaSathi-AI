import React, { useState } from "react";
import { AlertTriangle, ShieldAlert, CheckCircle, Copy, Check, Sparkles, FileText, ArrowRight } from "lucide-react";
import { analyzeClause } from "../services/api";

const SAMPLE_CLAUSES = [
  {
    name: "Unilateral Immediate Termination",
    type: "Commercial Agreement",
    text: "The Company may terminate this Agreement immediately at its sole and absolute discretion without prior notice to the Vendor and without liability for any unpaid fees or accrued milestones."
  },
  {
    name: "Uncapped Broad-Form Indemnity",
    type: "Vendor Master Services Agreement",
    text: "Vendor shall defend, indemnify, and hold harmless Customer from and against any and all claims, damages, liabilities, costs, and expenses (including attorneys' fees and indirect or consequential losses) arising out of or related to this Agreement without limit."
  },
  {
    name: "Blanket Post-Employment Restraint",
    type: "Employment Contract",
    text: "Employee agrees that for a period of twenty-four (24) months following termination of employment for any reason, Employee shall not engage in or contribute to any business or company that competes directly or indirectly with the Employer anywhere in India."
  }
];

export default function ClauseAnalyzer() {
  const [clauseText, setClauseText] = useState(SAMPLE_CLAUSES[0].text);
  const [contractType, setContractType] = useState(SAMPLE_CLAUSES[0].type);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copiedDraft, setCopiedDraft] = useState(false);

  const handleAnalyze = async (textToAnalyze = null, typeToAnalyze = null) => {
    const text = textToAnalyze || clauseText;
    const type = typeToAnalyze || contractType;
    if (!text.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const data = await analyzeClause(text, type);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyDraft = (draft) => {
    navigator.clipboard.writeText(draft);
    setCopiedDraft(true);
    setTimeout(() => setCopiedDraft(false), 2000);
  };

  const getRiskBadge = (score) => {
    switch (score?.toUpperCase()) {
      case "CRITICAL":
        return <span className="badge badge-critical">Critical Risk</span>;
      case "HIGH":
        return <span className="badge badge-high">High Risk</span>;
      case "MEDIUM":
        return <span className="badge badge-medium">Medium Risk</span>;
      default:
        return <span className="badge badge-low">Low / Acceptable</span>;
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* Input Panel */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <FileText size={20} color="var(--gold-accent)" />
            <span>Contract Clause Risk & Enforceability Review</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>Quick Test Samples:</span>
            {SAMPLE_CLAUSES.map((s, idx) => (
              <button
                key={idx}
                className="btn-secondary"
                style={{ padding: "0.25rem 0.6rem", fontSize: "0.74rem" }}
                onClick={() => {
                  setClauseText(s.text);
                  setContractType(s.type);
                  handleAnalyze(s.text, s.type);
                }}
              >
                {s.name}
              </button>
            ))}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: "1rem", marginBottom: "1rem" }}>
          <div>
            <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Contract Type
            </label>
            <input
              type="text"
              value={contractType}
              onChange={(e) => setContractType(e.target.value)}
              placeholder="e.g. SaaS Agreement, NDA, Employment"
              style={{
                width: "100%",
                background: "#081220",
                border: "1px solid var(--navy-border)",
                padding: "0.65rem",
                borderRadius: "6px",
                color: "#E2E8F0",
                fontSize: "0.84rem"
              }}
            />
          </div>

          <div>
            <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Clause Text to Evaluate *
            </label>
            <textarea
              rows={4}
              value={clauseText}
              onChange={(e) => setClauseText(e.target.value)}
              placeholder="Paste contract clause here for legal risk analysis..."
              style={{
                width: "100%",
                background: "#081220",
                border: "1px solid var(--navy-border)",
                padding: "0.75rem",
                borderRadius: "6px",
                color: "#E2E8F0",
                fontSize: "0.86rem",
                fontFamily: "inherit",
                resize: "vertical"
              }}
            />
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            className="btn-primary"
            onClick={() => handleAnalyze()}
            disabled={loading || !clauseText.trim()}
          >
            <Sparkles size={16} />
            <span>{loading ? "Analyzing Risks..." : "Evaluate Clause"}</span>
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: "1rem", background: "var(--danger-bg)", border: "1px solid var(--danger-border)", borderRadius: "8px", color: "var(--danger-text)", fontSize: "0.88rem" }}>
          Analysis Error: {error}
        </div>
      )}

      {/* Analysis Results */}
      {result && (
        <div className="analysis-layout">
          {/* Identified Risk Factors */}
          <div className="card">
            <div className="card-header">
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                <ShieldAlert size={20} color={result.overall_risk_score === "CRITICAL" ? "#EF4444" : "#F59E0B"} />
                <span className="card-title" style={{ margin: 0 }}>Risk Exposure Assessment</span>
              </div>
              {getRiskBadge(result.overall_risk_score)}
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {result.risk_factors.map((rf, idx) => (
                <div
                  key={idx}
                  style={{
                    background: "#0A1627",
                    border: "1px solid var(--navy-border)",
                    borderRadius: "8px",
                    padding: "1rem"
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                    <h4 style={{ fontSize: "0.92rem", fontWeight: 700, color: "#FFFFFF" }}>
                      {rf.risk_title}
                    </h4>
                    {getRiskBadge(rf.risk_level)}
                  </div>

                  <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginBottom: "0.6rem" }}>
                    {rf.risk_summary}
                  </p>

                  <div style={{ background: "rgba(0,0,0,0.25)", padding: "0.65rem", borderRadius: "6px", fontSize: "0.76rem", display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                    <div>
                      <strong style={{ color: "var(--gold-accent)" }}>Statutory Authority: </strong>
                      <span>{rf.statutory_reference || "Indian Contract Act, 1872"}</span>
                    </div>
                    <div>
                      <strong style={{ color: "#FCA5A5" }}>Potential Consequence: </strong>
                      <span>{rf.potential_consequence}</span>
                    </div>
                    <div>
                      <strong style={{ color: "#6EE7B7" }}>Recommended Revision: </strong>
                      <span>{rf.recommended_revision}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Improved Clause Draft */}
          <div className="card" style={{ display: "flex", flexDirection: "column" }}>
            <div className="card-header">
              <div className="card-title">
                <CheckCircle size={20} color="#10B981" />
                <span>Legally Balanced Redraft</span>
              </div>

              <button
                className="btn-secondary"
                style={{ padding: "0.3rem 0.75rem", fontSize: "0.78rem" }}
                onClick={() => handleCopyDraft(result.improved_clause_draft)}
              >
                {copiedDraft ? <Check size={14} color="#10B981" /> : <Copy size={14} />}
                <span>{copiedDraft ? "Copied" : "Copy Revised Draft"}</span>
              </button>
            </div>

            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div style={{ background: "#060D1A", border: "1px solid var(--navy-border)", borderRadius: "8px", padding: "1.1rem", fontSize: "0.84rem", color: "#E2E8F0", lineHeight: 1.6, whiteSpace: "pre-line" }}>
                {result.improved_clause_draft}
              </div>

              {result.compliance_notes && result.compliance_notes.length > 0 && (
                <div style={{ background: "rgba(30,62,98,0.25)", border: "1px solid var(--navy-border)", borderRadius: "8px", padding: "0.85rem" }}>
                  <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--gold-accent)", marginBottom: "0.35rem" }}>
                    Compliance Notes:
                  </div>
                  <ul style={{ paddingLeft: "1.2rem", fontSize: "0.76rem", color: "var(--text-muted)", display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                    {result.compliance_notes.map((n, i) => (
                      <li key={i}>{n}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div style={{ marginTop: "auto", fontSize: "0.72rem", color: "var(--text-dim)", padding: "0.5rem", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                Draft provided for advisory review. Incorporate into execution copies following formal legal clearance.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
