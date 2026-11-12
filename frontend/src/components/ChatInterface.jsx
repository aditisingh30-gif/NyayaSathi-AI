import React, { useState } from "react";
import { Send, Scale, BookOpen, Copy, Check, Sparkles, AlertCircle } from "lucide-react";
import { queryLegalCorpus } from "../services/api";

const PRESET_QUERIES = [
  "What is the maximum penalty for failure to prevent personal data breach under DPDP Act 2023?",
  "Are post-termination non-compete clauses enforceable under the Indian Contract Act?",
  "What are the essential conditions for a valid contract under Section 10?",
  "What constitutes cheating and fraud under Bharatiya Nyaya Sanhita (BNS)?",
  "How are limitation of liability and super-caps structured in enterprise SaaS agreements?"
];

export default function ChatInterface({ documents = [] }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: (
        "Namaste! I am **NyayaSathi (न्यायसाथी)**, your AI Legal & Policy Assistant. "
        + "I am equipped with statutory grounding in Indian Commercial, Criminal, and Data Governance frameworks, "
        + "including the **DPDP Act 2023**, **Indian Contract Act 1872**, and **Bharatiya Nyaya Sanhita 2023**.\n\n"
        + "Ask me any statutory question, or choose a benchmark query below to start."
      ),
      citations: [],
      confidence: 1.0,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  const [input, setInput] = useState("");
  const [selectedDocId, setSelectedDocId] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeCitations, setActiveCitations] = useState([]);
  const [copiedIndex, setCopiedIndex] = useState(null);

  const handleSend = async (queryText = null) => {
    const textToSend = queryText || input;
    if (!textToSend.trim() || loading) return;

    const userMsg = {
      role: "user",
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const docFilter = selectedDocId ? [selectedDocId] : null;
      const res = await queryLegalCorpus(textToSend, docFilter, 4);

      const assistantMsg = {
        role: "assistant",
        content: res.answer,
        citations: res.citations || [],
        confidence: res.confidence_score,
        disclaimer: res.disclaimer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
      if (res.citations && res.citations.length > 0) {
        setActiveCitations(res.citations);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ Legal analysis error: ${err.message}. Please check backend connection.`,
          citations: [],
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="chat-layout">
      {/* Left Chat Window */}
      <div className="chat-main">
        {/* Chat Control Header */}
        <div className="chat-control-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <Scale size={18} color="var(--gold-accent)" />
            <span style={{ fontSize: "0.88rem", fontWeight: 700 }}>Interactive Legal RAG Counsel</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", width: "100%", maxWidth: "340px" }}>
            <BookOpen size={16} color="var(--text-muted)" style={{ flexShrink: 0 }} />
            <select
              style={{ background: "#081220", color: "#F1F5F9", border: "1px solid var(--navy-border)", padding: "0.4rem 0.75rem", borderRadius: "6px", fontSize: "0.8rem", outline: "none", width: "100%" }}
              value={selectedDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
            >
              <option value="">All Knowledge Base Documents</option>
              {documents.map((doc) => (
                <option key={doc.doc_id} value={doc.doc_id}>
                  {doc.title.length > 38 ? doc.title.substring(0, 38) + "..." : doc.title}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Message Feed */}
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`chat-bubble ${msg.role}`}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                <span style={{ fontSize: "0.74rem", fontWeight: 700, color: msg.role === "assistant" ? "var(--gold-accent)" : "#94A3B8" }}>
                  {msg.role === "assistant" ? "NyayaSathi Legal Counsel" : "You (Inquirer)"} • {msg.timestamp}
                </span>

                {msg.role === "assistant" && (
                  <button
                    onClick={() => handleCopy(msg.content, idx)}
                    style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.72rem" }}
                  >
                    {copiedIndex === idx ? <Check size={13} color="#10B981" /> : <Copy size={13} />}
                    {copiedIndex === idx ? "Copied" : "Copy"}
                  </button>
                )}
              </div>

              {/* Render lines / formatting */}
              <div style={{ whiteSpace: "pre-line" }}>
                {msg.content}
              </div>

              {/* Show inline citation button if available */}
              {msg.citations && msg.citations.length > 0 && (
                <div style={{ marginTop: "0.85rem", paddingTop: "0.6rem", borderTop: "1px solid rgba(255,255,255,0.08)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: "0.74rem", color: "var(--gold-accent)", fontWeight: 600 }}>
                    📚 Cited {msg.citations.length} Statutory Source{msg.citations.length > 1 ? "s" : ""}
                  </span>
                  <button
                    className="btn-secondary"
                    style={{ padding: "0.25rem 0.6rem", fontSize: "0.74rem" }}
                    onClick={() => setActiveCitations(msg.citations)}
                  >
                    View Sources
                  </button>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="chat-bubble assistant" style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <Sparkles className="animate-spin" size={18} color="var(--gold-accent)" />
              <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Retrieving statutory chunks from ChromaDB & synthesizing legal citations...
              </span>
            </div>
          )}
        </div>

        {/* Suggested Queries */}
        <div style={{ padding: "0.5rem 1.25rem", background: "rgba(11,25,44,0.4)", borderTop: "1px solid rgba(255,255,255,0.05)", display: "flex", gap: "0.5rem", overflowX: "auto" }}>
          {PRESET_QUERIES.map((q, i) => (
            <button
              key={i}
              onClick={() => handleSend(q)}
              style={{ whiteSpace: "nowrap", background: "rgba(30,62,98,0.3)", border: "1px solid var(--navy-border)", color: "#CBD5E1", fontSize: "0.74rem", padding: "0.3rem 0.65rem", borderRadius: "9999px", cursor: "pointer", transition: "all 0.2s" }}
              onMouseEnter={(e) => (e.target.style.borderColor = "var(--gold-accent)")}
              onMouseLeave={(e) => (e.target.style.borderColor = "var(--navy-border)")}
            >
              {q.length > 50 ? q.substring(0, 50) + "..." : q}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <form
          className="chat-input-bar"
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
        >
          <input
            className="chat-input"
            type="text"
            placeholder="Ask a legal or policy question (e.g. 'What is the notice requirement for termination?')..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button className="btn-primary" type="submit" disabled={loading || !input.trim()}>
            <Send size={16} />
            <span>Advise</span>
          </button>
        </form>
      </div>

      {/* Right Citation & Evidence Panel */}
      <div className="citation-panel">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid var(--navy-border)", paddingBottom: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <BookOpen size={18} color="var(--gold-accent)" />
            <span style={{ fontSize: "0.95rem", fontWeight: 700 }}>Statutory Citations</span>
          </div>
          <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
            {activeCitations.length} Grounded Source{activeCitations.length !== 1 ? "s" : ""}
          </span>
        </div>

        {activeCitations.length === 0 ? (
          <div style={{ textAlign: "center", padding: "3rem 1rem", color: "var(--text-dim)", fontSize: "0.85rem" }}>
            <Scale size={32} style={{ margin: "0 auto 1rem", opacity: 0.3 }} />
            Ask any question or pick a suggested topic to see the verbatim statutory citations, sections, and page numbers retrieved from ChromaDB.
          </div>
        ) : (
          activeCitations.map((cit, idx) => (
            <div key={idx} className="citation-card">
              <div className="citation-header">
                <span className="citation-doc">{cit.document_title}</span>
                <span className="citation-page">Page {cit.page_number}</span>
              </div>

              <div className="citation-sec">{cit.section_header || "Statutory Provision"}</div>
              <div className="citation-snippet">"{cit.text_snippet}"</div>

              <div style={{ marginTop: "0.5rem", display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--text-muted)" }}>
                <span>Relevance Score</span>
                <span style={{ color: "var(--gold-accent)", fontWeight: 700 }}>
                  {Math.round(cit.relevance_score * 100)}%
                </span>
              </div>
            </div>
          ))
        )}

        <div style={{ marginTop: "auto", padding: "0.75rem", background: "rgba(0,0,0,0.25)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.04)", fontSize: "0.72rem", color: "var(--text-dim)" }}>
          <AlertCircle size={14} style={{ display: "inline", verticalAlign: "middle", marginRight: "4px", color: "var(--gold-accent)" }} />
          Every answer is anchored directly to the vectorized legal corpus using ChromaDB cosine indexing.
        </div>
      </div>
    </div>
  );
}
