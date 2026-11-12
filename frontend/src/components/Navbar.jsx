import React from "react";
import { Scale, BookOpen, MessageSquare, ShieldCheck, FileText, KeyRound } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, health, onOpenApiKeyModal }) {
  return (
    <header className="navbar">
      <div className="navbar-top">
        <div className="brand">
          <div className="brand-icon">
            <Scale size={22} />
          </div>
          <div className="brand-text">
            <div className="brand-title">
              Nyaya<span>Sathi</span>
            </div>
            <div className="brand-sub">AI Legal & Policy Assistant • न्यायसाथी</div>
          </div>
        </div>

        <div className="nav-actions">
          <div className="status-pill" title={`Collection: ${health?.chroma_collection || 'Active'}`}>
            <span className="status-dot"></span>
            <span className="status-pill-text">ChromaDB: {health?.total_chunks || 0} Chunks</span>
            <span className="status-pill-mobile">{health?.total_chunks || 0} Chunks</span>
          </div>

          <button className="btn-gemini-key" onClick={onOpenApiKeyModal}>
            <KeyRound size={14} />
            <span className="gemini-key-text">{health?.gemini_configured ? "Gemini Live" : "Set Gemini Key"}</span>
            <span className="gemini-key-mobile">{health?.gemini_configured ? "Live" : "Key"}</span>
          </button>
        </div>
      </div>

      <nav className="nav-tabs" aria-label="Main Navigation">
        <button
          className={`tab-btn ${activeTab === "chat" ? "active" : ""}`}
          onClick={() => setActiveTab("chat")}
        >
          <MessageSquare size={16} />
          <span>Legal Q&A & RAG</span>
        </button>

        <button
          className={`tab-btn ${activeTab === "documents" ? "active" : ""}`}
          onClick={() => setActiveTab("documents")}
        >
          <BookOpen size={16} />
          <span>Document Library ({health?.total_documents || 0})</span>
        </button>

        <button
          className={`tab-btn ${activeTab === "clause" ? "active" : ""}`}
          onClick={() => setActiveTab("clause")}
        >
          <FileText size={16} />
          <span>Clause Risk Analyzer</span>
        </button>

        <button
          className={`tab-btn ${activeTab === "policy" ? "active" : ""}`}
          onClick={() => setActiveTab("policy")}
        >
          <ShieldCheck size={16} />
          <span>Policy Auditor</span>
        </button>
      </nav>
    </header>
  );
}
