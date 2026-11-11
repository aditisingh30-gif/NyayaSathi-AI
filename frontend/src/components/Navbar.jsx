import React from "react";
import { Scale, BookOpen, MessageSquare, ShieldCheck, FileText, KeyRound, Cpu } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, health, onOpenApiKeyModal }) {
  return (
    <header className="navbar">
      <div className="brand">
        <div className="brand-icon">
          <Scale size={22} />
        </div>
        <div>
          <div className="brand-title">
            Nyaya<span>Sathi</span>
          </div>
          <div className="brand-sub">AI Legal & Policy Assistant • न्यायसाथी</div>
        </div>
      </div>

      <nav className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === "chat" ? "active" : ""}`}
          onClick={() => setActiveTab("chat")}
        >
          <MessageSquare size={16} />
          Legal Q&A & RAG
        </button>

        <button
          className={`tab-btn ${activeTab === "documents" ? "active" : ""}`}
          onClick={() => setActiveTab("documents")}
        >
          <BookOpen size={16} />
          Document Library ({health?.total_documents || 0})
        </button>

        <button
          className={`tab-btn ${activeTab === "clause" ? "active" : ""}`}
          onClick={() => setActiveTab("clause")}
        >
          <FileText size={16} />
          Clause Risk Analyzer
        </button>

        <button
          className={`tab-btn ${activeTab === "policy" ? "active" : ""}`}
          onClick={() => setActiveTab("policy")}
        >
          <ShieldCheck size={16} />
          Policy Auditor
        </button>
      </nav>

      <div className="nav-actions">
        <div className="status-pill" title={`Collection: ${health?.chroma_collection || 'Active'}`}>
          <span className="status-dot"></span>
          <span>ChromaDB: {health?.total_chunks || 0} Chunks</span>
        </div>

        <button className="btn-gemini-key" onClick={onOpenApiKeyModal}>
          <KeyRound size={14} />
          <span>{health?.gemini_configured ? "Gemini Live" : "Set Gemini Key"}</span>
        </button>
      </div>
    </header>
  );
}
