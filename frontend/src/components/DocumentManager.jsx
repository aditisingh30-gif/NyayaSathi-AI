import React, { useState } from "react";
import { Upload, FileText, Trash2, RefreshCw, CheckCircle, AlertTriangle, Layers, BookCheck } from "lucide-react";
import { uploadDocument, deleteDocument, preloadSamples } from "../services/api";

const CATEGORIES = [
  "Statutory Act",
  "Commercial Contract",
  "Criminal Statute",
  "Corporate Policy",
  "Regulatory Circular",
  "Court Judgment",
  "General Legal"
];

export default function DocumentManager({ documents = [], onRefresh }) {
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState("");
  const [category, setCategory] = useState("Statutory Act");
  const [uploading, setUploading] = useState(false);
  const [preloading, setPreloading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setStatusMsg(null);

    try {
      const res = await uploadDocument(file, title, category);
      setStatusMsg({
        type: "success",
        text: `Successfully ingested "${res.title}": ${res.num_pages} pages, ${res.num_chunks} vector chunks indexed in ChromaDB.`
      });
      setFile(null);
      setTitle("");
      onRefresh();
    } catch (err) {
      setStatusMsg({
        type: "error",
        text: `Upload failed: ${err.message}`
      });
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId, docTitle) => {
    if (!window.confirm(`Are you sure you want to delete "${docTitle}" from the legal corpus?`)) return;
    try {
      await deleteDocument(docId);
      onRefresh();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  const handlePreload = async () => {
    setPreloading(true);
    try {
      await preloadSamples();
      setStatusMsg({
        type: "success",
        text: "Preloaded Indian benchmark statutes (DPDP Act 2023, Indian Contract Act 1872, BNS 2023, Enterprise MSA) into ChromaDB."
      });
      onRefresh();
    } catch (err) {
      setStatusMsg({ type: "error", text: `Preload failed: ${err.message}` });
    } finally {
      setPreloading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* Upload & Benchmark Controls */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Upload size={20} color="var(--gold-accent)" />
            <span>Ingest New Legal Document (PDF)</span>
          </div>

          <button
            className="btn-secondary"
            onClick={handlePreload}
            disabled={preloading}
          >
            <RefreshCw size={14} className={preloading ? "animate-spin" : ""} />
            <span>{preloading ? "Indexing..." : "Reload Benchmark Statutes"}</span>
          </button>
        </div>

        {statusMsg && (
          <div
            style={{
              padding: "0.85rem 1.25rem",
              borderRadius: "8px",
              marginBottom: "1.25rem",
              display: "flex",
              alignItems: "center",
              gap: "0.75rem",
              background: statusMsg.type === "success" ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
              border: `1px solid ${statusMsg.type === "success" ? "#10B981" : "#EF4444"}`,
              color: statusMsg.type === "success" ? "#A7F3D0" : "#FCA5A5",
              fontSize: "0.86rem"
            }}
          >
            {statusMsg.type === "success" ? <CheckCircle size={18} /> : <AlertTriangle size={18} />}
            <span>{statusMsg.text}</span>
          </div>
        )}

        <form onSubmit={handleUpload} className="doc-upload-form">
          <div>
            <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Select Legal PDF Document *
            </label>
            <input
              type="file"
              accept=".pdf"
              required
              onChange={(e) => {
                const f = e.target.files[0];
                setFile(f);
                if (f && !title) {
                  setTitle(f.name.replace(/\.pdf$/i, "").replace(/_/g, " "));
                }
              }}
              style={{
                width: "100%",
                background: "#081220",
                border: "1px solid var(--navy-border)",
                padding: "0.6rem",
                borderRadius: "6px",
                color: "#E2E8F0",
                fontSize: "0.84rem"
              }}
            />
          </div>

          <div>
            <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Document Title / Act Name
            </label>
            <input
              type="text"
              placeholder="e.g. Arbitration & Conciliation Act 1996"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              style={{
                width: "100%",
                background: "#081220",
                border: "1px solid var(--navy-border)",
                padding: "0.65rem 0.85rem",
                borderRadius: "6px",
                color: "#E2E8F0",
                fontSize: "0.84rem"
              }}
            />
          </div>

          <div>
            <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Legal Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
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
              {CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>

          <button className="btn-primary doc-upload-btn" type="submit" disabled={uploading || !file}>
            <Upload size={16} />
            <span>{uploading ? "Extracting..." : "Ingest & Index"}</span>
          </button>
        </form>
      </div>

      {/* Document Library Grid */}
      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <BookCheck size={20} color="var(--gold-accent)" />
            Active Legal Corpus ({documents.length} Documents)
          </h2>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Processed with PyMuPDF & Vectorized into ChromaDB
          </span>
        </div>

        {documents.length === 0 ? (
          <div className="card" style={{ textAlign: "center", padding: "3rem 1rem", color: "var(--text-muted)" }}>
            <Layers size={40} style={{ margin: "0 auto 1rem", opacity: 0.3 }} />
            <p style={{ fontSize: "0.95rem" }}>No legal documents currently indexed.</p>
            <button className="btn-primary" style={{ marginTop: "1rem" }} onClick={handlePreload}>
              Load Benchmark Legal Statutes
            </button>
          </div>
        ) : (
          <div className="doc-grid">
            {documents.map((doc) => (
              <div key={doc.doc_id} className="doc-card">
                <div>
                  <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.5rem", marginBottom: "0.75rem" }}>
                    <span className="badge badge-low" style={{ background: "rgba(229,184,105,0.15)", borderColor: "var(--gold-accent)", color: "var(--gold-accent)" }}>
                      {doc.category || "Legal Document"}
                    </span>
                    <button
                      onClick={() => handleDelete(doc.doc_id, doc.title)}
                      title="Delete document and chunks"
                      style={{ background: "transparent", border: "none", color: "#94A3B8", cursor: "pointer", transition: "color 0.2s" }}
                      onMouseEnter={(e) => (e.target.style.color = "#EF4444")}
                      onMouseLeave={(e) => (e.target.style.color = "#94A3B8")}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>

                  <h3 style={{ fontSize: "1.02rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "0.4rem", lineHeight: 1.4 }}>
                    {doc.title}
                  </h3>
                  <div style={{ fontSize: "0.76rem", color: "var(--text-dim)", wordBreak: "break-all", marginBottom: "0.75rem" }}>
                    📄 {doc.filename}
                  </div>

                  {doc.summary && (
                    <p style={{ fontSize: "0.78rem", color: "#94A3B8", lineHeight: 1.45, marginBottom: "0.85rem" }}>
                      {doc.summary}
                    </p>
                  )}
                </div>

                <div style={{ paddingTop: "0.85rem", borderTop: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "0.76rem", color: "var(--text-muted)" }}>
                  <div style={{ display: "flex", gap: "0.85rem" }}>
                    <span>📖 <strong>{doc.num_pages}</strong> Pages</span>
                    <span>🧩 <strong>{doc.num_chunks}</strong> Chunks</span>
                  </div>
                  <span>{new Date(doc.uploaded_at).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
