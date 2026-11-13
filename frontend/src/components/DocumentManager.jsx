import React, { useState } from "react";
import {
  Upload,
  FileText,
  Trash2,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  Layers,
  BookCheck,
  Cloud,
  RotateCcw,
} from "lucide-react";
import {
  uploadDocument,
  deleteDocument,
  preloadSamples,
  syncDocuments,
  retryDocument,
} from "../services/api";

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
  const [syncing, setSyncing] = useState(false);
  const [retryingDocId, setRetryingDocId] = useState(null);
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
      setStatusMsg({
        type: "success",
        text: `Deleted "${docTitle}" from the database.`
      });
      onRefresh();
    } catch (err) {
      setStatusMsg({
        type: "error",
        text: `Delete failed: ${err.message}`
      });
    }
  };

  const handlePreload = async () => {
    setPreloading(true);
    setStatusMsg(null);
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

  const handleSync = async () => {
    setSyncing(true);
    setStatusMsg(null);
    try {
      const res = await syncDocuments(false);
      if (res.status === "in_progress") {
        setStatusMsg({
          type: "info",
          text: "A cloud synchronization is already in progress. Checking for updates..."
        });
      } else if (res.status === "not_configured") {
        setStatusMsg({
          type: "warning",
          text: "Supabase Storage is not configured. Direct cloud synchronization is unavailable."
        });
      } else {
        setStatusMsg({
          type: res.failed_count > 0 ? "warning" : "success",
          text: res.message || `Sync completed: ${res.synced_count} new document(s) indexed.`
        });
      }
      onRefresh();
    } catch (err) {
      setStatusMsg({
        type: "error",
        text: `Cloud synchronization failed: ${err.message}`
      });
    } finally {
      setSyncing(false);
    }
  };

  const handleRetry = async (docId, docName) => {
    setRetryingDocId(docId);
    setStatusMsg(null);
    try {
      const res = await retryDocument(docId);
      if (res.status === "success") {
        setStatusMsg({
          type: "success",
          text: res.message || `Successfully indexed "${docName}".`
        });
      } else {
        setStatusMsg({
          type: "error",
          text: res.message || `Retry failed for "${docName}".`
        });
      }
      onRefresh();
    } catch (err) {
      setStatusMsg({
        type: "error",
        text: `Retry error: ${err.message}`
      });
    } finally {
      setRetryingDocId(null);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* Upload & Cloud Synchronization Controls */}
      <div className="card">
        <div className="card-header" style={{ flexWrap: "wrap", gap: "0.75rem" }}>
          <div className="card-title">
            <Upload size={20} color="var(--gold-accent)" />
            <span>Ingest & Synchronize Legal Documents</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
            <button
              className="btn-primary"
              onClick={handleSync}
              disabled={syncing}
              style={{
                background: "linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%)",
                borderColor: "#3B82F6",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem"
              }}
              title="Synchronize all PDFs uploaded directly to Supabase Storage"
            >
              <Cloud size={15} className={syncing ? "animate-spin" : ""} />
              <span>{syncing ? "Syncing Supabase..." : "Sync from Cloud"}</span>
            </button>

            <button
              className="btn-secondary"
              onClick={handlePreload}
              disabled={preloading}
              title="Load standard benchmark statutory corpora"
            >
              <RefreshCw size={14} className={preloading ? "animate-spin" : ""} />
              <span>{preloading ? "Indexing..." : "Reload Benchmarks"}</span>
            </button>
          </div>
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
              background:
                statusMsg.type === "success"
                  ? "rgba(16, 185, 129, 0.15)"
                  : statusMsg.type === "warning"
                  ? "rgba(245, 158, 11, 0.15)"
                  : statusMsg.type === "info"
                  ? "rgba(59, 130, 246, 0.15)"
                  : "rgba(239, 68, 68, 0.15)",
              border: `1px solid ${
                statusMsg.type === "success"
                  ? "#10B981"
                  : statusMsg.type === "warning"
                  ? "#F59E0B"
                  : statusMsg.type === "info"
                  ? "#3B82F6"
                  : "#EF4444"
              }`,
              color:
                statusMsg.type === "success"
                  ? "#A7F3D0"
                  : statusMsg.type === "warning"
                  ? "#FDE68A"
                  : statusMsg.type === "info"
                  ? "#93C5FD"
                  : "#FCA5A5",
              fontSize: "0.86rem"
            }}
          >
            {statusMsg.type === "success" ? (
              <CheckCircle size={18} />
            ) : statusMsg.type === "info" ? (
              <RefreshCw size={18} className="animate-spin" />
            ) : (
              <AlertTriangle size={18} />
            )}
            <span style={{ flex: 1 }}>{statusMsg.text}</span>
            <button
              onClick={() => setStatusMsg(null)}
              style={{
                background: "transparent",
                border: "none",
                color: "inherit",
                cursor: "pointer",
                fontWeight: "bold",
                fontSize: "1rem"
              }}
            >
              ×
            </button>
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
              placeholder="e.g. Consumer Protection Act 2019"
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
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
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
            <div style={{ display: "flex", justifyContent: "center", gap: "1rem", marginTop: "1rem", flexWrap: "wrap" }}>
              <button className="btn-primary" onClick={handleSync} disabled={syncing}>
                <Cloud size={15} />
                <span>Sync Supabase Documents</span>
              </button>
              <button className="btn-secondary" onClick={handlePreload} disabled={preloading}>
                Load Benchmark Legal Statutes
              </button>
            </div>
          </div>
        ) : (
          <div className="doc-grid">
            {documents.map((doc) => {
              const status = doc.indexing_status || "indexed";
              const isFailed = status === "failed";
              const isSyncing = status === "syncing";
              const isRetrying = retryingDocId === doc.doc_id;

              return (
                <div
                  key={doc.doc_id}
                  className="doc-card"
                  style={{
                    border: isFailed
                      ? "1px solid rgba(239, 68, 68, 0.4)"
                      : undefined
                  }}
                >
                  <div>
                    {/* Header: Category, Status & Action */}
                    <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.5rem", marginBottom: "0.75rem", flexWrap: "wrap" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", flexWrap: "wrap" }}>
                        <span className="badge badge-low" style={{ background: "rgba(229,184,105,0.15)", borderColor: "var(--gold-accent)", color: "var(--gold-accent)" }}>
                          {doc.category || "Legal Document"}
                        </span>

                        {/* Indexing Status Badge */}
                        {isFailed ? (
                          <span
                            className="badge"
                            style={{
                              background: "rgba(239, 68, 68, 0.18)",
                              borderColor: "#EF4444",
                              color: "#FCA5A5",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "0.25rem"
                            }}
                          >
                            <AlertTriangle size={11} /> Failed
                          </span>
                        ) : isSyncing ? (
                          <span
                            className="badge"
                            style={{
                              background: "rgba(245, 158, 11, 0.18)",
                              borderColor: "#F59E0B",
                              color: "#FDE68A",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "0.25rem"
                            }}
                          >
                            <RefreshCw size={11} className="animate-spin" /> Syncing
                          </span>
                        ) : (
                          <span
                            className="badge"
                            style={{
                              background: "rgba(16, 185, 129, 0.18)",
                              borderColor: "#10B981",
                              color: "#A7F3D0",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "0.25rem"
                            }}
                          >
                            <CheckCircle size={11} /> Indexed
                          </span>
                        )}

                        {/* Source Badge */}
                        {doc.source === "supabase" && (
                          <span
                            className="badge"
                            style={{
                              background: "rgba(59, 130, 246, 0.15)",
                              borderColor: "#3B82F6",
                              color: "#93C5FD",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "0.25rem"
                            }}
                          >
                            <Cloud size={11} /> Cloud
                          </span>
                        )}
                      </div>

                      <button
                        onClick={() => handleDelete(doc.doc_id, doc.title || doc.filename)}
                        title="Delete document and chunks"
                        style={{
                          background: "transparent",
                          border: "none",
                          color: "#94A3B8",
                          cursor: "pointer",
                          transition: "color 0.2s",
                          padding: "0.2rem"
                        }}
                        onMouseEnter={(e) => (e.target.style.color = "#EF4444")}
                        onMouseLeave={(e) => (e.target.style.color = "#94A3B8")}
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>

                    <h3 style={{ fontSize: "1.02rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "0.4rem", lineHeight: 1.4 }}>
                      {doc.title || doc.filename}
                    </h3>
                    <div style={{ fontSize: "0.76rem", color: "var(--text-dim)", wordBreak: "break-all", marginBottom: "0.75rem" }}>
                      📄 {doc.filename}
                    </div>

                    {/* Error message and Retry Button if failed */}
                    {isFailed && (
                      <div
                        style={{
                          background: "rgba(239, 68, 68, 0.12)",
                          border: "1px solid rgba(239, 68, 68, 0.3)",
                          borderRadius: "6px",
                          padding: "0.6rem 0.75rem",
                          marginBottom: "0.85rem",
                          fontSize: "0.76rem",
                          color: "#FCA5A5"
                        }}
                      >
                        <p style={{ margin: 0, marginBottom: "0.5rem" }}>
                          <strong>Indexing failed:</strong> {doc.error_message || "Unknown processing error"}
                        </p>
                        <button
                          className="btn-secondary"
                          onClick={() => handleRetry(doc.doc_id, doc.filename)}
                          disabled={isRetrying}
                          style={{
                            fontSize: "0.72rem",
                            padding: "0.3rem 0.6rem",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "0.35rem"
                          }}
                        >
                          <RotateCcw size={12} className={isRetrying ? "animate-spin" : ""} />
                          <span>{isRetrying ? "Retrying..." : "Retry Indexing"}</span>
                        </button>
                      </div>
                    )}

                    {doc.summary && !isFailed && (
                      <p style={{ fontSize: "0.78rem", color: "#94A3B8", lineHeight: 1.45, marginBottom: "0.85rem" }}>
                        {doc.summary}
                      </p>
                    )}
                  </div>

                  <div style={{ paddingTop: "0.85rem", borderTop: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "0.76rem", color: "var(--text-muted)" }}>
                    <div style={{ display: "flex", gap: "0.85rem" }}>
                      <span>📖 <strong>{doc.num_pages || 0}</strong> Pages</span>
                      <span>🧩 <strong>{doc.num_chunks || 0}</strong> Chunks</span>
                    </div>
                    <span>{doc.uploaded_at ? new Date(doc.uploaded_at).toLocaleDateString() : "Recent"}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
