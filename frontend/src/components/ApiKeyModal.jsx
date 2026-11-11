import React, { useState } from "react";
import { X, KeyRound, Check, ExternalLink } from "lucide-react";
import { updateGeminiKey } from "../services/api";

export default function ApiKeyModal({ isOpen, onClose, onKeyUpdated }) {
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gemini-1.5-flash");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  if (!isOpen) return null;

  const handleSave = async (e) => {
    e.preventDefault();
    if (!apiKey.trim()) return;

    setLoading(true);
    setStatus(null);

    try {
      const res = await updateGeminiKey(apiKey, model);
      if (res.status === "success") {
        setStatus({ type: "success", text: res.message });
        setTimeout(() => {
          onKeyUpdated();
          onClose();
        }, 1200);
      } else {
        setStatus({ type: "error", text: res.message });
      }
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <KeyRound size={20} color="var(--gold-accent)" />
            <h3 style={{ fontSize: "1.15rem", fontWeight: 700, color: "#FFFFFF" }}>
              Configure Google Gemini API
            </h3>
          </div>
          <button
            onClick={onClose}
            style={{ background: "transparent", border: "none", color: "#94A3B8", cursor: "pointer" }}
          >
            <X size={20} />
          </button>
        </div>

        <p style={{ fontSize: "0.84rem", color: "var(--text-muted)", marginBottom: "1.25rem", lineHeight: 1.5 }}>
          NyayaSathi uses Google Gemini for deep legal reasoning, citation grounding, and clause redrafting.
          If not provided, the local offline legal intelligence engine operates seamlessly.
        </p>

        {status && (
          <div
            style={{
              padding: "0.75rem",
              borderRadius: "6px",
              marginBottom: "1rem",
              fontSize: "0.82rem",
              background: status.type === "success" ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
              border: `1px solid ${status.type === "success" ? "#10B981" : "#EF4444"}`,
              color: status.type === "success" ? "#A7F3D0" : "#FCA5A5"
            }}
          >
            {status.text}
          </div>
        )}

        <form onSubmit={handleSave} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div>
            <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Gemini API Key *
            </label>
            <input
              type="password"
              placeholder="AIzaSy..."
              required
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              style={{
                width: "100%",
                background: "#081220",
                border: "1px solid var(--navy-border)",
                padding: "0.7rem 0.9rem",
                borderRadius: "6px",
                color: "#FFFFFF",
                fontSize: "0.88rem"
              }}
            />
          </div>

          <div>
            <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Model Tier
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              style={{
                width: "100%",
                background: "#081220",
                border: "1px solid var(--navy-border)",
                padding: "0.7rem 0.9rem",
                borderRadius: "6px",
                color: "#FFFFFF",
                fontSize: "0.88rem"
              }}
            >
              <option value="gemini-1.5-flash">Gemini 1.5 Flash (Ultra Fast & Efficient)</option>
              <option value="gemini-1.5-pro">Gemini 1.5 Pro (Deep Statutory Reasoning)</option>
              <option value="gemini-2.0-flash">Gemini 2.0 Flash (Next-Gen High Speed)</option>
            </select>
          </div>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "0.5rem" }}>
            <a
              href="https://aistudio.google.com/app/apikey"
              target="_blank"
              rel="noreferrer"
              style={{ fontSize: "0.76rem", color: "var(--gold-accent)", display: "flex", alignItems: "center", gap: "0.3rem", textDecoration: "none" }}
            >
              <span>Get Gemini API Key</span>
              <ExternalLink size={12} />
            </a>

            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button
                type="button"
                className="btn-secondary"
                onClick={onClose}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn-primary"
                disabled={loading || !apiKey.trim()}
              >
                {loading ? "Validating..." : "Connect Key"}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
