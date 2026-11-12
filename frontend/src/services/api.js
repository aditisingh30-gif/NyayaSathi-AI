// Use environment variable in production (e.g., on Vercel), fallback to local FastAPI for development
const rawUrl = (import.meta.env.VITE_API_BASE_URL || "").trim();
const API_BASE = rawUrl
  ? `${rawUrl.replace(/\/+$/, "").replace(/\/api$/, "")}/api`
  : "http://localhost:8000/api";

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function getDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function uploadDocument(file, title, category) {
  const formData = new FormData();
  formData.append("file", file);
  if (title) formData.append("title", title);
  if (category) formData.append("category", category);

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "PDF upload failed");
  }
  return res.json();
}

export async function deleteDocument(docId) {
  const res = await fetch(`${API_BASE}/documents/${docId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Delete failed");
  return res.json();
}

export async function preloadSamples() {
  const res = await fetch(`${API_BASE}/documents/preload-samples`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Preload failed");
  return res.json();
}

export async function queryLegalCorpus(question, docIds = null, topK = 5) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      doc_ids: docIds && docIds.length > 0 ? docIds : null,
      top_k: topK,
      include_disclaimer: true,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Legal query failed");
  }
  return res.json();
}

export async function sendChatMessage(messages, docIds = null) {
  const res = await fetch(`${API_BASE}/query/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      doc_ids: docIds && docIds.length > 0 ? docIds : null,
      top_k: 5,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat request failed");
  }
  return res.json();
}

export async function analyzeClause(clauseText, contractType) {
  const res = await fetch(`${API_BASE}/analysis/clause`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      clause_text: clauseText,
      contract_type: contractType || "Commercial Contract",
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Clause analysis failed");
  }
  return res.json();
}

export async function auditPolicy(policyText, framework, docId = null) {
  const res = await fetch(`${API_BASE}/analysis/policy`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      policy_text: policyText,
      framework: framework || "DPDP_ACT_2023",
      doc_id: docId,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Policy audit failed");
  }
  return res.json();
}

export async function updateGeminiKey(apiKey, modelName) {
  const res = await fetch(`${API_BASE}/settings/gemini-key`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_key: apiKey,
      model_name: modelName || "gemini-1.5-flash",
    }),
  });
  if (!res.ok) throw new Error("Failed to set Gemini API key");
  return res.json();
}
