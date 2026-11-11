import React, { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import ChatInterface from "./components/ChatInterface";
import DocumentManager from "./components/DocumentManager";
import ClauseAnalyzer from "./components/ClauseAnalyzer";
import PolicyAuditor from "./components/PolicyAuditor";
import ApiKeyModal from "./components/ApiKeyModal";
import { getHealth, getDocuments } from "./services/api";
import { AlertCircle } from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [apiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  const [loadingInitial, setLoadingInitial] = useState(true);

  const loadData = async () => {
    try {
      const [hData, dData] = await Promise.all([
        getHealth().catch(() => null),
        getDocuments().catch(() => ({ documents: [] })),
      ]);
      if (hData) setHealth(hData);
      if (dData && dData.documents) setDocuments(dData.documents);
    } finally {
      setLoadingInitial(false);
    }
  };

  useEffect(() => {
    loadData();
    // Poll health periodically
    const interval = setInterval(loadData, 8000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        onOpenApiKeyModal={() => setApiKeyModalOpen(true)}
      />

      <main className="main-content">
        {/* Statutory Legal Disclaimer Banner */}
        <div className="disclaimer-banner">
          <AlertCircle size={18} color="var(--gold-accent)" style={{ flexShrink: 0 }} />
          <div>
            <strong>Bar Council & Statutory Advisory Notice:</strong> NyayaSathi is an AI-powered legal and policy research tool designed for informational and analytical assistance. It does not constitute formal legal representation or an attorney-client relationship under the Advocates Act, 1961. Always consult an enrolled advocate or qualified legal practitioner.
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === "chat" && (
          <ChatInterface documents={documents} />
        )}

        {activeTab === "documents" && (
          <DocumentManager
            documents={documents}
            onRefresh={loadData}
          />
        )}

        {activeTab === "clause" && (
          <ClauseAnalyzer />
        )}

        {activeTab === "policy" && (
          <PolicyAuditor documents={documents} />
        )}
      </main>

      <ApiKeyModal
        isOpen={apiKeyModalOpen}
        onClose={() => setApiKeyModalOpen(false)}
        onKeyUpdated={loadData}
      />
    </div>
  );
}
