'use client';

import React, { useState, useEffect } from 'react';
import Header from '@/components/Header';
import QueryView from '@/components/QueryView';
import ChatView from '@/components/ChatView';
import DocumentsView from '@/components/DocumentsView';
import EvaluationView from '@/components/EvaluationView';
import { api } from '@/lib/api';
import { HealthResponse } from '@/lib/types';
import {
  Search,
  MessageSquare,
  FolderOpen,
  LineChart,
} from 'lucide-react';

type ActiveTab = 'query' | 'chat' | 'documents' | 'evaluation';

export default function Home() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('query');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  const checkHealth = async () => {
    setHealthLoading(true);
    try {
      const data = await api.getHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
    // Poll health status periodically
    const timer = setInterval(checkHealth, 30000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="app-container">
      {/* Header */}
      <Header
        health={health}
        loading={healthLoading}
        onRefresh={checkHealth}
      />

      {/* Main Navigation Tabs */}
      <nav className="tabs-nav">
        <button
          onClick={() => setActiveTab('query')}
          className={`tab-btn ${activeTab === 'query' ? 'active' : ''}`}
        >
          <Search size={16} />
          <span>Grounded Search</span>
        </button>

        <button
          onClick={() => setActiveTab('chat')}
          className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
        >
          <MessageSquare size={16} />
          <span>Conversational Chat</span>
        </button>

        <button
          onClick={() => setActiveTab('documents')}
          className={`tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
        >
          <FolderOpen size={16} />
          <span>Knowledge Documents</span>
        </button>

        <button
          onClick={() => setActiveTab('evaluation')}
          className={`tab-btn ${activeTab === 'evaluation' ? 'active' : ''}`}
        >
          <LineChart size={16} />
          <span>Quality Evaluation</span>
        </button>
      </nav>

      {/* Main View Area */}
      <main style={{ flex: 1 }}>
        {activeTab === 'query' && <QueryView />}
        {activeTab === 'chat' && <ChatView />}
        {activeTab === 'documents' && (
          <DocumentsView
            onSelectDocForQuery={() => {
              setActiveTab('query');
            }}
          />
        )}
        {activeTab === 'evaluation' && <EvaluationView />}
      </main>

      {/* Footer */}
      <footer
        style={{
          marginTop: '60px',
          paddingTop: '20px',
          borderTop: '1px solid var(--border-subtle)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          color: 'var(--text-muted)',
          fontSize: '12px',
        }}
      >
        <span>Knovera RAG Engine & Studio • FastAPI Backend + Next.js Frontend</span>
        <span>Local Port 3000 (UI) ⇄ Port 8000 (API)</span>
      </footer>
    </div>
  );
}
