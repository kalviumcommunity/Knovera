'use client';

import React, { useState, useEffect } from 'react';
import { AdminSettings } from '@/lib/types';
import {
  Save,
  CheckCircle2,
  Bot,
  MessageSquare,
  Database,
  ShieldCheck,
  Server,
  Sliders,
  Sparkles,
  RotateCcw,
} from 'lucide-react';

const DEFAULT_SETTINGS: AdminSettings = {
  model: {
    primaryModel: 'claude-3-5-sonnet',
    temperature: 0.2,
    maxTokens: 1024,
    systemPrompt:
      'You are Knovera, an enterprise grounded AI assistant. Always ground your factual answers strictly in the retrieved context documents. If the context does not contain sufficient facts to answer, politely state that you cannot verify the answer.',
  },
  chat: {
    sessionTimeoutMins: 60,
    streamingEnabled: true,
    feedbackEnabled: true,
    defaultK: 4,
  },
  retrieval: {
    similarityThreshold: 0.70,
    hybridAlpha: 0.65,
    rerankerEnabled: true,
    contextCompression: true,
  },
  guardrails: {
    strictness: 'standard',
    piiMaskType: 'redacted',
    autoRefusalNotice:
      'I cannot find sufficient evidence in enterprise documentation to verify an answer to this question.',
  },
  system: {
    backendUrl: 'http://127.0.0.1:8000',
    chromaHost: 'local_sqlite_embedded',
    healthPollSec: 30,
    organizationName: 'Knovera Enterprise Systems',
  },
};

export default function SettingsView() {
  const [settings, setSettings] = useState<AdminSettings>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('knovera_settings');
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch {}
      }
    }
    return DEFAULT_SETTINGS;
  });

  const [activeTab, setActiveTab] = useState<'model' | 'chat' | 'retrieval' | 'guardrails' | 'system'>('model');
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    localStorage.setItem('knovera_settings', JSON.stringify(settings));
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const handleReset = () => {
    setSettings(DEFAULT_SETTINGS);
    localStorage.setItem('knovera_settings', JSON.stringify(DEFAULT_SETTINGS));
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2500);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1
            style={{
              fontSize: '22px',
              fontWeight: 700,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
            }}
          >
            Platform & Engine Settings
          </h1>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            Configure model inference parameters, retrieval thresholds, safety strictness, and infrastructure.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            type="button"
            onClick={handleReset}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '7px 12px',
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              fontSize: '13px',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
            }}
          >
            <RotateCcw size={14} />
            <span>Reset Defaults</span>
          </button>

          <button
            type="button"
            onClick={handleSave}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '7px 18px',
              backgroundColor: 'var(--accent-primary)',
              color: '#ffffff',
              borderRadius: 'var(--radius-md)',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <Save size={14} />
            <span>Save Settings</span>
          </button>
        </div>
      </div>

      {saveSuccess && (
        <div
          className="animate-fade-in"
          style={{
            padding: '10px 14px',
            backgroundColor: 'var(--status-success-bg)',
            border: '1px solid var(--status-success-border)',
            borderRadius: 'var(--radius-md)',
            fontSize: '13px',
            color: 'var(--status-success)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <CheckCircle2 size={16} />
          <span>Settings saved successfully. Changes deployed to local runtime.</span>
        </div>
      )}

      {/* Settings Navigation Tabs */}
      <div
        style={{
          display: 'flex',
          gap: '4px',
          borderBottom: '1px solid var(--border-light)',
          paddingBottom: '2px',
        }}
      >
        {[
          { id: 'model', label: 'AI Model & Generation', icon: <Bot size={15} /> },
          { id: 'chat', label: 'Chat Experience', icon: <MessageSquare size={15} /> },
          { id: 'retrieval', label: 'Knowledge Retrieval (RAG)', icon: <Database size={15} /> },
          { id: 'guardrails', label: 'Guardrail Strictness', icon: <ShieldCheck size={15} /> },
          { id: 'system', label: 'System & Connectivity', icon: <Server size={15} /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 14px',
              fontSize: '13px',
              fontWeight: activeTab === tab.id ? 600 : 500,
              color: activeTab === tab.id ? 'var(--accent-primary)' : 'var(--text-secondary)',
              borderBottom: activeTab === tab.id ? '2px solid var(--accent-primary)' : '2px solid transparent',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Settings Card */}
      <form
        onSubmit={handleSave}
        style={{
          backgroundColor: '#ffffff',
          border: '1px solid var(--border-light)',
          borderRadius: 'var(--radius-lg)',
          padding: '24px',
          boxShadow: 'var(--shadow-xs)',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
        }}
      >
        {/* 1. MODEL CONFIGURATION */}
        {activeTab === 'model' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  PRIMARY CHAT MODEL
                </label>
                <select
                  value={settings.model.primaryModel}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      model: { ...prev.model, primaryModel: e.target.value },
                    }))
                  }
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    padding: '8px 12px',
                    fontSize: '13.5px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                  }}
                >
                  <option value="claude-3-5-sonnet">Claude 3.5 Sonnet (Recommended)</option>
                  <option value="gpt-4o">GPT-4o</option>
                  <option value="llama-3.3-70b">Llama 3.3 70B (Self-Hosted)</option>
                  <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                </select>
              </div>

              <div>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  MAX OUTPUT TOKENS
                </label>
                <input
                  type="number"
                  min={128}
                  max={4096}
                  value={settings.model.maxTokens}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      model: { ...prev.model, maxTokens: parseInt(e.target.value) || 1024 },
                    }))
                  }
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    padding: '8px 12px',
                    fontSize: '13.5px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                  }}
                />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  TEMPERATURE ({settings.model.temperature})
                </label>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Lower = more factual & deterministic
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.model.temperature}
                onChange={(e) =>
                  setSettings((prev) => ({
                    ...prev,
                    model: { ...prev.model, temperature: parseFloat(e.target.value) },
                  }))
                }
                style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                CORE SYSTEM INSTRUCTION
              </label>
              <textarea
                rows={4}
                value={settings.model.systemPrompt}
                onChange={(e) =>
                  setSettings((prev) => ({
                    ...prev,
                    model: { ...prev.model, systemPrompt: e.target.value },
                  }))
                }
                style={{
                  width: '100%',
                  marginTop: '6px',
                  padding: '10px 12px',
                  fontSize: '13.5px',
                  lineHeight: 1.5,
                  border: '1px solid var(--border-light)',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'var(--bg-app)',
                  color: 'var(--text-primary)',
                  fontFamily: 'inherit',
                }}
              />
            </div>
          </div>
        )}

        {/* 2. CHAT CONFIGURATION */}
        {activeTab === 'chat' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  SESSION INACTIVITY TIMEOUT (MINUTES)
                </label>
                <input
                  type="number"
                  value={settings.chat.sessionTimeoutMins}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      chat: { ...prev.chat, sessionTimeoutMins: parseInt(e.target.value) || 60 },
                    }))
                  }
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    padding: '8px 12px',
                    fontSize: '13.5px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  DEFAULT RETRIEVAL K CHUNKS
                </label>
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={settings.chat.defaultK}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      chat: { ...prev.chat, defaultK: parseInt(e.target.value) || 4 },
                    }))
                  }
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    padding: '8px 12px',
                    fontSize: '13.5px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                  }}
                />
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 14px',
                backgroundColor: 'var(--bg-app)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-light)',
              }}
            >
              <div>
                <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Enable User Thumbs Up / Down Feedback
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Allow end users to rate AI responses and submit improvement feedback
                </div>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={settings.chat.feedbackEnabled}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      chat: { ...prev.chat, feedbackEnabled: e.target.checked },
                    }))
                  }
                />
                <span className="toggle-slider" />
              </label>
            </div>
          </div>
        )}

        {/* 3. KNOWLEDGE RETRIEVAL CONFIGURATION */}
        {activeTab === 'retrieval' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  COSINE SIMILARITY THRESHOLD ({settings.retrieval.similarityThreshold})
                </label>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Chunks below this score are filtered to prevent hallucination
                </span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={settings.retrieval.similarityThreshold}
                onChange={(e) =>
                  setSettings((prev) => ({
                    ...prev,
                    retrieval: { ...prev.retrieval, similarityThreshold: parseFloat(e.target.value) },
                  }))
                }
                style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  HYBRID SEARCH ALPHA ({settings.retrieval.hybridAlpha})
                </label>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  0.0 = Pure BM25 Keyword ⇄ 1.0 = Pure Dense Vector
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.retrieval.hybridAlpha}
                onChange={(e) =>
                  setSettings((prev) => ({
                    ...prev,
                    retrieval: { ...prev.retrieval, hybridAlpha: parseFloat(e.target.value) },
                  }))
                }
                style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
              />
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 14px',
                backgroundColor: 'var(--bg-app)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-light)',
              }}
            >
              <div>
                <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Context Compression & Deduplication
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Remove overlapping text across adjacent chunks before prompt injection
                </div>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={settings.retrieval.contextCompression}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      retrieval: { ...prev.retrieval, contextCompression: e.target.checked },
                    }))
                  }
                />
                <span className="toggle-slider" />
              </label>
            </div>
          </div>
        )}

        {/* 4. GUARDRAILS CONFIGURATION */}
        {activeTab === 'guardrails' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  SAFETY STRICTNESS LEVEL
                </label>
                <select
                  value={settings.guardrails.strictness}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      guardrails: { ...prev.guardrails, strictness: e.target.value as any },
                    }))
                  }
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    padding: '8px 12px',
                    fontSize: '13.5px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                  }}
                >
                  <option value="low">Low (Permissive)</option>
                  <option value="standard">Standard Enterprise (Recommended)</option>
                  <option value="strict">Strict (Financial & Healthcare Grade)</option>
                </select>
              </div>

              <div>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  PII MASKING FORMAT
                </label>
                <select
                  value={settings.guardrails.piiMaskType}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      guardrails: { ...prev.guardrails, piiMaskType: e.target.value as any },
                    }))
                  }
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    padding: '8px 12px',
                    fontSize: '13.5px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                  }}
                >
                  <option value="redacted">[REDACTED_ENTITY]</option>
                  <option value="asterisk">Asterisks (***-**-****)</option>
                  <option value="hash">Cryptographic SHA Hash</option>
                </select>
              </div>
            </div>

            <div>
              <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                AUTO-REFUSAL NOTICE TEXT
              </label>
              <input
                value={settings.guardrails.autoRefusalNotice}
                onChange={(e) =>
                  setSettings((prev) => ({
                    ...prev,
                    guardrails: { ...prev.guardrails, autoRefusalNotice: e.target.value },
                  }))
                }
                style={{
                  width: '100%',
                  marginTop: '6px',
                  padding: '8px 12px',
                  fontSize: '13.5px',
                  border: '1px solid var(--border-light)',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'var(--bg-app)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
          </div>
        )}

        {/* 5. SYSTEM & CONNECTIVITY */}
        {activeTab === 'system' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  FASTAPI BACKEND URL
                </label>
                <input
                  value={settings.system.backendUrl}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      system: { ...prev.system, backendUrl: e.target.value },
                    }))
                  }
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    padding: '8px 12px',
                    fontSize: '13.5px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  ORGANIZATION NAME
                </label>
                <input
                  value={settings.system.organizationName}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      system: { ...prev.system, organizationName: e.target.value },
                    }))
                  }
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    padding: '8px 12px',
                    fontSize: '13.5px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                  }}
                />
              </div>
            </div>

            <div>
              <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                CHROMADB HOST STORAGE TARGET
              </label>
              <input
                disabled
                value={settings.system.chromaHost}
                style={{
                  width: '100%',
                  marginTop: '6px',
                  padding: '8px 12px',
                  fontSize: '13.5px',
                  border: '1px solid var(--border-light)',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'var(--bg-muted)',
                  color: 'var(--text-muted)',
                }}
              />
            </div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: '10px' }}>
          <button
            type="submit"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 20px',
              backgroundColor: 'var(--accent-primary)',
              color: '#ffffff',
              borderRadius: 'var(--radius-md)',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <Save size={15} />
            <span>Save Configuration</span>
          </button>
        </div>
      </form>
    </div>
  );
}
