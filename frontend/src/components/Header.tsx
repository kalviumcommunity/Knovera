'use client';

import React from 'react';
import { HealthResponse } from '@/lib/types';
import { Sparkles, RefreshCw, Database, Cpu, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  health: HealthResponse | null;
  loading: boolean;
  onRefresh: () => void;
}

export default function Header({ health, loading, onRefresh }: HeaderProps) {
  const isOnline = health?.status === 'ok';

  return (
    <header className="header-wrapper">
      <div className="brand-section">
        <div className="brand-icon">
          <Sparkles size={24} />
        </div>
        <div className="brand-title-group">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1>Knovera</h1>
            <span className="badge badge-indigo" style={{ fontSize: '10px' }}>
              v1.0 RAG Engine
            </span>
          </div>
          <p>Enterprise Grounded Question-Answering & Conversational Retrieval</p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* Collection & Model info badge */}
        {health && isOnline && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <div className="badge badge-cyan" title="Active Vector DB Collection">
              <Database size={13} />
              <span>{health.collection_name}</span>
              {health.total_indexed_chunks !== undefined && (
                <span style={{ opacity: 0.8 }}>({health.total_indexed_chunks} chunks)</span>
              )}
            </div>
            <div className="badge badge-emerald" title="Embedding & Chat Models">
              <Cpu size={13} />
              <span>{health.embedding_model}</span>
            </div>
          </div>
        )}

        {/* Live Status Pill */}
        <div className="header-status-pill">
          <div
            className={`status-dot ${
              loading ? 'connecting' : isOnline ? 'online' : 'offline'
            }`}
          />
          <span style={{ color: isOnline ? 'var(--text-primary)' : 'var(--accent-rose)' }}>
            {loading
              ? 'Connecting...'
              : isOnline
              ? 'Backend Online'
              : 'Backend Disconnected'}
          </span>
          <button
            onClick={onRefresh}
            className="btn-ghost"
            style={{ padding: '2px 6px', marginLeft: '4px' }}
            title="Refresh backend status"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>
    </header>
  );
}
