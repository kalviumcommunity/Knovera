'use client';

import React, { useState, useEffect, useRef } from 'react';
import { api } from '@/lib/api';
import { DocumentInfo, DocumentUploadResponse } from '@/lib/types';
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Clock,
  Layers,
  File,
  Sparkles,
  ArrowRight,
} from 'lucide-react';

interface DocumentsViewProps {
  onSelectDocForQuery?: (docName: string) => void;
}

export default function DocumentsView({ onSelectDocForQuery }: DocumentsViewProps) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [totalChunks, setTotalChunks] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<DocumentUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getDocuments();
      setDocuments(data.documents);
      setTotalChunks(data.total_chunks);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch indexed documents');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileUpload = async (file: File) => {
    if (!file) return;

    setUploading(true);
    setError(null);
    setUploadResult(null);

    try {
      const res = await api.uploadDocument(file);
      setUploadResult(res);
      // Refresh documents list
      await fetchDocuments();
    } catch (err: any) {
      setError(err.message || 'File upload and indexing failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Upload Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        className={`glass-panel ${isDragOver ? 'border-accent' : ''}`}
        style={{
          padding: '36px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          gap: '14px',
          borderStyle: 'dashed',
          borderColor: isDragOver ? 'var(--accent-primary)' : 'var(--border-medium)',
          backgroundColor: isDragOver ? 'var(--accent-primary-subtle)' : 'var(--bg-glass)',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".txt,.md,.pdf,.html"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFileUpload(e.target.files[0]);
            }
          }}
        />

        <div
          style={{
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            background: 'var(--accent-primary-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--accent-primary)',
          }}
        >
          {uploading ? (
            <Sparkles size={28} className="animate-spin" />
          ) : (
            <UploadCloud size={28} />
          )}
        </div>

        <div>
          <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '4px' }}>
            {uploading
              ? 'Cleaning, Chunking, Embedding & Indexing...'
              : 'Drop document here or click to browse'}
          </h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Supports Markdown (.md), Plain Text (.txt), PDF (.pdf), and HTML (.html)
          </p>
        </div>

        <span className="badge badge-indigo" style={{ fontSize: '11px' }}>
          Instantly indexed into ChromaDB vector database
        </span>
      </div>

      {/* Upload Result Toast / Card */}
      {uploadResult && (
        <div
          className="animate-fade-in glass-panel"
          style={{
            padding: '18px 24px',
            background: 'var(--accent-emerald-subtle)',
            borderColor: 'rgba(16, 185, 129, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '16px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <CheckCircle2 size={22} color="var(--accent-emerald)" />
            <div>
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#6ee7b7' }}>
                {uploadResult.message}
              </h4>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                File: <strong>{uploadResult.filename}</strong> • Chunks Created: <strong>{uploadResult.summary.chunks}</strong> • Indexed: <strong>{uploadResult.summary.indexed}</strong>
              </p>
            </div>
          </div>
          <button onClick={() => setUploadResult(null)} className="btn-ghost" style={{ fontSize: '12px' }}>
            Dismiss
          </button>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div
          className="animate-fade-in"
          style={{
            padding: '16px 20px',
            background: 'var(--accent-rose-subtle)',
            border: '1px solid rgba(244, 63, 94, 0.4)',
            borderRadius: 'var(--radius-md)',
            color: '#fecdd3',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontSize: '14px',
          }}
        >
          <AlertTriangle size={18} color="#f43f5e" />
          <span>{error}</span>
        </div>
      )}

      {/* Document Library Table */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h2 style={{ fontSize: '16px', fontWeight: 600 }}>Indexed Knowledge Base Documents</h2>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Total indexed chunks across collections: <strong style={{ color: 'var(--text-primary)' }}>{totalChunks}</strong>
            </p>
          </div>

          <button onClick={fetchDocuments} className="btn-ghost" disabled={loading}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            <span>Refresh</span>
          </button>
        </div>

        {documents.length === 0 && !loading ? (
          <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <File size={32} style={{ margin: '0 auto 12px auto', opacity: 0.5 }} />
            <p>No documents uploaded yet. Upload a document above to get started.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '10px 14px' }}>Document Name</th>
                  <th style={{ padding: '10px 14px' }}>Collection</th>
                  <th style={{ padding: '10px 14px' }}>Last Modified / Indexed</th>
                  <th style={{ padding: '10px 14px', textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc, idx) => (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: '1px solid var(--border-subtle)',
                      transition: 'background 0.15s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'var(--bg-surface-hover)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    <td style={{ padding: '12px 14px', fontWeight: 500 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <FileText size={16} color="var(--accent-primary)" />
                        <span>{doc.filename}</span>
                      </div>
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span className="badge badge-cyan" style={{ fontSize: '11px' }}>
                        {doc.collection_name}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>
                      {doc.indexed_at
                        ? new Date(doc.indexed_at).toLocaleDateString(undefined, {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })
                        : '—'}
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'right' }}>
                      {onSelectDocForQuery && (
                        <button
                          onClick={() => onSelectDocForQuery(doc.filename)}
                          className="btn-ghost"
                          style={{ fontSize: '12px' }}
                        >
                          <span>Query This</span>
                          <ArrowRight size={12} />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
