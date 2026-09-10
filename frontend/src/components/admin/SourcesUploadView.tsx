'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import {
  UploadCloud,
  FileText,
  Trash2,
  RefreshCw,
  CheckCircle2,
  Clock,
  AlertCircle,
  FileUp,
  Search,
  Loader2,
  Database,
} from 'lucide-react';

interface UploadedSource {
  id: string;
  filename: string;
  format: 'PDF' | 'DOCX' | 'TXT' | 'CSV' | 'MD' | 'JSON';
  sizeBytes: number;
  uploadDate: string;
  chunkCount: number;
  status: 'indexed' | 'indexing' | 'failed';
}

export default function SourcesUploadView() {
  const [sources, setSources] = useState<UploadedSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadingFilename, setUploadingFilename] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [replacingSource, setReplacingSource] = useState<UploadedSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const replaceInputRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getDocuments();
      if (res.documents) {
        const backendSources: UploadedSource[] = res.documents.map((d, idx) => ({
          id: `doc_${idx}_${d.filename}`,
          filename: d.filename,
          format: (d.filename.split('.').pop()?.toUpperCase() || 'TXT') as any,
          sizeBytes: (d.chunk_count || 5) * 1400,
          uploadDate: d.indexed_at || 'Indexed',
          chunkCount: d.chunk_count,
          status: 'indexed',
        }));
        setSources(backendSources);
      } else {
        setSources([]);
      }
    } catch (err: any) {
      console.error('Failed to load documents from database:', err);
      setError(err?.message || 'Failed to fetch documents from database.');
      setSources([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  const handleFileUpload = async (file: File, replaceTarget?: UploadedSource) => {
    setUploadingFilename(file.name);
    setUploadProgress(20);
    setError(null);

    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev === null) return 20;
        if (prev >= 85) {
          clearInterval(progressInterval);
          return 85;
        }
        return prev + 20;
      });
    }, 180);

    try {
      // If replacing, delete previous version first
      if (replaceTarget && replaceTarget.filename !== file.name) {
        try {
          await api.deleteDocument(replaceTarget.filename);
        } catch {
          // continue with upload
        }
      }

      await api.uploadDocument(file);
      clearInterval(progressInterval);
      setUploadProgress(100);

      setTimeout(async () => {
        setUploadProgress(null);
        setUploadingFilename(null);
        setReplacingSource(null);
        await fetchDocs();
      }, 500);
    } catch (err: any) {
      clearInterval(progressInterval);
      setUploadProgress(null);
      setUploadingFilename(null);
      setReplacingSource(null);
      setError(err?.message || 'Failed to upload and vectorize document.');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleDelete = async (source: UploadedSource) => {
    const confirmDelete = window.confirm(`Are you sure you want to remove "${source.filename}" from the database and vector index?`);
    if (!confirmDelete) return;

    try {
      await api.deleteDocument(source.filename);
      setSources((prev) => prev.filter((s) => s.filename !== source.filename));
    } catch (err: any) {
      console.error('Delete failed:', err);
      setError(err?.message || 'Failed to delete document from database.');
    }
  };

  const handleStartReplace = (source: UploadedSource) => {
    setReplacingSource(source);
    replaceInputRef.current?.click();
  };

  const filteredSources = sources.filter((s) =>
    s.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatFileSize = (bytes: number) => {
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(0)} KB`;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Hidden File Inputs */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => {
          if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
          e.target.value = '';
        }}
        style={{ display: 'none' }}
        accept=".pdf,.docx,.txt,.csv,.md,.json"
      />
      <input
        type="file"
        ref={replaceInputRef}
        onChange={(e) => {
          if (e.target.files?.[0] && replacingSource) {
            handleFileUpload(e.target.files[0], replacingSource);
          }
          e.target.value = '';
        }}
        style={{ display: 'none' }}
        accept=".pdf,.docx,.txt,.csv,.md,.json"
      />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1
              style={{
                fontSize: '22px',
                fontWeight: 700,
                letterSpacing: '-0.02em',
                color: 'var(--text-primary)',
              }}
            >
              Sources & Document Upload
            </h1>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '11px',
                padding: '2px 8px',
                backgroundColor: 'var(--status-success-bg)',
                color: 'var(--status-success)',
                border: '1px solid var(--status-success-border)',
                borderRadius: 'var(--radius-full)',
                fontWeight: 600,
              }}
            >
              <Database size={11} />
              <span>MongoDB Vector Store</span>
            </span>
          </div>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            Ingest enterprise documents, policies, and contracts directly into MongoDB and SQLite.
          </p>
        </div>

        <button
          onClick={fetchDocs}
          disabled={loading}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            padding: '7px 12px',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-md)',
            fontSize: '13px',
            fontWeight: 500,
            color: 'var(--text-primary)',
            cursor: loading ? 'not-allowed' : 'pointer',
            boxShadow: 'var(--shadow-xs)',
          }}
        >
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          <span>{loading ? 'Refreshing...' : 'Refresh Sources'}</span>
        </button>
      </div>

      {error && (
        <div
          style={{
            padding: '12px 16px',
            backgroundColor: 'var(--status-danger-bg)',
            border: '1px solid var(--status-danger-border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--status-danger)',
            fontSize: '13px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Drag and Drop Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        style={{
          border: isDragging ? '2px dashed var(--accent-primary)' : '2px dashed var(--border-medium)',
          backgroundColor: isDragging ? 'var(--accent-primary-subtle)' : '#ffffff',
          borderRadius: 'var(--radius-lg)',
          padding: '36px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.18s ease',
          boxShadow: 'var(--shadow-xs)',
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: 'var(--radius-full)',
            backgroundColor: 'var(--bg-muted)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 12px auto',
            color: 'var(--accent-primary)',
          }}
        >
          <UploadCloud size={24} />
        </div>

        <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
          Click to upload or drag & drop files here
        </div>
        <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', marginTop: '4px' }}>
          Supports PDF, DOCX, TXT, CSV, MD, and JSON documents up to 50 MB
        </p>

        {uploadProgress !== null && (
          <div style={{ maxWidth: '360px', margin: '16px auto 0 auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
              <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                Vectorizing {uploadingFilename}...
              </span>
              <span style={{ color: 'var(--text-secondary)' }}>{uploadProgress}%</span>
            </div>
            <div
              style={{
                width: '100%',
                height: '6px',
                backgroundColor: 'var(--bg-muted)',
                borderRadius: 'var(--radius-full)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${uploadProgress}%`,
                  height: '100%',
                  backgroundColor: 'var(--accent-primary)',
                  transition: 'width 0.2s ease',
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Sources Table Header & Search */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
          Indexed Documents ({sources.length})
        </div>

        <div style={{ position: 'relative', width: '280px' }}>
          <Search
            size={15}
            style={{
              position: 'absolute',
              left: '10px',
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--text-light)',
            }}
          />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter indexed files..."
            style={{
              width: '100%',
              padding: '6px 10px 6px 32px',
              fontSize: '12.5px',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              backgroundColor: '#ffffff',
              outline: 'none',
              color: 'var(--text-primary)',
            }}
          />
        </div>
      </div>

      {/* Sources List Table */}
      <div
        style={{
          backgroundColor: '#ffffff',
          border: '1px solid var(--border-light)',
          borderRadius: 'var(--radius-lg)',
          overflow: 'hidden',
          boxShadow: 'var(--shadow-xs)',
        }}
      >
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
            <thead>
              <tr style={{ backgroundColor: 'var(--bg-app)', borderBottom: '1px solid var(--border-light)' }}>
                <th style={{ padding: '10px 18px', fontWeight: 600, color: 'var(--text-secondary)' }}>File Name</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>Format</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>File Size</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>Chunks</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>Status</th>
                <th style={{ padding: '10px 18px', fontWeight: 600, color: 'var(--text-secondary)', textAlign: 'right' }}>
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6} style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                      <Loader2 size={16} className="animate-spin" color="var(--accent-primary)" />
                      <span>Loading documents from vector storage...</span>
                    </div>
                  </td>
                </tr>
              ) : filteredSources.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    {searchQuery ? 'No documents match your search.' : 'No documents indexed in database yet. Upload a document above.'}
                  </td>
                </tr>
              ) : (
                filteredSources.map((source) => (
                  <tr
                    key={source.id}
                    style={{
                      borderBottom: '1px solid var(--border-subtle)',
                      transition: 'background 0.12s ease',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-app)')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#ffffff')}
                  >
                    <td style={{ padding: '12px 18px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <FileText size={16} color="var(--accent-primary)" />
                        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{source.filename}</span>
                      </div>
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span
                        style={{
                          fontSize: '11px',
                          fontWeight: 600,
                          padding: '2px 6px',
                          borderRadius: 'var(--radius-xs)',
                          backgroundColor: 'var(--bg-muted)',
                          color: 'var(--text-secondary)',
                        }}
                      >
                        {source.format}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>
                      {formatFileSize(source.sizeBytes)}
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-primary)', fontWeight: 500 }}>
                      {source.chunkCount} chunks
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontSize: '11.5px',
                          fontWeight: 600,
                          color: 'var(--status-success)',
                          backgroundColor: 'var(--status-success-bg)',
                          padding: '2px 8px',
                          borderRadius: 'var(--radius-full)',
                        }}
                      >
                        <CheckCircle2 size={12} />
                        Indexed
                      </span>
                    </td>
                    <td style={{ padding: '12px 18px', textAlign: 'right' }}>
                      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '6px' }}>
                        <button
                          onClick={() => handleStartReplace(source)}
                          title="Replace source with updated version"
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            padding: '4px 8px',
                            borderRadius: 'var(--radius-xs)',
                            fontSize: '12px',
                            color: 'var(--text-secondary)',
                            backgroundColor: 'var(--bg-muted)',
                            border: '1px solid var(--border-light)',
                            cursor: 'pointer',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = 'var(--bg-sidebar-active)';
                            e.currentTarget.style.color = 'var(--text-primary)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = 'var(--bg-muted)';
                            e.currentTarget.style.color = 'var(--text-secondary)';
                          }}
                        >
                          <RefreshCw size={12} />
                          <span>Replace</span>
                        </button>

                        <button
                          onClick={() => handleDelete(source)}
                          title="Delete source from database & MongoDB"
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            padding: '4px 8px',
                            borderRadius: 'var(--radius-xs)',
                            fontSize: '12px',
                            color: 'var(--status-danger)',
                            backgroundColor: 'var(--status-danger-bg)',
                            border: '1px solid var(--status-danger-border)',
                            cursor: 'pointer',
                          }}
                          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#fee2e2')}
                          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'var(--status-danger-bg)')}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
