'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import {
  Search,
  RefreshCw,
  Eye,
  FileText,
  CheckCircle2,
  X,
  Loader2,
  Database,
  Layers,
} from 'lucide-react';

interface KnowledgeChunk {
  id: string;
  sourceDoc: string;
  section: string;
  chunkIndex: number;
  tokenCount: number;
  content: string;
  embeddingModel: string;
  indexedAt: string;
  metadata: Record<string, string>;
}

export default function KnowledgeBaseView() {
  const [chunks, setChunks] = useState<KnowledgeChunk[]>([]);
  const [availableDocs, setAvailableDocs] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFilterDoc, setSelectedFilterDoc] = useState<string>('all');
  const [selectedChunk, setSelectedChunk] = useState<KnowledgeChunk | null>(null);
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshSuccess, setRefreshSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchChunksAndDocs = useCallback(async (search: string, doc: string, isSilent = false) => {
    if (!isSilent) setLoading(true);
    setError(null);
    try {
      // Parallel fetch chunks and indexed docs
      const [chunksRes, docsRes] = await Promise.all([
        api.getChunks({ search: search.trim() || undefined, doc: doc !== 'all' ? doc : undefined }),
        api.getDocuments().catch(() => ({ documents: [], total_indexed: 0 })),
      ]);

      setChunks((chunksRes.chunks || []) as KnowledgeChunk[]);
      
      // Extract unique doc names from backend documents endpoint or fallback to unique docs in chunks
      const docNames = (docsRes.documents || []).map((d) => d.filename);
      const chunkDocNames = (chunksRes.chunks || []).map((c) => c.sourceDoc);
      const uniqueDocs = Array.from(new Set([...docNames, ...chunkDocNames]));
      setAvailableDocs(uniqueDocs);
    } catch (err: any) {
      console.error('Failed to load knowledge base from database:', err);
      setError(err?.message || 'Failed to fetch vectorized chunks from database.');
      setChunks([]);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchChunksAndDocs(searchQuery, selectedFilterDoc);
  }, [fetchChunksAndDocs, selectedFilterDoc]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchChunksAndDocs(searchQuery, selectedFilterDoc);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchChunksAndDocs(searchQuery, selectedFilterDoc, true);
    setRefreshSuccess(true);
    setTimeout(() => setRefreshSuccess(false), 3000);
  };

  // Dynamic calculations from database
  const distinctDocsCount = new Set(chunks.map((c) => c.sourceDoc)).size;
  const totalChunksCount = chunks.length;
  const avgChunkTokens = totalChunksCount > 0 
    ? Math.round(chunks.reduce((acc, c) => acc + (c.tokenCount || 0), 0) / totalChunksCount) 
    : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header with Stats & Actions */}
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
              Knowledge Base & Indexed Vectors
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
              <span>MongoDB & SQLite Live</span>
            </span>
          </div>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            Inspect vectorized passages, token chunks, semantic boundaries, and collection health loaded live from database.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '7px 14px',
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              fontSize: '13px',
              fontWeight: 500,
              color: 'var(--text-primary)',
              cursor: isRefreshing ? 'not-allowed' : 'pointer',
              boxShadow: 'var(--shadow-xs)',
              transition: 'background 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-app)')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#ffffff')}
          >
            <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
            <span>{isRefreshing ? 'Re-syncing...' : 'Refresh Knowledge Base'}</span>
          </button>
        </div>
      </div>

      {refreshSuccess && (
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
          <span>Knowledge index refreshed. Synchronized live with MongoDB Atlas cluster collection <code>knovera_docs</code>.</span>
        </div>
      )}

      {error && (
        <div
          style={{
            padding: '12px 16px',
            backgroundColor: 'var(--status-danger-bg)',
            border: '1px solid var(--status-danger-border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--status-danger)',
            fontSize: '13px',
          }}
        >
          {error}
        </div>
      )}

      {/* Overview Metric Pills - Derived Directly from Database */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '14px',
        }}
      >
        <div
          style={{
            padding: '14px 18px',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-xs)',
          }}
        >
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Indexed Documents</div>
          <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
            {distinctDocsCount} Documents
          </div>
        </div>

        <div
          style={{
            padding: '14px 18px',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-xs)',
          }}
        >
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Total Vector Chunks</div>
          <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
            {totalChunksCount} Chunks
          </div>
        </div>

        <div
          style={{
            padding: '14px 18px',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-xs)',
          }}
        >
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Embedding Vector Dim</div>
          <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--accent-primary)', marginTop: '2px' }}>
            384 Dim (Cosine)
          </div>
        </div>

        <div
          style={{
            padding: '14px 18px',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-xs)',
          }}
        >
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Average Chunk Size</div>
          <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
            {avgChunkTokens} Tokens
          </div>
        </div>
      </div>

      {/* Filter and Search Controls */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '12px',
          alignItems: 'center',
          backgroundColor: '#ffffff',
          padding: '12px 16px',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-light)',
          boxShadow: 'var(--shadow-xs)',
        }}
      >
        <div style={{ position: 'relative', flex: 1, minWidth: '240px' }}>
          <Search
            size={16}
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
            placeholder="Search chunk text, section, or keyword in database..."
            style={{
              width: '100%',
              padding: '7px 10px 7px 34px',
              fontSize: '13px',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-app)',
              outline: 'none',
              color: 'var(--text-primary)',
            }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>Source:</span>
          <select
            value={selectedFilterDoc}
            onChange={(e) => setSelectedFilterDoc(e.target.value)}
            style={{
              padding: '6px 12px',
              fontSize: '12.5px',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-app)',
              color: 'var(--text-primary)',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            <option value="all">All Documents ({availableDocs.length})</option>
            {availableDocs.map((doc) => (
              <option key={doc} value={doc}>
                {doc}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Chunks List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {loading ? (
          <div
            style={{
              padding: '48px',
              textAlign: 'center',
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-secondary)',
              fontSize: '13.5px',
            }}
          >
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
              <Loader2 size={18} className="animate-spin" color="var(--accent-primary)" />
              <span>Querying vector database records...</span>
            </div>
          </div>
        ) : chunks.length === 0 ? (
          <div
            style={{
              padding: '36px',
              textAlign: 'center',
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-muted)',
              fontSize: '13.5px',
            }}
          >
            No indexed chunks found in database matching your filter criteria.
          </div>
        ) : (
          chunks.map((chunk) => (
            <div
              key={chunk.id}
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-md)',
                padding: '16px 18px',
                boxShadow: 'var(--shadow-xs)',
                transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-medium)';
                e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-light)';
                e.currentTarget.style.boxShadow = 'var(--shadow-xs)';
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: '8px',
                  gap: '12px',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span
                      style={{
                        fontSize: '11px',
                        fontWeight: 600,
                        padding: '2px 7px',
                        borderRadius: 'var(--radius-xs)',
                        backgroundColor: '#eff6ff',
                        color: 'var(--accent-primary)',
                        border: '1px solid #bfdbfe',
                      }}
                    >
                      {chunk.id}
                    </span>
                    <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {chunk.section || 'General Passage'}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: '12px',
                      color: 'var(--text-muted)',
                      marginTop: '3px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      flexWrap: 'wrap',
                    }}
                  >
                    <FileText size={12} />
                    <span>{chunk.sourceDoc}</span>
                    <span>•</span>
                    <span>{chunk.tokenCount} tokens</span>
                    {chunk.indexedAt && (
                      <>
                        <span>•</span>
                        <span>Indexed {chunk.indexedAt}</span>
                      </>
                    )}
                  </div>
                </div>

                <button
                  onClick={() => setSelectedChunk(chunk)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px',
                    padding: '5px 10px',
                    borderRadius: 'var(--radius-xs)',
                    backgroundColor: 'var(--bg-muted)',
                    border: '1px solid var(--border-light)',
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    fontWeight: 500,
                    cursor: 'pointer',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-sidebar-active)')}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-muted)')}
                >
                  <Eye size={13} />
                  <span>Inspect</span>
                </button>
              </div>

              <div
                style={{
                  fontSize: '13.5px',
                  lineHeight: 1.55,
                  color: 'var(--text-secondary)',
                  backgroundColor: 'var(--bg-app)',
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                {chunk.content}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Inspect Chunk Modal */}
      {selectedChunk && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.4)',
            backdropFilter: 'blur(3px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
            padding: '16px',
          }}
          onClick={() => setSelectedChunk(null)}
        >
          <div
            className="animate-slide-down"
            style={{
              width: '100%',
              maxWidth: '620px',
              backgroundColor: '#ffffff',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border-light)',
              boxShadow: 'var(--shadow-lg)',
              overflow: 'hidden',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '16px 20px',
                borderBottom: '1px solid var(--border-light)',
              }}
            >
              <div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Vector Chunk Inspector
                </h3>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  ID: {selectedChunk.id} • Collection: knovera_docs
                </span>
              </div>
              <button
                onClick={() => setSelectedChunk(null)}
                style={{
                  padding: '4px',
                  borderRadius: 'var(--radius-xs)',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  border: 'none',
                  background: 'none',
                }}
              >
                <X size={18} />
              </button>
            </div>

            {/* Modal Body */}
            <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  RAW PASSAGE CONTENT
                </label>
                <div
                  style={{
                    marginTop: '6px',
                    padding: '12px',
                    backgroundColor: 'var(--bg-app)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-light)',
                    fontSize: '13.5px',
                    lineHeight: 1.55,
                    color: 'var(--text-primary)',
                    maxHeight: '200px',
                    overflowY: 'auto',
                  }}
                >
                  {selectedChunk.content}
                </div>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '12px',
                  fontSize: '12.5px',
                }}
              >
                <div
                  style={{
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-app)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <div style={{ color: 'var(--text-muted)' }}>Source Document</div>
                  <div style={{ fontWeight: 500, color: 'var(--text-primary)', marginTop: '2px' }}>
                    {selectedChunk.sourceDoc}
                  </div>
                </div>

                <div
                  style={{
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-app)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <div style={{ color: 'var(--text-muted)' }}>Embedding Dimension</div>
                  <div style={{ fontWeight: 500, color: 'var(--text-primary)', marginTop: '2px' }}>
                    384 Float32 Vectors ({selectedChunk.embeddingModel || 'BAAI/bge-small-en-v1.5'})
                  </div>
                </div>
              </div>

              {selectedChunk.metadata && Object.keys(selectedChunk.metadata).length > 0 && (
                <div>
                  <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    METADATA TAGS
                  </label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '6px' }}>
                    {Object.entries(selectedChunk.metadata).map(([key, val]) => (
                      <span
                        key={key}
                        style={{
                          fontSize: '11.5px',
                          padding: '3px 8px',
                          borderRadius: 'var(--radius-xs)',
                          backgroundColor: 'var(--bg-muted)',
                          border: '1px solid var(--border-light)',
                          color: 'var(--text-secondary)',
                        }}
                      >
                        <strong>{key}:</strong> {String(val)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div
              style={{
                padding: '12px 20px',
                backgroundColor: 'var(--bg-app)',
                borderTop: '1px solid var(--border-light)',
                display: 'flex',
                justifyContent: 'flex-end',
              }}
            >
              <button
                onClick={() => setSelectedChunk(null)}
                style={{
                  padding: '6px 14px',
                  backgroundColor: 'var(--accent-slate)',
                  color: '#ffffff',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '13px',
                  fontWeight: 500,
                  cursor: 'pointer',
                  border: 'none',
                }}
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
