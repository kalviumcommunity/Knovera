'use client';

import React, { useState } from 'react';
import { SourceCitation } from '@/lib/types';
import { FileText, Bookmark, ExternalLink, X, Award } from 'lucide-react';

interface CitationCardProps {
  citation: SourceCitation;
  index: number;
}

export default function CitationCard({ citation, index }: CitationCardProps) {
  const [modalOpen, setModalOpen] = useState(false);

  // Score percentage
  const scorePercent = citation.score !== undefined ? Math.round(citation.score * 100) : null;

  return (
    <>
      <div
        onClick={() => setModalOpen(true)}
        className="glass-panel"
        style={{
          padding: '14px 16px',
          cursor: 'pointer',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
          borderRadius: 'var(--radius-md)',
          position: 'relative',
          overflow: 'hidden',
          transition: 'all 0.2s ease',
        }}
      >
        {/* Top bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span
              style={{
                width: '22px',
                height: '22px',
                borderRadius: '50%',
                background: 'var(--accent-primary-subtle)',
                color: 'var(--accent-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '11px',
                fontWeight: 700,
                border: '1px solid rgba(99, 102, 241, 0.3)',
              }}
            >
              #{citation.rank || index + 1}
            </span>
            <span
              style={{
                fontSize: '13px',
                fontWeight: 600,
                color: 'var(--text-primary)',
                maxWidth: '180px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={citation.source}
            >
              {citation.doc_title || citation.source}
            </span>
          </div>

          {scorePercent !== null && (
            <span
              className="badge"
              style={{
                background: scorePercent >= 70 ? 'var(--accent-emerald-subtle)' : 'var(--accent-cyan-subtle)',
                color: scorePercent >= 70 ? '#6ee7b7' : '#67e8f9',
                fontSize: '11px',
                fontWeight: 600,
              }}
            >
              {scorePercent}% match
            </span>
          )}
        </div>

        {/* Section / Chunk Info */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
          <Bookmark size={12} />
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {citation.section ? `Section: ${citation.section}` : citation.chunk_id ? `Chunk: ${citation.chunk_id}` : 'Retrieved Evidence'}
          </span>
        </div>

        {/* Score progress bar */}
        {scorePercent !== null && (
          <div
            style={{
              height: '4px',
              width: '100%',
              background: 'rgba(255, 255, 255, 0.08)',
              borderRadius: 'var(--radius-full)',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${scorePercent}%`,
                background: 'linear-gradient(90deg, #6366f1 0%, #06b6d4 100%)',
                borderRadius: 'var(--radius-full)',
              }}
            />
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {modalOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px',
          }}
          onClick={() => setModalOpen(false)}
        >
          <div
            className="glass-panel-elevated"
            style={{
              width: '100%',
              maxWidth: '540px',
              padding: '24px',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <FileText size={20} color="var(--accent-primary)" />
                <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Source Attribution Details</h3>
              </div>
              <button
                onClick={() => setModalOpen(false)}
                className="btn-ghost"
                style={{ padding: '4px' }}
              >
                <X size={18} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Document Source</span>
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{citation.source}</span>
              </div>

              {citation.doc_title && (
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Document Title</span>
                  <span style={{ fontWeight: 500 }}>{citation.doc_title}</span>
                </div>
              )}

              {citation.section && (
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Section Heading</span>
                  <span style={{ fontWeight: 500, color: 'var(--text-accent)' }}>{citation.section}</span>
                </div>
              )}

              {citation.chunk_id && (
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Chunk ID</span>
                  <code style={{ background: 'var(--bg-surface)', padding: '2px 6px', borderRadius: '4px', fontSize: '12px' }}>
                    {citation.chunk_id}
                  </code>
                </div>
              )}

              {citation.score !== undefined && (
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Cosine Similarity Score</span>
                  <span style={{ fontWeight: 600, color: 'var(--accent-emerald)' }}>
                    {citation.score.toFixed(4)} ({scorePercent}%)
                  </span>
                </div>
              )}

              {citation.rank !== undefined && (
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Retrieval Rank</span>
                  <span style={{ fontWeight: 600 }}>Rank #{citation.rank}</span>
                </div>
              )}
            </div>

            <button
              onClick={() => setModalOpen(false)}
              className="btn-secondary"
              style={{ alignSelf: 'flex-end', marginTop: '8px' }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}
