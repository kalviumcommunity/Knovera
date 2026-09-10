'use client';

import React, { useState } from 'react';
import { api } from '@/lib/api';
import { QueryResponse } from '@/lib/types';
import CitationCard from './CitationCard';
import {
  Search,
  Sparkles,
  SlidersHorizontal,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Copy,
  Check,
  Send,
  HelpCircle,
} from 'lucide-react';

const SUGGESTED_QUERIES = [
  'What evidence is required for project submission?',
  'What is the standard customer refund window?',
  'What is the company policy for remote work equipment?',
  'Explain the onboarding verification protocol.',
];

export default function QueryView() {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(4);
  const [scoreThreshold, setScoreThreshold] = useState(0.45);
  const [useApi, setUseApi] = useState(true);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showOptions, setShowOptions] = useState(false);

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);

    try {
      const res = await api.submitQuery({
        question: question.trim(),
        k: topK,
        score_threshold: scoreThreshold,
        use_api: useApi,
      });
      setResponse(res);
    } catch (err: any) {
      setError(err.message || 'Failed to execute query');
    } finally {
      setLoading(false);
    }
  };

  const copyAnswer = () => {
    if (!response?.answer) return;
    navigator.clipboard.writeText(response.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Search Input Box */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '12px' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Search
              size={18}
              style={{
                position: 'absolute',
                left: '16px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--text-muted)',
              }}
            />
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question grounded against Knovera's knowledge base..."
              style={{
                width: '100%',
                padding: '14px 16px 14px 44px',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                fontSize: '15px',
                outline: 'none',
                transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
              }}
              onFocus={(e) => {
                e.target.style.borderColor = 'var(--accent-primary)';
                e.target.style.boxShadow = 'var(--shadow-glow)';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = 'var(--border-medium)';
                e.target.style.boxShadow = 'none';
              }}
            />
          </div>

          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="btn-primary"
            style={{ padding: '0 24px' }}
          >
            {loading ? (
              <>
                <Sparkles size={16} className="animate-spin" />
                <span>Synthesizing...</span>
              </>
            ) : (
              <>
                <Send size={16} />
                <span>Search</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={() => setShowOptions(!showOptions)}
            className="btn-secondary"
            title="Retrieval Options"
          >
            <SlidersHorizontal size={16} />
          </button>
        </form>

        {/* Retrieval Parameters Drawer */}
        {showOptions && (
          <div
            className="animate-fade-in"
            style={{
              padding: '16px',
              background: 'var(--bg-surface)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-subtle)',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '16px',
              fontSize: '13px',
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Top-K Chunks</span>
                <span style={{ fontWeight: 600, color: 'var(--text-accent)' }}>{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Min Similarity Threshold</span>
                <span style={{ fontWeight: 600, color: 'var(--text-accent)' }}>{scoreThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={scoreThreshold}
                onChange={(e) => setScoreThreshold(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingTop: '16px' }}>
              <input
                type="checkbox"
                id="useApiCheck"
                checked={useApi}
                onChange={(e) => setUseApi(e.target.checked)}
                style={{ width: '16px', height: '16px', accentColor: 'var(--accent-primary)' }}
              />
              <label htmlFor="useApiCheck" style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>
                Enable Live LLM Generation
              </label>
            </div>
          </div>
        )}

        {/* Suggested Queries */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Examples:</span>
          {SUGGESTED_QUERIES.map((q, idx) => (
            <button
              key={idx}
              onClick={() => {
                setQuestion(q);
              }}
              style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-secondary)',
                fontSize: '12px',
                padding: '4px 10px',
                borderRadius: 'var(--radius-full)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-accent)';
                e.currentTarget.style.color = 'var(--text-primary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-subtle)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Error Alert */}
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

      {/* Answer & Citations View */}
      {response && (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Main Answer Card */}
          <div className="glass-panel" style={{ padding: '28px', position: 'relative' }}>
            {/* Header / Latency stats */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '16px',
                borderBottom: '1px solid var(--border-subtle)',
                paddingBottom: '12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span
                  className={`badge ${
                    response.status === 'answered'
                      ? 'badge-emerald'
                      : response.status.includes('refused')
                      ? 'badge-amber'
                      : 'badge-indigo'
                  }`}
                >
                  {response.status === 'answered' ? (
                    <>
                      <CheckCircle2 size={13} />
                      <span>Grounded Answer</span>
                    </>
                  ) : (
                    <>
                      <AlertTriangle size={13} />
                      <span>{response.status.replace(/_/g, ' ')}</span>
                    </>
                  )}
                </span>

                {response.latency_ms && (
                  <span className="badge badge-cyan" style={{ fontSize: '11px' }}>
                    <Clock size={11} />
                    <span>{response.latency_ms} ms</span>
                  </span>
                )}
              </div>

              <button onClick={copyAnswer} className="btn-ghost" title="Copy answer">
                {copied ? <Check size={14} color="var(--accent-emerald)" /> : <Copy size={14} />}
                <span style={{ fontSize: '12px' }}>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            {/* Answer Content */}
            <div className="prose-answer">
              <p style={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{response.answer}</p>
            </div>

            {/* Stage Latencies Pill Bar */}
            {response.stage_latencies_ms && (
              <div
                style={{
                  display: 'flex',
                  gap: '8px',
                  flexWrap: 'wrap',
                  marginTop: '20px',
                  paddingTop: '16px',
                  borderTop: '1px solid var(--border-subtle)',
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                }}
              >
                <span>Pipeline Stage Latencies:</span>
                {Object.entries(response.stage_latencies_ms).map(([stage, ms]) => (
                  <span
                    key={stage}
                    style={{
                      background: 'var(--bg-surface)',
                      padding: '2px 8px',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    {stage.replace('_ms', '')}: <strong style={{ color: 'var(--text-primary)' }}>{ms}ms</strong>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Sources Section */}
          {response.sources && response.sources.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <h2 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Attributed Source Citations ({response.sources.length})
                </h2>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Click a source card to inspect chunk metadata
                </span>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                  gap: '14px',
                }}
              >
                {response.sources.map((src, i) => (
                  <CitationCard key={i} citation={src} index={i} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
