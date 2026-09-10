'use client';

import React, { useState } from 'react';
import { api } from '@/lib/api';
import { EvaluationResponse, EvaluationItem } from '@/lib/types';
import {
  BarChart3,
  Play,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Target,
  FileCheck,
  TrendingUp,
} from 'lucide-react';

const DEFAULT_BENCHMARK_CASES: EvaluationItem[] = [
  {
    query: 'What evidence is required for project submission?',
    expected_doc: 'equipment-policy.md',
    min_score: 0.4,
  },
  {
    query: 'What is the standard customer refund window?',
    expected_doc: 'customer_policy.txt',
    min_score: 0.4,
  },
  {
    query: 'What is the policy for home office hardware reimbursement?',
    expected_doc: 'remote-work-policy.md',
    min_score: 0.4,
  },
  {
    query: 'How should staff handle disputed customer refund requests?',
    expected_doc: 'customer_policy.txt',
    min_score: 0.4,
  },
];

export default function EvaluationView() {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runBenchmark = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runEvaluation({
        test_cases: DEFAULT_BENCHMARK_CASES,
      });
      setResults(res);
    } catch (err: any) {
      setError(err.message || 'Evaluation run failed');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Top action panel */}
      <div
        className="glass-panel"
        style={{
          padding: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 700 }}>RAG Benchmark & Retrieval Quality Evaluation</h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Run accuracy, precision@k, recall, and groundedness tests across Knovera test questions
          </p>
        </div>

        <button
          onClick={runBenchmark}
          disabled={running}
          className="btn-primary"
          style={{ padding: '12px 24px' }}
        >
          <Play size={16} className={running ? 'animate-spin' : ''} />
          <span>{running ? 'Evaluating Pipeline...' : 'Run Benchmark Suite'}</span>
        </button>
      </div>

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

      {/* KPI Cards */}
      {results && (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '16px',
            }}
          >
            {/* Precision */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '12px' }}>
                <Target size={14} color="var(--accent-primary)" />
                <span>Mean Precision@K</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--accent-primary)', marginTop: '8px' }}>
                {(results.summary.mean_precision_at_k * 100).toFixed(1)}%
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Top-K relevant chunk ratio</span>
            </div>

            {/* Recall */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '12px' }}>
                <TrendingUp size={14} color="var(--accent-cyan)" />
                <span>Mean Recall</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--accent-cyan)', marginTop: '8px' }}>
                {(results.summary.mean_recall * 100).toFixed(1)}%
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Gold document coverage</span>
            </div>

            {/* Groundedness */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '12px' }}>
                <FileCheck size={14} color="var(--accent-emerald)" />
                <span>Groundedness Score</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--accent-emerald)', marginTop: '8px' }}>
                {(results.summary.mean_groundedness * 100).toFixed(1)}%
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Factual context fidelity</span>
            </div>

            {/* Average Latency */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '12px' }}>
                <Clock size={14} color="var(--accent-amber)" />
                <span>Average Latency</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--accent-amber)', marginTop: '8px' }}>
                {results.summary.average_latency_ms.toFixed(0)} ms
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>End-to-end pipeline speed</span>
            </div>
          </div>

          {/* Test Case Details Table */}
          <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Test Case Evaluation Results</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)', textAlign: 'left', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '10px 14px' }}>Query</th>
                    <th style={{ padding: '10px 14px' }}>Status</th>
                    <th style={{ padding: '10px 14px' }}>Latency</th>
                    <th style={{ padding: '10px 14px' }}>Retrieved Sources</th>
                  </tr>
                </thead>
                <tbody>
                  {results.details.map((item, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 500, maxWidth: '300px' }}>
                        {item.query}
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        <span
                          className={`badge ${
                            item.status === 'answered' ? 'badge-emerald' : 'badge-amber'
                          }`}
                          style={{ fontSize: '11px' }}
                        >
                          {item.status}
                        </span>
                      </td>
                      <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>
                        {item.latency_ms} ms
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                          {item.retrieved_sources && item.retrieved_sources.length > 0 ? (
                            item.retrieved_sources.map((src, idx) => (
                              <span key={idx} className="badge badge-indigo" style={{ fontSize: '10px' }}>
                                {src}
                              </span>
                            ))
                          ) : (
                            <span style={{ color: 'var(--text-muted)' }}>None</span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
