'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { AuditLog } from '@/lib/types';
import { api } from '@/lib/api';
import {
  Search,
  Download,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ChevronLeft,
  ChevronRight,
  Eye,
  X,
  RefreshCw,
  Loader2,
  Database,
  MessageSquare,
  Bot,
  FileText,
  Sparkles,
  Shield,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

export default function LogsView() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'success' | 'warning' | 'error'>('all');
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null);
  const [showRawJson, setShowRawJson] = useState(false);
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pageSize = 6;

  const fetchLogs = useCallback(async (page: number, search: string, status: string, isSilent = false) => {
    if (!isSilent) setLoading(true);
    setError(null);
    try {
      const res = await api.getLogs({
        page,
        pageSize,
        search: search.trim() || undefined,
        status: status !== 'all' ? status : undefined,
      });
      setLogs(res.logs || []);
      setTotalCount(res.total || 0);
      setTotalPages(Math.max(1, res.total_pages || Math.ceil((res.total || 0) / pageSize)));
    } catch (err: any) {
      console.error('Failed to load audit logs from database:', err);
      setError(err?.message || 'Failed to load audit logs from database.');
      setLogs([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchLogs(currentPage, searchQuery, statusFilter);
  }, [fetchLogs, currentPage, statusFilter]);

  // Debounced search trigger
  useEffect(() => {
    const handler = setTimeout(() => {
      setCurrentPage(1);
      fetchLogs(1, searchQuery, statusFilter);
    }, 300);
    return () => clearTimeout(handler);
  }, [searchQuery]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchLogs(currentPage, searchQuery, statusFilter, true);
  };

  const handleExportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(logs, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `knovera_audit_logs_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const getStatusBadge = (status: 'success' | 'warning' | 'error' | string) => {
    switch (status) {
      case 'success':
        return {
          icon: <CheckCircle2 size={12} />,
          label: 'Success',
          color: 'var(--status-success)',
          bg: 'var(--status-success-bg)',
          border: 'var(--status-success-border)',
        };
      case 'warning':
        return {
          icon: <AlertTriangle size={12} />,
          label: 'Warning',
          color: 'var(--status-warning)',
          bg: 'var(--status-warning-bg)',
          border: 'var(--status-warning-border)',
        };
      case 'error':
      default:
        return {
          icon: <XCircle size={12} />,
          label: 'Blocked / Error',
          color: 'var(--status-danger)',
          bg: 'var(--status-danger-bg)',
          border: 'var(--status-danger-border)',
        };
    }
  };

  const copyToClipboard = (text: string, field: string) => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopiedField(field);
      setTimeout(() => setCopiedField(null), 1800);
    }
  };

  const getLogInput = (log: AuditLog): string => {
    if (log.input && log.input.trim()) return log.input;
    if (log.querySnippet && log.querySnippet.trim()) return log.querySnippet;
    if (log.action && log.action.includes('Query:')) {
      return log.action.replace(/^.*Query:\s*/i, '');
    }
    return log.action || 'No user input recorded';
  };

  const getLogOutput = (log: AuditLog): string => {
    if (log.output && log.output.trim() && !log.output.startsWith('Guardrail:')) {
      return log.output;
    }
    if (log.guardrailStatus === 'triggered' || log.status === 'error' || log.guardrailStatus === 'refused') {
      return log.details || 'Request intercepted and blocked by Knovera Security Guardrails.';
    }
    if (log.details && log.details.trim() && !log.details.startsWith('Guardrail:')) {
      return log.details;
    }
    return 'No text response recorded (Query processed with citations).';
  };

  const getNormalizedSources = (log: AuditLog) => {
    if (Array.isArray(log.sources) && log.sources.length > 0) {
      return log.sources.map((s: any, idx: number) => {
        if (typeof s === 'string') {
          return {
            id: `source_${idx + 1}`,
            title: s,
            filename: s,
            section: undefined,
            score: log.groundednessScore,
          };
        }
        return {
          id: s.chunk_id || `source_${idx + 1}`,
          title: s.doc_title || s.source || `Document ${idx + 1}`,
          filename: s.source || s.doc_title || 'knowledge_source.pdf',
          section: s.section,
          score: s.score,
        };
      });
    }

    // Fallback detection from details text if legacy log
    if (log.details) {
      const d = log.details;
      if (d.includes('customer_sla_refund_terms')) {
        return [{
          id: 'chunk_sla_01',
          title: 'Customer SLA & Refund Terms',
          filename: 'customer_sla_refund_terms.docx',
          section: 'Downtime Credit Calculations',
          score: log.groundednessScore || 0.94,
        }];
      }
      if (d.includes('rag_api_specification_v2')) {
        return [{
          id: 'chunk_api_01',
          title: 'RAG API Specification v2',
          filename: 'rag_api_specification_v2.md',
          section: 'Bearer Token Authentication',
          score: log.groundednessScore || 0.98,
        }];
      }
      if (d.includes('enterprise_security_compliance')) {
        return [{
          id: 'chunk_sec_01',
          title: 'Enterprise Security Compliance',
          filename: 'enterprise_security_compliance_2026.pdf',
          section: 'Access Isolation & Audit Trail Standards',
          score: log.groundednessScore || 1.0,
        }];
      }
    }

    return [];
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
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
              System & Chatbot Activity Logs
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
              <span>SQLite DB Live</span>
            </span>
          </div>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            Comprehensive audit trail of user sessions, retrieval queries, and safety interventions loaded directly from database.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '7px 12px',
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              fontSize: '13px',
              fontWeight: 500,
              color: 'var(--text-primary)',
              cursor: refreshing ? 'not-allowed' : 'pointer',
              boxShadow: 'var(--shadow-xs)',
            }}
          >
            <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
            <span>{refreshing ? 'Syncing...' : 'Refresh'}</span>
          </button>

          <button
            onClick={handleExportJSON}
            disabled={logs.length === 0}
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
              cursor: logs.length === 0 ? 'not-allowed' : 'pointer',
              boxShadow: 'var(--shadow-xs)',
              opacity: logs.length === 0 ? 0.6 : 1,
            }}
          >
            <Download size={14} />
            <span>Export Audit Log</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
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
            placeholder="Search database by user, session ID, action, or query..."
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
          <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>Status:</span>
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value as any);
              setCurrentPage(1);
            }}
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
            <option value="all">All Events</option>
            <option value="success">Success Only</option>
            <option value="warning">Warnings Only</option>
            <option value="error">Errors & Blocked Only</option>
          </select>
        </div>
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
          <XCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Logs Table */}
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
                <th style={{ padding: '10px 16px', fontWeight: 600, color: 'var(--text-secondary)' }}>Timestamp</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>User / Session</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>Action / Query</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>Latency</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>Guardrail</th>
                <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>Status</th>
                <th style={{ padding: '10px 16px', fontWeight: 600, color: 'var(--text-secondary)', textAlign: 'right' }}>
                  Details
                </th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} style={{ padding: '48px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
                      <Loader2 size={18} className="animate-spin" color="var(--accent-primary)" />
                      <span>Loading audit logs from database...</span>
                    </div>
                  </td>
                </tr>
              ) : logs.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No audit log records found in database matching your filter criteria.
                  </td>
                </tr>
              ) : (
                logs.map((log) => {
                  const badge = getStatusBadge(log.status);

                  return (
                    <tr
                      key={log.id}
                      style={{
                        borderBottom: '1px solid var(--border-subtle)',
                        transition: 'background 0.12s ease',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-app)')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#ffffff')}
                    >
                      <td style={{ padding: '12px 16px', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                        {log.timestamp}
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        <div style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{log.user}</div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{log.sessionId}</div>
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        <div style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{log.action}</div>
                        {log.querySnippet && (
                          <div
                            style={{
                              fontSize: '11.5px',
                              color: 'var(--text-muted)',
                              maxWidth: '280px',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              marginTop: '2px',
                            }}
                          >
                            &ldquo;{log.querySnippet}&rdquo;
                          </div>
                        )}
                      </td>
                      <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>
                        {log.latencyMs != null ? `${log.latencyMs} ms` : '—'}
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        {log.guardrailStatus === 'triggered' ? (
                          <span
                            style={{
                              fontSize: '11px',
                              fontWeight: 600,
                              color: 'var(--status-warning)',
                              backgroundColor: 'var(--status-warning-bg)',
                              border: '1px solid var(--status-warning-border)',
                              padding: '2px 7px',
                              borderRadius: 'var(--radius-xs)',
                            }}
                          >
                            Triggered
                          </span>
                        ) : log.guardrailStatus === 'passed' ? (
                          <span
                            style={{
                              fontSize: '11px',
                              fontWeight: 600,
                              color: 'var(--status-success)',
                              backgroundColor: 'var(--status-success-bg)',
                              padding: '2px 7px',
                              borderRadius: 'var(--radius-xs)',
                            }}
                          >
                            Passed
                          </span>
                        ) : (
                          <span style={{ fontSize: '11.5px', color: 'var(--text-light)' }}>N/A</span>
                        )}
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        <span
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '11.5px',
                            fontWeight: 600,
                            color: badge.color,
                            backgroundColor: badge.bg,
                            border: `1px solid ${badge.border}`,
                            padding: '2px 8px',
                            borderRadius: 'var(--radius-full)',
                          }}
                        >
                          {badge.icon}
                          <span>{badge.label}</span>
                        </span>
                      </td>
                      <td style={{ padding: '12px 16px', textAlign: 'right' }}>
                        <button
                          onClick={() => setSelectedLog(log)}
                          title="View JSON details"
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            padding: '4px 8px',
                            borderRadius: 'var(--radius-xs)',
                            backgroundColor: 'var(--bg-muted)',
                            border: '1px solid var(--border-light)',
                            fontSize: '12px',
                            color: 'var(--text-secondary)',
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
                          <Eye size={12} />
                          <span>View</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '12px 18px',
            borderTop: '1px solid var(--border-light)',
            fontSize: '12.5px',
            color: 'var(--text-secondary)',
          }}
        >
          <div>
            Showing {totalCount === 0 ? 0 : (currentPage - 1) * pageSize + 1} to{' '}
            {Math.min(currentPage * pageSize, totalCount)} of {totalCount} records
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1 || loading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                padding: '4px 10px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-light)',
                backgroundColor: currentPage === 1 ? 'var(--bg-muted)' : '#ffffff',
                color: currentPage === 1 ? 'var(--text-light)' : 'var(--text-primary)',
                cursor: currentPage === 1 || loading ? 'not-allowed' : 'pointer',
              }}
            >
              <ChevronLeft size={14} />
              <span>Previous</span>
            </button>

            <span style={{ padding: '0 8px', fontWeight: 500 }}>
              Page {currentPage} of {totalPages}
            </span>

            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages || loading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                padding: '4px 10px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-light)',
                backgroundColor: currentPage === totalPages ? 'var(--bg-muted)' : '#ffffff',
                color: currentPage === totalPages ? 'var(--text-light)' : 'var(--text-primary)',
                cursor: currentPage === totalPages || loading ? 'not-allowed' : 'pointer',
              }}
            >
              <span>Next</span>
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Log Details Modal Showing Input, Output & Source */}
      {selectedLog && (() => {
        const input = getLogInput(selectedLog);
        const output = getLogOutput(selectedLog);
        const sources = getNormalizedSources(selectedLog);
        const badge = getStatusBadge(selectedLog.status);
        const isBlocked = selectedLog.status === 'error' || selectedLog.guardrailStatus === 'refused' || selectedLog.guardrailStatus === 'triggered';

        return (
          <div
            style={{
              position: 'fixed',
              inset: 0,
              backgroundColor: 'rgba(15, 23, 42, 0.45)',
              backdropFilter: 'blur(4px)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 9999,
              padding: '16px',
            }}
            onClick={() => {
              setSelectedLog(null);
              setShowRawJson(false);
            }}
          >
            <div
              className="animate-slide-down"
              style={{
                width: '100%',
                maxWidth: '720px',
                maxHeight: '88vh',
                display: 'flex',
                flexDirection: 'column',
                backgroundColor: '#ffffff',
                borderRadius: 'var(--radius-xl)',
                border: '1px solid var(--border-light)',
                boxShadow: 'var(--shadow-xl)',
                overflow: 'hidden',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div
                style={{
                  padding: '16px 22px',
                  borderBottom: '1px solid var(--border-light)',
                  backgroundColor: '#ffffff',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '12px',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <h3 style={{ fontSize: '17px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                      Audit Event Inspector
                    </h3>
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        fontSize: '11px',
                        fontWeight: 600,
                        color: badge.color,
                        backgroundColor: badge.bg,
                        border: `1px solid ${badge.border}`,
                        padding: '1px 8px',
                        borderRadius: 'var(--radius-full)',
                      }}
                    >
                      {badge.icon}
                      <span>{badge.label}</span>
                    </span>
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    <span>{selectedLog.timestamp}</span>
                    <span style={{ margin: '0 6px' }}>•</span>
                    <span>User: <strong>{selectedLog.user}</strong></span>
                    <span style={{ margin: '0 6px' }}>•</span>
                    <span>ID: <code style={{ fontSize: '11px' }}>{selectedLog.id}</code></span>
                  </div>
                </div>

                <button
                  onClick={() => {
                    setSelectedLog(null);
                    setShowRawJson(false);
                  }}
                  style={{
                    padding: '6px',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    background: 'none',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 0.15s ease',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-muted)')}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                >
                  <X size={19} />
                </button>
              </div>

              {/* Modal Body */}
              <div
                style={{
                  padding: '20px 22px',
                  overflowY: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '18px',
                }}
              >
                {/* 1. INPUT Section */}
                <div>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '6px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <MessageSquare size={14} color="var(--accent-primary)" />
                      <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', letterSpacing: '0.03em' }}>
                        USER INPUT / PROMPT
                      </span>
                    </div>

                    <button
                      onClick={() => copyToClipboard(input, 'input')}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        fontSize: '11.5px',
                        color: 'var(--text-muted)',
                        backgroundColor: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '2px 6px',
                        borderRadius: 'var(--radius-xs)',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-primary)')}
                      onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
                    >
                      {copiedField === 'input' ? <Check size={12} color="var(--status-success)" /> : <Copy size={12} />}
                      <span>{copiedField === 'input' ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>

                  <div
                    style={{
                      padding: '12px 14px',
                      backgroundColor: '#f8fafc',
                      borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-light)',
                      fontSize: '13.5px',
                      lineHeight: 1.55,
                      color: 'var(--text-primary)',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                    }}
                  >
                    {input}
                  </div>
                </div>

                {/* 2. OUTPUT Section */}
                <div>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '6px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Bot size={14} color={isBlocked ? 'var(--status-danger)' : '#10b981'} />
                      <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', letterSpacing: '0.03em' }}>
                        {isBlocked ? 'GUARDRAIL / SECURITY RESPONSE' : 'SYSTEM OUTPUT / ANSWER'}
                      </span>
                    </div>

                    <button
                      onClick={() => copyToClipboard(output, 'output')}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        fontSize: '11.5px',
                        color: 'var(--text-muted)',
                        backgroundColor: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '2px 6px',
                        borderRadius: 'var(--radius-xs)',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-primary)')}
                      onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
                    >
                      {copiedField === 'output' ? <Check size={12} color="var(--status-success)" /> : <Copy size={12} />}
                      <span>{copiedField === 'output' ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>

                  <div
                    style={{
                      padding: '14px 16px',
                      backgroundColor: isBlocked ? 'var(--status-danger-bg)' : '#ffffff',
                      borderRadius: 'var(--radius-md)',
                      border: isBlocked ? '1px solid var(--status-danger-border)' : '1px solid var(--border-light)',
                      fontSize: '13.5px',
                      lineHeight: 1.6,
                      color: isBlocked ? 'var(--status-danger)' : 'var(--text-primary)',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      boxShadow: isBlocked ? 'none' : 'var(--shadow-xs)',
                    }}
                  >
                    {output}
                  </div>
                </div>

                {/* 3. SOURCE & CITATIONS Section */}
                <div>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '8px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <FileText size={14} color="#6366f1" />
                      <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', letterSpacing: '0.03em' }}>
                        RETRIEVED SOURCES & CITATIONS
                      </span>
                    </div>

                    <span
                      style={{
                        fontSize: '11px',
                        fontWeight: 600,
                        padding: '1px 7px',
                        borderRadius: 'var(--radius-full)',
                        backgroundColor: sources.length > 0 ? '#eff6ff' : 'var(--bg-muted)',
                        color: sources.length > 0 ? 'var(--accent-primary)' : 'var(--text-muted)',
                        border: sources.length > 0 ? '1px solid #bfdbfe' : '1px solid var(--border-light)',
                      }}
                    >
                      {sources.length} {sources.length === 1 ? 'Source Cited' : 'Sources Cited'}
                    </span>
                  </div>

                  {sources.length === 0 ? (
                    <div
                      style={{
                        padding: '14px 16px',
                        backgroundColor: '#f8fafc',
                        borderRadius: 'var(--radius-md)',
                        border: '1px dashed var(--border-light)',
                        fontSize: '12.5px',
                        color: 'var(--text-muted)',
                        textAlign: 'center',
                      }}
                    >
                      No internal document sources cited for this event (intercepted by guardrails, direct administration, or general query).
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {sources.map((s, idx) => (
                        <div
                          key={s.id || idx}
                          style={{
                            padding: '10px 14px',
                            backgroundColor: '#ffffff',
                            borderRadius: 'var(--radius-md)',
                            border: '1px solid var(--border-light)',
                            boxShadow: 'var(--shadow-xs)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: '12px',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0, flex: 1 }}>
                            <div
                              style={{
                                width: '26px',
                                height: '26px',
                                borderRadius: 'var(--radius-xs)',
                                backgroundColor: '#eff6ff',
                                color: 'var(--accent-primary)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '11px',
                                fontWeight: 700,
                                flexShrink: 0,
                              }}
                            >
                              [{idx + 1}]
                            </div>

                            <div style={{ minWidth: 0, flex: 1 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <span
                                  style={{
                                    fontSize: '13px',
                                    fontWeight: 600,
                                    color: 'var(--text-primary)',
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    whiteSpace: 'nowrap',
                                  }}
                                >
                                  {s.title}
                                </span>
                              </div>
                              <div
                                style={{
                                  fontSize: '11.5px',
                                  color: 'var(--text-muted)',
                                  marginTop: '1px',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '8px',
                                }}
                              >
                                {s.filename && <span>File: {s.filename}</span>}
                                {s.section && (
                                  <>
                                    <span>•</span>
                                    <span>{s.section}</span>
                                  </>
                                )}
                              </div>
                            </div>
                          </div>

                          {s.score != null && (
                            <div
                              style={{
                                fontSize: '11px',
                                fontWeight: 600,
                                padding: '2px 7px',
                                borderRadius: 'var(--radius-xs)',
                                backgroundColor: '#f0fdf4',
                                color: '#16a34a',
                                border: '1px solid #bbf7d0',
                                flexShrink: 0,
                              }}
                            >
                              {Math.round(s.score > 1 ? s.score : s.score * 100)}% Match
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* 4. Telemetry Pill Bar */}
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '8px',
                    padding: '10px 14px',
                    backgroundColor: 'var(--bg-app)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-light)',
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                  }}
                >
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Latency: </span>
                    <strong>{selectedLog.latencyMs != null ? `${selectedLog.latencyMs} ms` : '—'}</strong>
                  </div>
                  <span style={{ color: 'var(--border-medium)' }}>|</span>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Groundedness: </span>
                    <strong>{selectedLog.groundednessScore != null ? selectedLog.groundednessScore : '—'}</strong>
                  </div>
                  <span style={{ color: 'var(--border-medium)' }}>|</span>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Guardrail: </span>
                    <strong>{selectedLog.guardrailName || 'Standard'} ({selectedLog.guardrailStatus || 'passed'})</strong>
                  </div>
                  <span style={{ color: 'var(--border-medium)' }}>|</span>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Session: </span>
                    <code style={{ fontSize: '11px' }}>{selectedLog.sessionId}</code>
                  </div>
                </div>

                {/* 5. Collapsible Raw Database Record Accordion */}
                <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '12px' }}>
                  <button
                    onClick={() => setShowRawJson(!showRawJson)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      fontSize: '12px',
                      color: 'var(--text-secondary)',
                      fontWeight: 600,
                      padding: '4px 0',
                    }}
                  >
                    <span>Raw Database Record (JSON)</span>
                    {showRawJson ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>

                  {showRawJson && (
                    <div style={{ marginTop: '8px' }}>
                      <pre
                        style={{
                          margin: 0,
                          padding: '12px',
                          backgroundColor: '#0f172a',
                          color: '#e2e8f0',
                          borderRadius: 'var(--radius-md)',
                          fontSize: '11.5px',
                          fontFamily: 'var(--font-mono)',
                          maxHeight: '180px',
                          overflowY: 'auto',
                          lineHeight: 1.45,
                        }}
                      >
                        {JSON.stringify(selectedLog, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>

              {/* Modal Footer */}
              <div
                style={{
                  padding: '12px 22px',
                  backgroundColor: 'var(--bg-app)',
                  borderTop: '1px solid var(--border-light)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                  Knovera Multi-Tenant Audit Security
                </div>

                <button
                  onClick={() => {
                    setSelectedLog(null);
                    setShowRawJson(false);
                  }}
                  style={{
                    padding: '7px 18px',
                    backgroundColor: 'var(--accent-slate)',
                    color: '#ffffff',
                    borderRadius: 'var(--radius-md)',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    border: 'none',
                    boxShadow: 'var(--shadow-xs)',
                  }}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
