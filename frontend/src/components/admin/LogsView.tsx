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
} from 'lucide-react';

export default function LogsView() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'success' | 'warning' | 'error'>('all');
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null);
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

      {/* Log Details Modal */}
      {selectedLog && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15, 23, 0.4)',
            backdropFilter: 'blur(3px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
            padding: '16px',
          }}
          onClick={() => setSelectedLog(null)}
        >
          <div
            className="animate-slide-down"
            style={{
              width: '100%',
              maxWidth: '600px',
              backgroundColor: '#ffffff',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border-light)',
              boxShadow: 'var(--shadow-lg)',
              overflow: 'hidden',
            }}
            onClick={(e) => e.stopPropagation()}
          >
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
                  Audit Event Details
                </h3>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Event ID: {selectedLog.id} • {selectedLog.timestamp}
                </span>
              </div>
              <button
                onClick={() => setSelectedLog(null)}
                style={{ padding: '4px', color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none' }}
              >
                <X size={18} />
              </button>
            </div>

            <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  EVENT SUMMARY
                </label>
                <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginTop: '4px' }}>
                  {selectedLog.action}
                </div>
              </div>

              <div>
                <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  DETAILED AUDIT MESSAGE
                </label>
                <div
                  style={{
                    marginTop: '4px',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-app)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-light)',
                    fontSize: '13px',
                    lineHeight: 1.5,
                    color: 'var(--text-secondary)',
                  }}
                >
                  {selectedLog.details}
                </div>
              </div>

              <div>
                <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  RAW DATABASE PAYLOAD
                </label>
                <pre
                  style={{
                    marginTop: '4px',
                    padding: '12px',
                    backgroundColor: 'var(--bg-app)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-light)',
                    fontSize: '12px',
                    fontFamily: 'var(--font-mono)',
                    color: '#1e293b',
                    maxHeight: '180px',
                    overflowY: 'auto',
                  }}
                >
                  {JSON.stringify(selectedLog, null, 2)}
                </pre>
              </div>
            </div>

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
                onClick={() => setSelectedLog(null)}
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
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
