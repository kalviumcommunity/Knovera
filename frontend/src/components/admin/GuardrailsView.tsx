'use client';

import React, { useState, useEffect } from 'react';
import { Guardrail, GuardrailCategory } from '@/lib/types';
import { api } from '@/lib/api';
import {
  ShieldCheck,
  Plus,
  ArrowUp,
  ArrowDown,
  Pencil,
  Trash2,
  AlertTriangle,
  Lock,
  Sparkles,
  Eye,
  CheckCircle2,
  X,
  Sliders,
  ShieldAlert,
  RefreshCw,
} from 'lucide-react';

export default function GuardrailsView() {
  const [guardrails, setGuardrails] = useState<Guardrail[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [modalOpen, setModalOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);

  // Form states for Add / Edit
  const [formName, setFormName] = useState('');
  const [formCategory, setFormCategory] = useState<GuardrailCategory>('custom');
  const [formDescription, setFormDescription] = useState('');
  const [formRule, setFormRule] = useState('');
  const [formTrigger, setFormTrigger] = useState('Cosine similarity < 0.70');
  const [formAction, setFormAction] = useState('Refuse query with polite disclaimer');
  const [formEnabled, setFormEnabled] = useState(true);

  // Load from database on mount
  const fetchGuardrails = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const data = await api.getGuardrails();
      setGuardrails(data);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to fetch guardrails from database.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchGuardrails();
  }, []);

  const handleToggle = async (id: string) => {
    const target = guardrails.find((g) => g.id === id);
    if (!target) return;

    const updatedStatus = !target.enabled;
    // Optimistic UI update
    setGuardrails((prev) =>
      prev.map((g) => (g.id === id ? { ...g, enabled: updatedStatus } : g))
    );

    try {
      await api.updateGuardrail(id, { ...target, enabled: updatedStatus });
    } catch (err: any) {
      // Rollback on error
      setGuardrails((prev) =>
        prev.map((g) => (g.id === id ? { ...g, enabled: target.enabled } : g))
      );
      alert(`Database error: ${err.message}`);
    }
  };

  const handleMoveUp = async (index: number) => {
    if (index === 0) return;
    const copy = [...guardrails];
    const temp = copy[index - 1];
    copy[index - 1] = copy[index];
    copy[index] = temp;
    const reordered = copy.map((item, idx) => ({ ...item, priority: idx + 1 }));
    setGuardrails(reordered);

    try {
      await api.reorderGuardrails(reordered.map((g) => g.id));
    } catch (err: any) {
      fetchGuardrails();
    }
  };

  const handleMoveDown = async (index: number) => {
    if (index === guardrails.length - 1) return;
    const copy = [...guardrails];
    const temp = copy[index + 1];
    copy[index + 1] = copy[index];
    copy[index] = temp;
    const reordered = copy.map((item, idx) => ({ ...item, priority: idx + 1 }));
    setGuardrails(reordered);

    try {
      await api.reorderGuardrails(reordered.map((g) => g.id));
    } catch (err: any) {
      fetchGuardrails();
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this guardrail from the database?')) return;
    try {
      await api.deleteGuardrail(id);
      fetchGuardrails();
    } catch (err: any) {
      alert(`Failed to delete guardrail: ${err.message}`);
    }
  };

  const handleOpenAdd = () => {
    setEditingId(null);
    setFormName('');
    setFormCategory('custom');
    setFormDescription('');
    setFormRule('');
    setFormTrigger('Cosine similarity < 0.70');
    setFormAction('Refuse query with polite disclaimer');
    setFormEnabled(true);
    setModalOpen(true);
  };

  const handleOpenEdit = (g: Guardrail) => {
    setEditingId(g.id);
    setFormName(g.name);
    setFormCategory(g.category);
    setFormDescription(g.description || '');
    setFormRule(g.rule);
    setFormTrigger(g.triggerCondition || '');
    setFormAction(g.action);
    setFormEnabled(g.enabled);
    setModalOpen(true);
  };

  const handleSaveModal = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formName.trim() || !formRule.trim()) return;

    try {
      if (editingId) {
        await api.updateGuardrail(editingId, {
          name: formName.trim(),
          category: formCategory,
          description: formDescription.trim(),
          rule: formRule.trim(),
          triggerCondition: formTrigger,
          action: formAction,
          enabled: formEnabled,
        });
      } else {
        await api.createGuardrail({
          name: formName.trim(),
          category: formCategory,
          description: formDescription.trim(),
          rule: formRule.trim(),
          triggerCondition: formTrigger,
          action: formAction,
          enabled: formEnabled,
          priority: guardrails.length + 1,
        });
      }
      setModalOpen(false);
      fetchGuardrails();
    } catch (err: any) {
      alert(`Database error: ${err.message}`);
    }
  };

  const getCategoryBadge = (cat: GuardrailCategory) => {
    switch (cat) {
      case 'injection':
        return { label: 'Injection Defense', color: '#dc2626', bg: '#fef2f2', border: '#fecaca' };
      case 'hallucination':
        return { label: 'Hallucination Shield', color: '#0284c7', bg: '#f0f9ff', border: '#bae6fd' };
      case 'pii':
        return { label: 'PII Masking', color: '#d97706', bg: '#fffbeb', border: '#fde68a' };
      case 'policy':
        return { label: 'Brand & Policy', color: '#7c3aed', bg: '#f5f3ff', border: '#ddd6fe' };
      default:
        return { label: 'Custom Rule', color: 'var(--text-secondary)', bg: 'var(--bg-muted)', border: 'var(--border-light)' };
    }
  };

  const activeCount = guardrails.filter((g) => g.enabled).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header with Add Button */}
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
            AI Safety & Response Guardrails
          </h1>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            Live safety interventions stored in database table <code>guardrails</code> (SQLite).
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={fetchGuardrails}
            title="Refresh from database"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 12px',
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              fontSize: '13px',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
            }}
          >
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
            <span>Sync DB</span>
          </button>

          <button
            onClick={handleOpenAdd}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 16px',
              backgroundColor: 'var(--accent-primary)',
              color: '#ffffff',
              borderRadius: 'var(--radius-md)',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <Plus size={16} />
            <span>Add Guardrail</span>
          </button>
        </div>
      </div>

      {errorMessage && (
        <div
          style={{
            padding: '10px 14px',
            backgroundColor: 'var(--status-danger-bg)',
            border: '1px solid var(--status-danger-border)',
            borderRadius: 'var(--radius-md)',
            fontSize: '13px',
            color: 'var(--status-danger)',
          }}
        >
          {errorMessage}
        </div>
      )}

      {/* Overview Stat Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
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
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Configured in DB</div>
          <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
            {guardrails.length} Rules
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
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Active in Pipeline</div>
          <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--status-success)', marginTop: '2px' }}>
            {activeCount} Active
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
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Storage Backend</div>
          <div style={{ fontSize: '14.5px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '6px' }}>
            SQLite <code>knovera.db</code>
          </div>
        </div>
      </div>

      {/* Loading state */}
      {isLoading && guardrails.length === 0 ? (
        <div
          style={{
            padding: '48px',
            textAlign: 'center',
            color: 'var(--text-muted)',
            backgroundColor: '#ffffff',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-light)',
          }}
        >
          Loading guardrails directly from database...
        </div>
      ) : guardrails.length === 0 ? (
        <div
          style={{
            padding: '48px',
            textAlign: 'center',
            color: 'var(--text-muted)',
            backgroundColor: '#ffffff',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-light)',
          }}
        >
          No guardrails found in the database. Click "+ Add Guardrail" above to create one.
        </div>
      ) : (
        /* Prioritized Guardrails List */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {guardrails.map((g, index) => {
            const badge = getCategoryBadge(g.category);

            return (
              <div
                key={g.id}
                style={{
                  backgroundColor: '#ffffff',
                  border: '1px solid var(--border-light)',
                  borderRadius: 'var(--radius-lg)',
                  padding: '18px 20px',
                  boxShadow: 'var(--shadow-xs)',
                  opacity: g.enabled ? 1 : 0.65,
                  transition: 'all 0.15s ease',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    gap: '16px',
                  }}
                >
                  <div style={{ display: 'flex', gap: '14px', flex: 1 }}>
                    {/* Priority Reordering Controls */}
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor: 'var(--bg-app)',
                        border: '1px solid var(--border-light)',
                        borderRadius: 'var(--radius-md)',
                        padding: '4px',
                        minWidth: '38px',
                      }}
                    >
                      <button
                        onClick={() => handleMoveUp(index)}
                        disabled={index === 0}
                        title="Move up in priority"
                        style={{
                          padding: '2px',
                          color: index === 0 ? 'var(--border-medium)' : 'var(--text-secondary)',
                          cursor: index === 0 ? 'not-allowed' : 'pointer',
                        }}
                      >
                        <ArrowUp size={14} />
                      </button>
                      <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
                        #{g.priority}
                      </span>
                      <button
                        onClick={() => handleMoveDown(index)}
                        disabled={index === guardrails.length - 1}
                        title="Move down in priority"
                        style={{
                          padding: '2px',
                          color:
                            index === guardrails.length - 1
                              ? 'var(--border-medium)'
                              : 'var(--text-secondary)',
                          cursor: index === guardrails.length - 1 ? 'not-allowed' : 'pointer',
                        }}
                      >
                        <ArrowDown size={14} />
                      </button>
                    </div>

                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        <span
                          style={{
                            fontSize: '11px',
                            fontWeight: 600,
                            padding: '2px 8px',
                            borderRadius: 'var(--radius-xs)',
                            backgroundColor: badge.bg,
                            color: badge.color,
                            border: `1px solid ${badge.border}`,
                          }}
                        >
                          {badge.label}
                        </span>
                        <h3 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          {g.name}
                        </h3>
                        <span style={{ fontSize: '11px', color: 'var(--text-light)', fontFamily: 'var(--font-mono)' }}>
                          id: {g.id}
                        </span>
                      </div>

                      {g.description && (
                        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                          {g.description}
                        </p>
                      )}

                      <div
                        style={{
                          marginTop: '12px',
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                          gap: '10px',
                          backgroundColor: 'var(--bg-app)',
                          padding: '12px 14px',
                          borderRadius: 'var(--radius-md)',
                          border: '1px solid var(--border-subtle)',
                          fontSize: '12.5px',
                        }}
                      >
                        <div>
                          <span style={{ fontWeight: 600, color: 'var(--text-muted)' }}>TRIGGER: </span>
                          <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>
                            {g.triggerCondition || 'None'}
                          </span>
                        </div>
                        <div>
                          <span style={{ fontWeight: 600, color: 'var(--text-muted)' }}>ACTION: </span>
                          <span style={{ color: 'var(--accent-primary)', fontWeight: 500 }}>
                            {g.action}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexShrink: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span
                        style={{
                          fontSize: '12px',
                          fontWeight: 600,
                          color: g.enabled ? 'var(--status-success)' : 'var(--text-muted)',
                        }}
                      >
                        {g.enabled ? 'Active' : 'Disabled'}
                      </span>
                      <label className="toggle-switch">
                        <input
                          type="checkbox"
                          checked={g.enabled}
                          onChange={() => handleToggle(g.id)}
                        />
                        <span className="toggle-slider" />
                      </label>
                    </div>

                    <div style={{ display: 'flex', gap: '4px' }}>
                      <button
                        onClick={() => handleOpenEdit(g)}
                        title="Edit guardrail rule"
                        style={{
                          padding: '6px 8px',
                          borderRadius: 'var(--radius-xs)',
                          backgroundColor: 'var(--bg-muted)',
                          color: 'var(--text-secondary)',
                          border: '1px solid var(--border-light)',
                          cursor: 'pointer',
                        }}
                      >
                        <Pencil size={14} />
                      </button>

                      <button
                        onClick={() => handleDelete(g.id)}
                        title="Delete guardrail"
                        style={{
                          padding: '6px 8px',
                          borderRadius: 'var(--radius-xs)',
                          backgroundColor: 'var(--status-danger-bg)',
                          color: 'var(--status-danger)',
                          border: '1px solid var(--status-danger-border)',
                          cursor: 'pointer',
                        }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Add / Edit Guardrail Modal */}
      {modalOpen && (
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
          onClick={() => setModalOpen(false)}
        >
          <div
            className="animate-slide-down"
            style={{
              width: '100%',
              maxWidth: '580px',
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
              <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>
                {editingId ? 'Edit AI Guardrail in DB' : 'Create Custom AI Guardrail in DB'}
              </h3>
              <button
                onClick={() => setModalOpen(false)}
                style={{ padding: '4px', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleSaveModal}>
              <div
                style={{
                  padding: '20px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '14px',
                  maxHeight: '75vh',
                  overflowY: 'auto',
                }}
              >
                <div>
                  <label style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    GUARDRAIL NAME *
                  </label>
                  <input
                    required
                    value={formName}
                    onChange={(e) => setFormName(e.target.value)}
                    placeholder="e.g. Financial Advice Disclaimer"
                    style={{
                      width: '100%',
                      marginTop: '4px',
                      padding: '8px 12px',
                      fontSize: '13.5px',
                      border: '1px solid var(--border-light)',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: 'var(--bg-app)',
                      color: 'var(--text-primary)',
                      outline: 'none',
                    }}
                  />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <label style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                      CATEGORY
                    </label>
                    <select
                      value={formCategory}
                      onChange={(e) => setFormCategory(e.target.value as GuardrailCategory)}
                      style={{
                        width: '100%',
                        marginTop: '4px',
                        padding: '8px 12px',
                        fontSize: '13px',
                        border: '1px solid var(--border-light)',
                        borderRadius: 'var(--radius-md)',
                        backgroundColor: 'var(--bg-app)',
                        color: 'var(--text-primary)',
                        outline: 'none',
                      }}
                    >
                      <option value="hallucination">Hallucination Prevention</option>
                      <option value="injection">Prompt Injection Defense</option>
                      <option value="pii">PII & Sensitive Redaction</option>
                      <option value="policy">Corporate Policy & Tone</option>
                      <option value="custom">Custom Evaluation Rule</option>
                    </select>
                  </div>

                  <div>
                    <label style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                      STATUS
                    </label>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '10px' }}>
                      <label className="toggle-switch">
                        <input
                          type="checkbox"
                          checked={formEnabled}
                          onChange={(e) => setFormEnabled(e.target.checked)}
                        />
                        <span className="toggle-slider" />
                      </label>
                      <span style={{ fontSize: '13px', color: 'var(--text-primary)' }}>
                        {formEnabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <label style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    DESCRIPTION
                  </label>
                  <input
                    value={formDescription}
                    onChange={(e) => setFormDescription(e.target.value)}
                    placeholder="Brief summary..."
                    style={{
                      width: '100%',
                      marginTop: '4px',
                      padding: '8px 12px',
                      fontSize: '13.5px',
                      border: '1px solid var(--border-light)',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: 'var(--bg-app)',
                      color: 'var(--text-primary)',
                      outline: 'none',
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    INSTRUCTION / RULE SPECIFICATION *
                  </label>
                  <textarea
                    required
                    rows={3}
                    value={formRule}
                    onChange={(e) => setFormRule(e.target.value)}
                    placeholder="Enter the explicit constraint or verification instruction..."
                    style={{
                      width: '100%',
                      marginTop: '4px',
                      padding: '8px 12px',
                      fontSize: '13.5px',
                      border: '1px solid var(--border-light)',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: 'var(--bg-app)',
                      color: 'var(--text-primary)',
                      outline: 'none',
                      resize: 'vertical',
                      fontFamily: 'inherit',
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    TRIGGER CONDITION
                  </label>
                  <input
                    value={formTrigger}
                    onChange={(e) => setFormTrigger(e.target.value)}
                    placeholder="e.g. Cosine similarity < 0.65"
                    style={{
                      width: '100%',
                      marginTop: '4px',
                      padding: '8px 12px',
                      fontSize: '13px',
                      border: '1px solid var(--border-light)',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: 'var(--bg-app)',
                      color: 'var(--text-primary)',
                      outline: 'none',
                      fontFamily: 'var(--font-mono)',
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    ACTION WHEN TRIGGERED
                  </label>
                  <select
                    value={formAction}
                    onChange={(e) => setFormAction(e.target.value)}
                    style={{
                      width: '100%',
                      marginTop: '4px',
                      padding: '8px 12px',
                      fontSize: '13px',
                      border: '1px solid var(--border-light)',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: 'var(--bg-app)',
                      color: 'var(--text-primary)',
                      outline: 'none',
                    }}
                  >
                    <option value="Refuse query with polite disclaimer">Refuse query with polite disclaimer</option>
                    <option value="Mask matching entity with [REDACTED]">Mask matching entity with [REDACTED]</option>
                    <option value="Hard block response and notify admin">Hard block response and log security alert</option>
                    <option value="Fallback to human agent ticket">Fallback to human support ticket</option>
                    <option value="Append compliance warning banner">Append compliance warning banner</option>
                  </select>
                </div>
              </div>

              <div
                style={{
                  padding: '14px 20px',
                  backgroundColor: 'var(--bg-app)',
                  borderTop: '1px solid var(--border-light)',
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: '8px',
                }}
              >
                <button
                  type="button"
                  onClick={() => setModalOpen(false)}
                  style={{
                    padding: '7px 14px',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-muted)',
                    color: 'var(--text-secondary)',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  style={{
                    padding: '7px 18px',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--accent-primary)',
                    color: '#ffffff',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  {editingId ? 'Save Changes' : 'Save to DB'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
