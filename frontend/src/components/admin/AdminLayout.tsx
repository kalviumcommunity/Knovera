'use client';

import React from 'react';
import {
  LayoutDashboard,
  UploadCloud,
  Database,
  ShieldCheck,
  FileText,
  Settings,
  ArrowLeft,
  Sparkles,
  Server,
  ExternalLink,
  LogOut,
  User,
} from 'lucide-react';

export type AdminTab =
  | 'dashboard'
  | 'sources'
  | 'knowledge'
  | 'guardrails'
  | 'logs'
  | 'settings';

interface AdminLayoutProps {
  activeTab: AdminTab;
  onSelectTab: (tab: AdminTab) => void;
  onBackToChat: () => void;
  onLogout?: () => void;
  adminName?: string;
  children: React.ReactNode;
  healthStatus?: 'ok' | 'degraded' | 'unhealthy' | null;
}

export default function AdminLayout({
  activeTab,
  onSelectTab,
  onBackToChat,
  onLogout,
  adminName = 'K Jayanth (Admin)',
  children,
  healthStatus = 'ok',
}: AdminLayoutProps) {
  const navItems = [
    {
      id: 'dashboard' as AdminTab,
      label: 'Dashboard',
      icon: <LayoutDashboard size={18} />,
      badge: undefined,
    },
    {
      id: 'sources' as AdminTab,
      label: 'Sources / Upload',
      icon: <UploadCloud size={18} />,
      badge: undefined,
    },
    {
      id: 'knowledge' as AdminTab,
      label: 'Knowledge Base',
      icon: <Database size={18} />,
      badge: 'MongoDB',
    },
    {
      id: 'guardrails' as AdminTab,
      label: 'Guardrails',
      icon: <ShieldCheck size={18} />,
      badge: 'Active',
      badgeColor: '#059669',
      badgeBg: '#ecfdf5',
    },
    {
      id: 'logs' as AdminTab,
      label: 'Logs & Audit',
      icon: <FileText size={18} />,
      badge: undefined,
    },
    {
      id: 'settings' as AdminTab,
      label: 'Settings',
      icon: <Settings size={18} />,
      badge: undefined,
    },
  ];

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        backgroundColor: 'var(--bg-app)',
        color: 'var(--text-primary)',
        overflow: 'hidden',
      }}
    >
      {/* Enterprise Top Bar */}
      <header
        style={{
          height: '54px',
          borderBottom: '1px solid var(--border-light)',
          backgroundColor: '#ffffff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 20px',
          zIndex: 20,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <button
            onClick={onBackToChat}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-muted)',
              border: '1px solid var(--border-light)',
              fontSize: '13px',
              fontWeight: 500,
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--bg-sidebar-hover)';
              e.currentTarget.style.color = 'var(--text-primary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--bg-muted)';
              e.currentTarget.style.color = 'var(--text-secondary)';
            }}
          >
            <ArrowLeft size={15} />
            <span>Back to Chat</span>
          </button>

          <div
            style={{
              height: '18px',
              width: '1px',
              backgroundColor: 'var(--border-light)',
            }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div
              style={{
                width: '26px',
                height: '26px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'var(--accent-slate)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#ffffff',
              }}
            >
              <Sparkles size={14} />
            </div>
            <span style={{ fontWeight: 600, fontSize: '15px', color: 'var(--text-primary)' }}>
              Knovera Studio
            </span>
            <span
              style={{
                fontSize: '11px',
                padding: '2px 7px',
                borderRadius: 'var(--radius-full)',
                backgroundColor: '#eff6ff',
                color: 'var(--accent-primary)',
                fontWeight: 600,
                border: '1px solid #bfdbfe',
              }}
            >
              Admin Center
            </span>
          </div>
        </div>

        {/* Right Status Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px',
              color: 'var(--text-secondary)',
            }}
          >
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor:
                  healthStatus === 'ok'
                    ? 'var(--status-success)'
                    : healthStatus === 'degraded'
                    ? 'var(--status-warning)'
                    : 'var(--status-danger)',
              }}
            />
            <span>
              Backend: {healthStatus === 'ok' ? 'Connected (Port 8000)' : 'Simulation / Standby'}
            </span>
          </div>

          <div
            style={{
              fontSize: '12px',
              padding: '4px 8px',
              backgroundColor: 'var(--bg-muted)',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-secondary)',
              fontWeight: 500,
            }}
          >
            Enterprise Org: Knovera Global
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              paddingLeft: '10px',
              borderLeft: '1px solid var(--border-light)',
            }}
          >
            <div
              style={{
                width: '26px',
                height: '26px',
                borderRadius: 'var(--radius-full)',
                backgroundColor: 'var(--accent-slate)',
                color: '#ffffff',
                fontSize: '11px',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              KJ
            </div>
            <span style={{ fontSize: '12.5px', fontWeight: 500, color: 'var(--text-primary)' }}>
              {adminName}
            </span>
          </div>

          {onLogout && (
            <button
              onClick={onLogout}
              title="Sign out of Admin Console"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                padding: '5px 10px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--bg-app)',
                border: '1px solid var(--border-light)',
                fontSize: '12px',
                color: 'var(--status-danger)',
                cursor: 'pointer',
                transition: 'background 0.15s ease',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--status-danger-bg)')}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-app)')}
            >
              <LogOut size={13} />
              <span>Sign Out</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Area: Sidebar + Active Tab Content */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Admin Navigation Sidebar */}
        <nav
          style={{
            width: '240px',
            backgroundColor: 'var(--bg-sidebar)',
            borderRight: '1px solid var(--border-light)',
            display: 'flex',
            flexDirection: 'column',
            padding: '16px 10px',
            flexShrink: 0,
            overflowY: 'auto',
          }}
        >
          <div
            style={{
              fontSize: '11px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              color: 'var(--text-light)',
              padding: '0 8px 8px 8px',
            }}
          >
            Platform Management
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '9px 12px',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: isActive ? 'var(--bg-sidebar-active)' : 'transparent',
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontWeight: isActive ? 600 : 400,
                    fontSize: '13.5px',
                    cursor: 'pointer',
                    transition: 'all 0.12s ease',
                    textAlign: 'left',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = 'var(--bg-sidebar-hover)';
                      e.currentTarget.style.color = 'var(--text-primary)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.color = 'var(--text-secondary)';
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ color: isActive ? 'var(--accent-primary)' : 'var(--text-muted)' }}>
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </div>

                  {item.badge && (
                    <span
                      style={{
                        fontSize: '10.5px',
                        padding: '1px 6px',
                        borderRadius: 'var(--radius-full)',
                        backgroundColor: item.badgeBg || 'var(--bg-muted)',
                        color: item.badgeColor || 'var(--text-muted)',
                        fontWeight: 600,
                      }}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Quick RAG stats badge in sidebar bottom */}
          <div
            style={{
              marginTop: 'auto',
              padding: '12px',
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-lg)',
              boxShadow: 'var(--shadow-xs)',
            }}
          >
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)' }}>
              RAG ENGINE HEALTH
            </div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginTop: '6px',
              }}
            >
              <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Vector Pipeline
              </span>
              <span
                style={{
                  fontSize: '11px',
                  color: 'var(--status-success)',
                  backgroundColor: 'var(--status-success-bg)',
                  padding: '1px 6px',
                  borderRadius: 'var(--radius-full)',
                  fontWeight: 600,
                }}
              >
                Operational
              </span>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Cosine similarity threshold: 0.70
            </div>
          </div>
        </nav>

        {/* Tab Content Canvas */}
        <main
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '24px 28px',
            backgroundColor: 'var(--bg-app)',
          }}
        >
          <div style={{ maxWidth: '1200px', margin: '0 auto' }}>{children}</div>
        </main>
      </div>
    </div>
  );
}
