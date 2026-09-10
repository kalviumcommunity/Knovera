'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getStoredAuth, AuthUser } from '@/lib/auth';
import {
  Sparkles,
  ShieldCheck,
  MessageSquare,
  ArrowRight,
  Database,
  Lock,
  FileText,
  Sliders,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react';

export default function RootPortalPage() {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    const auth = getStoredAuth();
    setUser(auth);
  }, []);

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'var(--bg-app)',
        color: 'var(--text-primary)',
      }}
    >
      {/* Top Bar */}
      <header
        style={{
          padding: '18px 32px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: '#ffffff',
          borderBottom: '1px solid var(--border-light)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--accent-slate)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
            }}
          >
            <Sparkles size={18} />
          </div>
          <span style={{ fontWeight: 700, fontSize: '17px', letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
            Knovera Enterprise AI
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
              marginLeft: '6px',
            }}
          >
            RAG Studio
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Link
            href="/login"
            style={{
              padding: '7px 14px',
              fontSize: '13px',
              fontWeight: 500,
              color: 'var(--text-secondary)',
              textDecoration: 'none',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-light)',
              backgroundColor: '#ffffff',
            }}
          >
            User Login
          </Link>
          <Link
            href="/admin/login"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '7px 16px',
              fontSize: '13px',
              fontWeight: 600,
              color: '#ffffff',
              textDecoration: 'none',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--accent-primary)',
              boxShadow: 'var(--shadow-xs)',
            }}
          >
            <ShieldCheck size={15} />
            <span>Admin Console</span>
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main
        style={{
          flex: 1,
          maxWidth: '1080px',
          width: '100%',
          margin: '0 auto',
          padding: '60px 24px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            padding: '5px 12px',
            borderRadius: 'var(--radius-full)',
            backgroundColor: '#eff6ff',
            border: '1px solid #bfdbfe',
            fontSize: '12.5px',
            fontWeight: 500,
            color: 'var(--accent-primary)',
            marginBottom: '20px',
          }}
        >
          <CheckCircle2 size={14} />
          <span>Strict Light-Theme • Dedicated User & Admin Experience</span>
        </div>

        <h1
          style={{
            fontSize: '38px',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            lineHeight: 1.2,
            color: 'var(--text-primary)',
            maxWidth: '740px',
          }}
        >
          Enterprise Conversational Intelligence & Safety Platform
        </h1>

        <p
          style={{
            fontSize: '16px',
            color: 'var(--text-secondary)',
            marginTop: '16px',
            maxWidth: '620px',
            lineHeight: 1.6,
          }}
        >
          Minimalist, grounded ChatGPT-style interface for end users paired with an advanced administrative
          command center for knowledge management, audit logs, and AI guardrails.
        </p>

        {/* Two Dedicated Portals Cards */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
            gap: '24px',
            width: '100%',
            marginTop: '48px',
            textAlign: 'left',
          }}
        >
          {/* Card 1: User Chatbot */}
          <div
            style={{
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-xl)',
              padding: '32px 28px',
              boxShadow: 'var(--shadow-md)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              transition: 'transform 0.15s ease, box-shadow 0.15s ease',
            }}
          >
            <div>
              <div
                style={{
                  width: '46px',
                  height: '46px',
                  borderRadius: 'var(--radius-lg)',
                  backgroundColor: 'var(--bg-muted)',
                  border: '1px solid var(--border-light)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--accent-primary)',
                  marginBottom: '18px',
                }}
              >
                <MessageSquare size={22} />
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 600,
                    padding: '2px 7px',
                    borderRadius: 'var(--radius-xs)',
                    backgroundColor: 'var(--bg-muted)',
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                  }}
                >
                  Route: /chat & /login
                </span>
              </div>

              <h2
                style={{
                  fontSize: '20px',
                  fontWeight: 700,
                  letterSpacing: '-0.02em',
                  color: 'var(--text-primary)',
                  marginTop: '8px',
                }}
              >
                User Chatbot Workspace
              </h2>

              <p
                style={{
                  fontSize: '13.5px',
                  color: 'var(--text-secondary)',
                  marginTop: '8px',
                  lineHeight: 1.55,
                }}
              >
                ChatGPT-style two-panel conversation interface with grouped history (Today, Yesterday, 7 Days,
                Older), 3-dot action menus, Markdown formatting with code block copy buttons, and verified citations.
              </p>

              <div style={{ marginTop: '18px', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle2 size={14} color="var(--status-success)" />
                  <span>Grouped chat history with inline Rename & Delete</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle2 size={14} color="var(--status-success)" />
                  <span>Fixed bottom composer with file attachment & voice action</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle2 size={14} color="var(--status-success)" />
                  <span>Grounded source citation badges & typing indicators</span>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px', marginTop: '28px' }}>
              <Link
                href="/chat"
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  padding: '10px',
                  backgroundColor: 'var(--accent-slate)',
                  color: '#ffffff',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '13.5px',
                  fontWeight: 600,
                  textDecoration: 'none',
                }}
              >
                <span>Launch User Chat</span>
                <ArrowRight size={15} />
              </Link>
              <Link
                href="/login"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '10px 14px',
                  backgroundColor: 'var(--bg-app)',
                  border: '1px solid var(--border-light)',
                  color: 'var(--text-secondary)',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '13px',
                  fontWeight: 500,
                  textDecoration: 'none',
                }}
              >
                Login Page
              </Link>
            </div>
          </div>

          {/* Card 2: Enterprise Admin Dashboard */}
          <div
            style={{
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-xl)',
              padding: '32px 28px',
              boxShadow: 'var(--shadow-md)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              transition: 'transform 0.15s ease, box-shadow 0.15s ease',
            }}
          >
            <div>
              <div
                style={{
                  width: '46px',
                  height: '46px',
                  borderRadius: 'var(--radius-lg)',
                  backgroundColor: '#eff6ff',
                  border: '1px solid #bfdbfe',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--accent-primary)',
                  marginBottom: '18px',
                }}
              >
                <ShieldCheck size={24} />
              </div>

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
                    textTransform: 'uppercase',
                  }}
                >
                  Route: /admin & /admin/login
                </span>
              </div>

              <h2
                style={{
                  fontSize: '20px',
                  fontWeight: 700,
                  letterSpacing: '-0.02em',
                  color: 'var(--text-primary)',
                  marginTop: '8px',
                }}
              >
                Enterprise Admin Dashboard
              </h2>

              <p
                style={{
                  fontSize: '13.5px',
                  color: 'var(--text-secondary)',
                  marginTop: '8px',
                  lineHeight: 1.55,
                }}
              >
                Multi-section administrator console with Dashboard telemetry, file uploads (PDF, DOCX, CSV),
                vector knowledge chunk search, priority guardrails with drag/reordering, audit logs, and settings.
              </p>

              <div style={{ marginTop: '18px', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle2 size={14} color="var(--accent-primary)" />
                  <span>Custom Guardrails: Add, Edit, Delete & Priority reorder</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle2 size={14} color="var(--accent-primary)" />
                  <span>Document drag & drop upload, chunk inspector, and replace/delete</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle2 size={14} color="var(--accent-primary)" />
                  <span>Audit activity logs with search, filter, and JSON modal</span>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px', marginTop: '28px' }}>
              <Link
                href="/admin"
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  padding: '10px',
                  backgroundColor: 'var(--accent-primary)',
                  color: '#ffffff',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '13.5px',
                  fontWeight: 600,
                  textDecoration: 'none',
                  boxShadow: 'var(--shadow-xs)',
                }}
              >
                <span>Admin Dashboard</span>
                <ArrowRight size={15} />
              </Link>
              <Link
                href="/admin/login"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '10px 14px',
                  backgroundColor: 'var(--bg-app)',
                  border: '1px solid var(--border-light)',
                  color: 'var(--text-secondary)',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '13px',
                  fontWeight: 500,
                  textDecoration: 'none',
                }}
              >
                Admin Login
              </Link>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          padding: '20px 32px',
          borderTop: '1px solid var(--border-light)',
          backgroundColor: '#ffffff',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '12px',
          color: 'var(--text-muted)',
        }}
      >
        <span>Knovera RAG Engine & Studio • Multi-Route Architecture</span>
        <div style={{ display: 'flex', gap: '16px' }}>
          <Link href="/login" style={{ color: 'inherit', textDecoration: 'none' }}>User Login (/login)</Link>
          <Link href="/chat" style={{ color: 'inherit', textDecoration: 'none' }}>User Chat (/chat)</Link>
          <Link href="/admin/login" style={{ color: 'inherit', textDecoration: 'none' }}>Admin Login (/admin/login)</Link>
          <Link href="/admin" style={{ color: 'inherit', textDecoration: 'none' }}>Admin Console (/admin)</Link>
        </div>
      </footer>
    </div>
  );
}
