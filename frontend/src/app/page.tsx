'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getStoredAuth, setStoredAuth, AuthUser } from '@/lib/auth';
import {
  Sparkles,
  ShieldCheck,
  Search,
  Lock,
  FileCheck2,
  Cpu,
  Layers,
  CheckCircle2,
  ArrowRight,
  LogOut,
  User,
  ExternalLink,
} from 'lucide-react';

export default function RootPortalPage() {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const auth = getStoredAuth();
    setUser(auth);
  }, []);

  const handleSignOut = () => {
    setStoredAuth(null);
    setUser(null);
  };

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
      {/* Top Header */}
      <header
        style={{
          padding: '16px 36px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: '#ffffff',
          borderBottom: '1px solid var(--border-light)',
          position: 'sticky',
          top: 0,
          zIndex: 50,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '34px',
              height: '34px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--accent-slate)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
              boxShadow: '0 2px 5px rgba(15, 23, 42, 0.15)',
            }}
          >
            <Sparkles size={18} color="#60a5fa" />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontWeight: 800, fontSize: '18px', letterSpacing: '-0.03em', color: 'var(--text-primary)' }}>
              Knovera
            </span>
            <span
              style={{
                fontSize: '11px',
                padding: '2px 8px',
                borderRadius: 'var(--radius-full)',
                backgroundColor: '#eff6ff',
                color: 'var(--accent-primary)',
                fontWeight: 600,
                border: '1px solid #bfdbfe',
              }}
            >
              Enterprise AI
            </span>
          </div>
        </div>

        {/* User Account / Auth Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {mounted && user ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-full)',
                  backgroundColor: '#f1f5f9',
                  border: '1px solid var(--border-light)',
                  fontSize: '13px',
                }}
              >
                <div
                  style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    backgroundColor: 'var(--accent-primary)',
                    color: '#ffffff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '11px',
                    fontWeight: 700,
                  }}
                >
                  {user.avatarInitials || user.name?.charAt(0) || 'U'}
                </div>
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                  {user.name || user.email}
                </span>
              </div>
              <button
                onClick={handleSignOut}
                title="Sign out of current account"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  padding: '7px 12px',
                  fontSize: '12.5px',
                  fontWeight: 500,
                  color: 'var(--text-secondary)',
                  backgroundColor: '#ffffff',
                  border: '1px solid var(--border-light)',
                  borderRadius: 'var(--radius-md)',
                  cursor: 'pointer',
                  transition: 'background 0.15s ease',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f8fafc')}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#ffffff')}
              >
                <LogOut size={13} />
                <span>Sign Out</span>
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Link
                href="/login"
                style={{
                  padding: '8px 16px',
                  fontSize: '13.5px',
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                  textDecoration: 'none',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-medium)',
                  backgroundColor: '#ffffff',
                  transition: 'background 0.15s ease',
                }}
              >
                Sign In
              </Link>
              <Link
                href="/signup"
                style={{
                  padding: '8px 16px',
                  fontSize: '13.5px',
                  fontWeight: 600,
                  color: '#ffffff',
                  textDecoration: 'none',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'var(--accent-slate)',
                  boxShadow: 'var(--shadow-xs)',
                  transition: 'opacity 0.15s ease',
                }}
              >
                Create Account
              </Link>
            </div>
          )}
        </div>
      </header>

      {/* Main Container */}
      <main
        style={{
          flex: 1,
          maxWidth: '1040px',
          width: '100%',
          margin: '0 auto',
          padding: '64px 24px 80px 24px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        {/* Badge */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: 'var(--radius-full)',
            backgroundColor: '#eff6ff',
            border: '1px solid #bfdbfe',
            fontSize: '12.5px',
            fontWeight: 600,
            color: 'var(--accent-primary)',
            marginBottom: '24px',
          }}
        >
          <Sparkles size={14} color="var(--accent-primary)" />
          <span>Grounded Knowledge Retrieval & Enterprise Intelligence</span>
        </div>

        {/* Hero Title */}
        <h1
          style={{
            fontSize: '44px',
            fontWeight: 800,
            letterSpacing: '-0.035em',
            lineHeight: 1.15,
            color: 'var(--text-primary)',
            maxWidth: '820px',
            margin: '0 0 20px 0',
          }}
        >
          Knovera — Verified Enterprise Intelligence, Protected & Grounded
        </h1>

        {/* Hero Subtitle */}
        <p
          style={{
            fontSize: '17px',
            color: 'var(--text-secondary)',
            maxWidth: '680px',
            lineHeight: 1.65,
            margin: '0 0 36px 0',
          }}
        >
          Knovera transforms corporate knowledge silos into an interactive, trustworthy conversational experience.
          Engineered with semantic retrieval, mathematical grounding, and real-time guardrail defense.
        </p>

        {/* Moveable AI Mode Interactive Callout Box */}
        <div
          style={{
            width: '100%',
            maxWidth: '740px',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-xl)',
            padding: '24px 28px',
            boxShadow: '0 10px 25px -5px rgba(15, 23, 42, 0.05)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '20px',
            textAlign: 'left',
            marginBottom: '56px',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '4px',
              height: '100%',
              backgroundColor: 'var(--accent-primary)',
            }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div
              style={{
                width: '44px',
                height: '44px',
                borderRadius: 'var(--radius-lg)',
                backgroundColor: '#eff6ff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-primary)',
                flexShrink: 0,
              }}
            >
              <Sparkles size={22} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h2 style={{ fontSize: '15.5px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                  Moveable AI Mode Assistant
                </h2>
                <span
                  style={{
                    fontSize: '10.5px',
                    padding: '2px 6px',
                    borderRadius: 'var(--radius-xs)',
                    backgroundColor: '#dcfce7',
                    color: '#15803d',
                    fontWeight: 600,
                  }}
                >
                  Active On Screen
                </span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: '4px 0 0 0', lineHeight: 1.45 }}>
                Click the floating <strong>AI Mode</strong> pill at any time to open your chatbot.
                You can drag and position it anywhere on your screen.
              </p>
            </div>
          </div>

          <div style={{ flexShrink: 0 }}>
            {mounted && user ? (
              <button
                onClick={() => router.push('/chat')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '9px 16px',
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
                <span>Open Chat</span>
                <ArrowRight size={14} />
              </button>
            ) : (
              <button
                onClick={() => router.push('/login')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '9px 16px',
                  backgroundColor: 'var(--accent-primary)',
                  color: '#ffffff',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  border: 'none',
                  boxShadow: 'var(--shadow-xs)',
                }}
              >
                <span>Sign In to Chat</span>
                <ArrowRight size={14} />
              </button>
            )}
          </div>
        </div>

        {/* Section: What is Knovera */}
        <div style={{ width: '100%', textAlign: 'left', marginBottom: '40px' }}>
          <h2
            style={{
              fontSize: '22px',
              fontWeight: 700,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
              marginBottom: '12px',
              textAlign: 'center',
            }}
          >
            Why Organizations Rely on Knovera
          </h2>
          <p
            style={{
              fontSize: '14.5px',
              color: 'var(--text-secondary)',
              maxWidth: '640px',
              margin: '0 auto 36px auto',
              textAlign: 'center',
              lineHeight: 1.6,
            }}
          >
            Traditional AI models hallucinate answers or leak confidential information.
            Knovera redefines organizational search through zero-trust retrieval architecture.
          </p>

          {/* Pillars Grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '20px',
            }}
          >
            {/* Pillar 1 */}
            <div
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-lg)',
                padding: '24px 22px',
                boxShadow: 'var(--shadow-xs)',
              }}
            >
              <div
                style={{
                  width: '38px',
                  height: '38px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: '#eff6ff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--accent-primary)',
                  marginBottom: '14px',
                }}
              >
                <FileCheck2 size={20} />
              </div>
              <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 8px 0' }}>
                Mathematical Grounding & Citations
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                Responses are strictly synthesized from indexed internal documentation.
                Every paragraph is accompanied by verified citation numbers linked directly to origin files.
              </p>
            </div>

            {/* Pillar 2 */}
            <div
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-lg)',
                padding: '24px 22px',
                boxShadow: 'var(--shadow-xs)',
              }}
            >
              <div
                style={{
                  width: '38px',
                  height: '38px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: '#f0fdf4',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#16a34a',
                  marginBottom: '14px',
                }}
              >
                <ShieldCheck size={20} />
              </div>
              <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 8px 0' }}>
                Active AI Guardrails
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                Enterprise defense layers scan input queries and generated outputs in real-time,
                intercepting prompt injection attacks, redacting sensitive PII, and enforcing policy compliance.
              </p>
            </div>

            {/* Pillar 3 */}
            <div
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-lg)',
                padding: '24px 22px',
                boxShadow: 'var(--shadow-xs)',
              }}
            >
              <div
                style={{
                  width: '38px',
                  height: '38px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: '#faf5ff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#9333ea',
                  marginBottom: '14px',
                }}
              >
                <Cpu size={20} />
              </div>
              <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 8px 0' }}>
                Semantic Intent Understanding
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                Equipped with custom semantic normalization for corporate inquiries, recognizing intent whether a user
                asks to define, describe, clarify, or explain company procedures.
              </p>
            </div>

            {/* Pillar 4 */}
            <div
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-lg)',
                padding: '24px 22px',
                boxShadow: 'var(--shadow-xs)',
              }}
            >
              <div
                style={{
                  width: '38px',
                  height: '38px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: '#fff7ed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#ea580c',
                  marginBottom: '14px',
                }}
              >
                <Lock size={20} />
              </div>
              <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 8px 0' }}>
                Data Sovereignty & Encryption
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                All document chunks and vector representations are encrypted at rest with AES-256 GCM
                and transmitted over TLS 1.3, guaranteeing zero data leakage or unauthorized cross-pollination.
              </p>
            </div>
          </div>
        </div>

        {/* Narrative Section About Knovera */}
        <div
          style={{
            width: '100%',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-xl)',
            padding: '36px 32px',
            textAlign: 'left',
            boxShadow: 'var(--shadow-sm)',
            marginTop: '20px',
          }}
        >
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '12px' }}>
            About Knovera Enterprise AI
          </h2>
          <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.7, margin: '0 0 14px 0' }}>
            In modern enterprises, thousands of hours are lost searching through fragmented intranet pages, handbook revisions,
            cloud drives, and outdated PDF manuals. Knovera was built to unify these sources into an intelligent, interactive
            knowledge fabric that responds in natural conversation while maintaining rigorous institutional precision.
          </p>
          <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.7, margin: 0 }}>
            Every interaction is recorded in privacy-safe audit trails, giving organizations complete transparency and confidence
            as their teams query corporate policy, legal guidelines, and operational standards.
          </p>
        </div>
      </main>

      {/* Clean Footer (No Admin Links) */}
      <footer
        style={{
          padding: '24px 36px',
          borderTop: '1px solid var(--border-light)',
          backgroundColor: '#ffffff',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '12.5px',
          color: 'var(--text-muted)',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Knovera AI</span>
          <span>•</span>
          <span>Enterprise Grounded Knowledge Platform</span>
        </div>
        <div style={{ display: 'flex', gap: '20px' }}>
          <span style={{ color: 'var(--text-muted)' }}>Secure RAG Architecture</span>
          <span style={{ color: 'var(--text-muted)' }}>AES-256 Encryption</span>
          <span style={{ color: 'var(--text-muted)' }}>Zero Data Spill</span>
        </div>
      </footer>
    </div>
  );
}
