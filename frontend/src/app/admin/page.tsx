'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredAuth, logout, AuthUser } from '@/lib/auth';
import { api } from '@/lib/api';
import { HealthResponse } from '@/lib/types';

import AdminLayout, { AdminTab } from '@/components/admin/AdminLayout';
import AdminDashboardView from '@/components/admin/AdminDashboardView';
import SourcesUploadView from '@/components/admin/SourcesUploadView';
import KnowledgeBaseView from '@/components/admin/KnowledgeBaseView';
import GuardrailsView from '@/components/admin/GuardrailsView';
import LogsView from '@/components/admin/LogsView';
import SettingsView from '@/components/admin/SettingsView';
import { ShieldAlert, LogIn, ArrowRight } from 'lucide-react';
import Link from 'next/link';

export default function AdminPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<AdminTab>('dashboard');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  // Check auth session
  useEffect(() => {
    const auth = getStoredAuth();
    setCurrentUser(auth);
    setIsCheckingAuth(false);
  }, []);

  // Health check
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const data = await api.getHealth();
        setHealth(data);
      } catch {
        setHealth(null);
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleLogout = () => {
    logout();
    router.push('/admin/login');
  };

  if (isCheckingAuth) {
    return (
      <div
        style={{
          height: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'var(--bg-app)',
          color: 'var(--text-secondary)',
          fontSize: '14px',
        }}
      >
        <span>Verifying administrator credentials...</span>
      </div>
    );
  }

  // If not logged in as admin, present a clean security gate with instant access button
  if (!currentUser || currentUser.role !== 'admin') {
    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'var(--bg-app)',
          padding: '24px',
        }}
      >
        <div
          className="animate-fade-in"
          style={{
            maxWidth: '440px',
            width: '100%',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-xl)',
            padding: '36px 32px',
            textAlign: 'center',
            boxShadow: 'var(--shadow-md)',
          }}
        >
          <div
            style={{
              width: '48px',
              height: '48px',
              borderRadius: 'var(--radius-lg)',
              backgroundColor: '#fef2f2',
              border: '1px solid #fecaca',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px auto',
              color: 'var(--status-danger)',
            }}
          >
            <ShieldAlert size={26} />
          </div>

          <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            Administrator Access Required
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.5 }}>
            The Knovera Admin Console is protected. Please authenticate with administrator credentials to manage
            sources, the knowledge base, and safety guardrails.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '24px' }}>
            <Link
              href="/admin/login"
              style={{
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
              }}
            >
              <LogIn size={16} />
              <span>Go to Admin Login</span>
            </Link>

            <Link
              href="/chat"
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                padding: '9px',
                backgroundColor: 'var(--bg-muted)',
                color: 'var(--text-secondary)',
                borderRadius: 'var(--radius-md)',
                fontSize: '13px',
                textDecoration: 'none',
              }}
            >
              <span>Return to User Chatbot</span>
              <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <AdminLayout
      activeTab={activeTab}
      onSelectTab={setActiveTab}
      onBackToChat={() => router.push('/chat')}
      onLogout={handleLogout}
      adminName={currentUser.name}
      healthStatus={health?.status || 'ok'}
    >
      {activeTab === 'dashboard' && <AdminDashboardView />}
      {activeTab === 'sources' && <SourcesUploadView />}
      {activeTab === 'knowledge' && <KnowledgeBaseView />}
      {activeTab === 'guardrails' && <GuardrailsView />}
      {activeTab === 'logs' && <LogsView />}
      {activeTab === 'settings' && <SettingsView />}
    </AdminLayout>
  );
}
