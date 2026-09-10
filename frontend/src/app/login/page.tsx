'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { loginAsUser, DEMO_CREDENTIALS } from '@/lib/auth';
import {
  Sparkles,
  Lock,
  Mail,
  Eye,
  EyeOff,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';

export default function UserLoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      await loginAsUser(email, password);
      router.push('/chat');
    } catch (err: any) {
      setError(err.message || 'Login failed. Please check your credentials.');
      setIsLoading(false);
    }
  };

  const handleFillDemo = async () => {
    setEmail(DEMO_CREDENTIALS.user.email);
    setPassword(DEMO_CREDENTIALS.user.password);
    setError(null);
    setIsLoading(true);
    try {
      await loginAsUser(DEMO_CREDENTIALS.user.email, DEMO_CREDENTIALS.user.password);
      router.push('/chat');
    } catch (err: any) {
      setError(err.message);
      setIsLoading(false);
    }
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
      {/* Top Navbar */}
      <header
        style={{
          padding: '18px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid var(--border-light)',
          backgroundColor: '#ffffff',
        }}
      >
        <Link href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '10px' }}>
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
          <div>
            <span style={{ fontWeight: 700, fontSize: '16px', letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
              Knovera AI
            </span>
            <span
              style={{
                fontSize: '11px',
                padding: '1px 6px',
                borderRadius: 'var(--radius-xs)',
                backgroundColor: 'var(--bg-muted)',
                color: 'var(--text-muted)',
                fontWeight: 600,
                marginLeft: '8px',
              }}
            >
              User Portal
            </span>
          </div>
        </Link>

        <Link
          href="/admin/login"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12.5px',
            color: 'var(--text-secondary)',
            textDecoration: 'none',
            padding: '6px 12px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'var(--bg-app)',
            border: '1px solid var(--border-light)',
            fontWeight: 500,
            transition: 'all 0.15s ease',
          }}
        >
          <ShieldCheck size={14} color="var(--accent-primary)" />
          <span>Admin Console Login →</span>
        </Link>
      </header>

      {/* Main Content Form Box */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '32px 16px',
        }}
      >
        <div
          className="animate-fade-in"
          style={{
            width: '100%',
            maxWidth: '420px',
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-xl)',
            padding: '36px 32px',
            boxShadow: 'var(--shadow-md)',
          }}
        >
          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: '28px' }}>
            <div
              style={{
                width: '44px',
                height: '44px',
                borderRadius: 'var(--radius-lg)',
                backgroundColor: 'var(--bg-muted)',
                border: '1px solid var(--border-light)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 14px auto',
                color: 'var(--accent-primary)',
              }}
            >
              <Sparkles size={22} />
            </div>

            <h1
              style={{
                fontSize: '22px',
                fontWeight: 700,
                letterSpacing: '-0.02em',
                color: 'var(--text-primary)',
              }}
            >
              Welcome back
            </h1>
            <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Sign in to your Knovera AI workspace
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div
              className="animate-fade-in"
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '10px',
                padding: '10px 12px',
                marginBottom: '20px',
                backgroundColor: 'var(--status-danger-bg)',
                border: '1px solid var(--status-danger-border)',
                borderRadius: 'var(--radius-md)',
                fontSize: '12.5px',
                color: 'var(--status-danger)',
              }}
            >
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '1px' }} />
              <div>{error}</div>
            </div>
          )}

          {/* Login Form */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label
                htmlFor="user-email"
                style={{ display: 'block', fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}
              >
                Work Email Address
              </label>
              <div style={{ position: 'relative' }}>
                <Mail
                  size={16}
                  style={{
                    position: 'absolute',
                    left: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    color: 'var(--text-light)',
                  }}
                />
                <input
                  id="user-email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@company.com"
                  style={{
                    width: '100%',
                    padding: '9px 12px 9px 36px',
                    fontSize: '14px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                    outline: 'none',
                    transition: 'border-color 0.15s ease',
                  }}
                />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <label
                  htmlFor="user-password"
                  style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}
                >
                  Password
                </label>
                <a
                  href="#forgot"
                  onClick={(e) => {
                    e.preventDefault();
                    alert('Password reset link sent to registered email.');
                  }}
                  style={{ fontSize: '12px', color: 'var(--accent-primary)', textDecoration: 'none' }}
                >
                  Forgot password?
                </a>
              </div>
              <div style={{ position: 'relative' }}>
                <Lock
                  size={16}
                  style={{
                    position: 'absolute',
                    left: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    color: 'var(--text-light)',
                  }}
                />
                <input
                  id="user-password"
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  style={{
                    width: '100%',
                    padding: '9px 36px 9px 36px',
                    fontSize: '14px',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-app)',
                    color: 'var(--text-primary)',
                    outline: 'none',
                    transition: 'border-color 0.15s ease',
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  style={{
                    position: 'absolute',
                    right: '10px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    padding: '4px',
                  }}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input
                id="remember"
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                style={{ accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
              />
              <label htmlFor="remember" style={{ fontSize: '12.5px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                Keep me signed in on this device
              </label>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              style={{
                width: '100%',
                marginTop: '4px',
                padding: '10px',
                backgroundColor: 'var(--accent-slate)',
                color: '#ffffff',
                border: 'none',
                borderRadius: 'var(--radius-md)',
                fontSize: '14px',
                fontWeight: 600,
                cursor: isLoading ? 'not-allowed' : 'pointer',
                transition: 'background 0.15s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
              onMouseEnter={(e) => {
                if (!isLoading) e.currentTarget.style.backgroundColor = 'var(--accent-slate-hover)';
              }}
              onMouseLeave={(e) => {
                if (!isLoading) e.currentTarget.style.backgroundColor = 'var(--accent-slate)';
              }}
            >
              {isLoading ? (
                <span>Signing in...</span>
              ) : (
                <>
                  <span>Sign In</span>
                  <ArrowRight size={15} />
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div
            style={{
              position: 'relative',
              textAlign: 'center',
              margin: '22px 0 16px 0',
            }}
          >
            <div style={{ borderBottom: '1px solid var(--border-light)', position: 'absolute', top: '50%', width: '100%' }} />
            <span
              style={{
                position: 'relative',
                backgroundColor: '#ffffff',
                padding: '0 10px',
                fontSize: '11.5px',
                color: 'var(--text-light)',
                textTransform: 'uppercase',
                fontWeight: 600,
              }}
            >
              Quick Demonstration Access
            </span>
          </div>

          {/* Quick Demo Login Button */}
          <button
            type="button"
            onClick={handleFillDemo}
            disabled={isLoading}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '9px 12px',
              backgroundColor: 'var(--accent-primary-subtle)',
              border: '1px solid var(--accent-primary-border)',
              borderRadius: 'var(--radius-md)',
              fontSize: '13px',
              fontWeight: 600,
              color: 'var(--accent-primary)',
              cursor: 'pointer',
              transition: 'background 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#dbeafe')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'var(--accent-primary-subtle)')}
          >
            <Sparkles size={15} />
            <span>One-Click Demo User Sign In</span>
          </button>
          <div style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-light)', marginTop: '4px' }}>
            Credentials: <code>user@knovera.ai</code> / <code>user123</code>
          </div>

          {/* Admin Switcher Footer */}
          <div
            style={{
              marginTop: '28px',
              paddingTop: '16px',
              borderTop: '1px solid var(--border-subtle)',
              textAlign: 'center',
              fontSize: '12.5px',
              color: 'var(--text-secondary)',
            }}
          >
            Are you a platform administrator?{' '}
            <Link
              href="/admin/login"
              style={{
                color: 'var(--accent-primary)',
                fontWeight: 600,
                textDecoration: 'none',
              }}
            >
              Sign in to Admin Console →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
