'use client';

import React, { useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { getStoredAuth } from '@/lib/auth';
import { Sparkles, MessageSquare, Bot } from 'lucide-react';

export default function FloatingAiButton() {
  const router = useRouter();
  const pathname = usePathname();
  const [isHovered, setIsHovered] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Don't show on the main chat page since user is already in the full chatbot interface
  if (!mounted || pathname === '/chat') {
    return null;
  }

  const handleClick = () => {
    const auth = getStoredAuth();
    if (auth && auth.role) {
      // User or admin is logged in -> open chatbot directly
      router.push('/chat');
    } else {
      // Not logged in -> redirect to login page
      router.push('/login');
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        zIndex: 9999,
      }}
    >
      <button
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        title="Open AI Mode (Knovera Chatbot)"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: isHovered ? '12px 20px 12px 16px' : '12px 18px',
          backgroundColor: 'var(--accent-slate)',
          color: '#ffffff',
          borderRadius: 'var(--radius-full)',
          boxShadow: isHovered
            ? '0 12px 24px -4px rgba(15, 23, 42, 0.25), 0 4px 8px -2px rgba(15, 23, 42, 0.15)'
            : '0 8px 16px -2px rgba(15, 23, 42, 0.18), 0 2px 6px -1px rgba(15, 23, 42, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          cursor: 'pointer',
          transform: isHovered ? 'translateY(-2px)' : 'translateY(0)',
          transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Animated icon with pulse dot */}
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
          <Sparkles
            size={19}
            color="#60a5fa"
            style={{
              transform: isHovered ? 'rotate(15deg) scale(1.1)' : 'rotate(0) scale(1)',
              transition: 'transform 0.2s ease',
            }}
          />
          {/* Active green pulsing dot */}
          <span
            style={{
              position: 'absolute',
              top: '-3px',
              right: '-3px',
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              backgroundColor: '#10b981',
              boxShadow: '0 0 6px #10b981',
            }}
          />
        </div>

        {/* Text label */}
        <div style={{ display: 'flex', flexDirection: 'column', textAlign: 'left' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                fontSize: '13.5px',
                fontWeight: 600,
                letterSpacing: '-0.01em',
                lineHeight: 1.2,
                color: '#ffffff',
              }}
            >
              AI Mode
            </span>
            <span
              style={{
                fontSize: '10px',
                padding: '1px 5px',
                borderRadius: 'var(--radius-xs)',
                backgroundColor: 'rgba(59, 130, 246, 0.25)',
                color: '#93c5fd',
                fontWeight: 600,
                letterSpacing: '0.04em',
              }}
            >
              RAG
            </span>
          </div>
          {isHovered && (
            <span
              className="animate-fade-in"
              style={{
                fontSize: '11px',
                color: '#94a3b8',
                marginTop: '1px',
                lineHeight: 1,
              }}
            >
              Ask documents & policies
            </span>
          )}
        </div>
      </button>
    </div>
  );
}
