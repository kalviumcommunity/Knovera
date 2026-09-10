'use client';

import React, { useState, useRef, useEffect } from 'react';
import { api } from '@/lib/api';
import { ChatMessage, SourceCitation } from '@/lib/types';
import CitationCard from './CitationCard';
import {
  Send,
  Sparkles,
  User,
  Bot,
  RotateCcw,
  Gauge,
  Info,
  Clock,
  ExternalLink,
} from 'lucide-react';

interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceCitation[];
  latency_ms?: number;
}

export default function ChatView() {
  const [sessionId, setSessionId] = useState(() => `session_${Date.now()}`);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatTurn[]>([
    {
      id: 'init',
      role: 'assistant',
      content:
        'Hello! I am Knovera, your conversational assistant. Ask me questions about company documentation, and I will maintain context across our conversation while citing sources.',
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [tokenEstimate, setTokenEstimate] = useState(35);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput('');

    const userTurn: ChatTurn = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: userText,
    };

    setMessages((prev) => [...prev, userTurn]);
    setLoading(true);

    // Rough token estimate (chars / 4)
    setTokenEstimate((prev) => Math.min(2000, prev + Math.ceil(userText.length / 4)));

    try {
      const historyPayload: ChatMessage[] = messages
        .filter((m) => m.id !== 'init')
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await api.sendChatMessage({
        session_id: sessionId,
        message: userText,
        history: historyPayload,
      });

      const assistantTurn: ChatTurn = {
        id: `assistant_${Date.now()}`,
        role: 'assistant',
        content: res.message,
        sources: res.sources,
        latency_ms: res.latency_ms,
      };

      setMessages((prev) => [...prev, assistantTurn]);
      setTokenEstimate((prev) => Math.min(2000, prev + Math.ceil(res.message.length / 4)));
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err_${Date.now()}`,
          role: 'assistant',
          content: `⚠️ Failed to get response: ${err.message || 'Unknown error'}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const resetSession = () => {
    setSessionId(`session_${Date.now()}`);
    setMessages([
      {
        id: 'init',
        role: 'assistant',
        content:
          'Session reset. You are now in a fresh conversational session. What would you like to explore?',
      },
    ]);
    setTokenEstimate(25);
  };

  const budgetPercent = Math.round((tokenEstimate / 2000) * 100);

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '700px', overflow: 'hidden' }}>
      {/* Chat Header */}
      <div
        style={{
          padding: '16px 20px',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'var(--bg-surface)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--accent-primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
            }}
          >
            <Bot size={18} />
          </div>
          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 600 }}>Conversational Knowledge Agent</h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Session: <code>{sessionId.slice(0, 16)}...</code>
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          {/* Token Budget Meter */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '12px',
              background: 'var(--bg-surface-elevated)',
              padding: '6px 12px',
              borderRadius: 'var(--radius-full)',
              border: '1px solid var(--border-subtle)',
            }}
            title="Sliding window token budget tracking (2,000 max budget)"
          >
            <Gauge size={13} color="var(--accent-cyan)" />
            <span style={{ color: 'var(--text-secondary)' }}>Budget:</span>
            <span style={{ fontWeight: 600, color: budgetPercent > 80 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
              {tokenEstimate} / 2,000
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>({budgetPercent}%)</span>
          </div>

          <button onClick={resetSession} className="btn-ghost" title="Reset Session & History">
            <RotateCcw size={14} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
        }}
      >
        {messages.map((msg) => {
          const isUser = msg.role === 'user';
          return (
            <div
              key={msg.id}
              className="animate-fade-in"
              style={{
                display: 'flex',
                gap: '12px',
                alignSelf: isUser ? 'flex-end' : 'flex-start',
                maxWidth: '85%',
              }}
            >
              {!isUser && (
                <div
                  style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #6366f1, #06b6d4)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    flexShrink: 0,
                  }}
                >
                  <Bot size={16} />
                </div>
              )}

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div
                  style={{
                    padding: '14px 18px',
                    borderRadius: '16px',
                    borderTopLeftRadius: !isUser ? '4px' : '16px',
                    borderTopRightRadius: isUser ? '4px' : '16px',
                    background: isUser
                      ? 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)'
                      : 'var(--bg-surface-elevated)',
                    color: isUser ? '#ffffff' : 'var(--text-primary)',
                    border: !isUser ? '1px solid var(--border-medium)' : 'none',
                    boxShadow: isUser ? 'var(--shadow-glow)' : 'var(--shadow-sm)',
                    fontSize: '14px',
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {msg.content}
                </div>

                {/* Footnote metadata and citations */}
                {!isUser && msg.sources && msg.sources.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '4px' }}>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Sources consulted:</span>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      {msg.sources.map((s, idx) => (
                        <div key={idx} style={{ maxWidth: '240px' }}>
                          <CitationCard citation={s} index={idx} />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {msg.latency_ms && (
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)', alignSelf: 'flex-start' }}>
                    Generated in {msg.latency_ms}ms
                  </span>
                )}
              </div>

              {isUser && (
                <div
                  style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    background: 'var(--bg-surface-hover)',
                    border: '1px solid var(--border-medium)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--text-secondary)',
                    flexShrink: 0,
                  }}
                >
                  <User size={16} />
                </div>
              )}
            </div>
          );
        })}

        {loading && (
          <div className="animate-fade-in" style={{ display: 'flex', gap: '12px', alignSelf: 'flex-start' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #6366f1, #06b6d4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
              }}
            >
              <Sparkles size={16} className="animate-spin" />
            </div>
            <div
              style={{
                padding: '12px 18px',
                borderRadius: '16px',
                borderTopLeftRadius: '4px',
                background: 'var(--bg-surface-elevated)',
                border: '1px solid var(--border-subtle)',
                fontSize: '13px',
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <span>Rewriting query & retrieving evidence...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div
        style={{
          padding: '16px 20px',
          borderTop: '1px solid var(--border-subtle)',
          background: 'var(--bg-surface)',
        }}
      >
        <form onSubmit={handleSend} style={{ display: 'flex', gap: '10px' }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your follow-up or question (e.g., 'What about the video explanation?')..."
            style={{
              flex: 1,
              padding: '12px 16px',
              background: 'var(--bg-surface-elevated)',
              border: '1px solid var(--border-medium)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              fontSize: '14px',
              outline: 'none',
            }}
            disabled={loading}
          />

          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="btn-primary"
            style={{ padding: '0 20px' }}
          >
            <Send size={16} />
            <span>Send</span>
          </button>
        </form>
      </div>
    </div>
  );
}
