'use client';

import React, { useRef, useEffect, useState } from 'react';
import { ChatTurn, SourceCitation } from '@/lib/types';
import MarkdownRenderer from './MarkdownRenderer';
import MessageComposer from './MessageComposer';
import {
  Bot,
  User,
  Sparkles,
  Copy,
  Check,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  BookOpen,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  ShieldAlert,
  HelpCircle,
  FileSearch,
  Database,
  Lock,
} from 'lucide-react';

interface ChatAreaProps {
  messages: ChatTurn[];
  isLoading: boolean;
  onSendMessage: (text: string, attachedFile?: { name: string; size: number }) => void;
  onRegenerateLast?: () => void;
  onFeedback?: (turnId: string, type: 'up' | 'down') => void;
}

export default function ChatArea({
  messages,
  isLoading,
  onSendMessage,
  onRegenerateLast,
  onFeedback,
}: ChatAreaProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [openCitationsTurnId, setOpenCitationsTurnId] = useState<string | null>(null);

  // Auto scroll to bottom smoothly
  const scrollToBottom = () => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const sampleSuggestions = [
    {
      title: 'Analyze RAG retrieval metrics',
      desc: 'Evaluate top-k recall, precision, and reciprocal rank guidelines',
      icon: <FileSearch size={16} color="var(--accent-primary)" />,
      query: 'What are the target evaluation metrics for Knovera RAG retrieval, and how is reciprocal rank calculated?',
    },
    {
      title: 'Summarize safety guardrails',
      desc: 'Review hallucination prevention, prompt injection, and PII masking',
      icon: <ShieldAlert size={16} color="#7c3aed" />,
      query: 'Summarize the active AI safety guardrails and explain how hallucination score verification works.',
    },
    {
      title: 'Document chunking strategy',
      desc: 'Understand recursive character and semantic token boundaries',
      icon: <Database size={16} color="#059669" />,
      query: 'Explain Knovera chunking strategies for multi-page PDFs and CSV documents.',
    },
    {
      title: 'API integration guide',
      desc: 'Provide a sample Python and TypeScript client request snippet',
      icon: <HelpCircle size={16} color="#d97706" />,
      query: 'Show me an example of querying the Knovera FastAPI `/api/query` and `/api/chat` endpoints in TypeScript.',
    },
  ];

  return (
    <div
      style={{
        flex: 1,
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        backgroundColor: 'var(--bg-surface)',
        overflow: 'hidden',
      }}
    >
      {/* Scrollable conversation stream */}
      <div
        ref={scrollContainerRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '24px 20px 40px 20px',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: '800px',
            margin: '0 auto',
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Empty State when no messages */}
          {messages.length === 0 ? (
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                textAlign: 'center',
                padding: '40px 16px',
              }}
            >
              <div
                style={{
                  width: '56px',
                  height: '56px',
                  borderRadius: 'var(--radius-lg)',
                  backgroundColor: 'var(--bg-muted)',
                  border: '1px solid var(--border-light)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '18px',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <Sparkles size={28} color="var(--accent-primary)" />
              </div>

              <h2
                style={{
                  fontSize: '22px',
                  fontWeight: 600,
                  letterSpacing: '-0.02em',
                  color: 'var(--text-primary)',
                  marginBottom: '8px',
                }}
              >
                Where knowledge meets precision
              </h2>
              <p
                style={{
                  fontSize: '14.5px',
                  color: 'var(--text-secondary)',
                  maxWidth: '480px',
                  lineHeight: 1.5,
                  marginBottom: '36px',
                }}
              >
                Ask questions about your uploaded documents, contracts, internal wikis, or API
                specifications with fully verifiable citations.
              </p>

              {/* Suggestion Cards */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                  gap: '12px',
                  width: '100%',
                  maxWidth: '680px',
                }}
              >
                {sampleSuggestions.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => onSendMessage(item.query)}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '12px',
                      padding: '14px 16px',
                      textAlign: 'left',
                      backgroundColor: 'var(--bg-surface)',
                      border: '1px solid var(--border-light)',
                      borderRadius: 'var(--radius-lg)',
                      boxShadow: 'var(--shadow-xs)',
                      cursor: 'pointer',
                      transition: 'border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = 'var(--border-medium)';
                      e.currentTarget.style.transform = 'translateY(-1px)';
                      e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = 'var(--border-light)';
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'var(--shadow-xs)';
                    }}
                  >
                    <div
                      style={{
                        padding: '8px',
                        backgroundColor: 'var(--bg-muted)',
                        borderRadius: 'var(--radius-sm)',
                        flexShrink: 0,
                      }}
                    >
                      {item.icon}
                    </div>
                    <div>
                      <div style={{ fontWeight: 500, fontSize: '13.5px', color: 'var(--text-primary)' }}>
                        {item.title}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        {item.desc}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Message turns list */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {messages.map((msg, index) => {
                const isUser = msg.role === 'user';
                const hasCitations = msg.sources && msg.sources.length > 0;
                const isCitationsOpen = openCitationsTurnId === msg.id;

                return (
                  <div
                    key={msg.id || index}
                    className="animate-fade-in"
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: isUser ? 'flex-end' : 'flex-start',
                      width: '100%',
                    }}
                  >
                    {/* Message Bubble Container */}
                    <div
                      style={{
                        display: 'flex',
                        gap: '12px',
                        maxWidth: isUser ? '85%' : '100%',
                        width: isUser ? 'auto' : '100%',
                      }}
                    >
                      {/* Assistant Avatar */}
                      {!isUser && (
                        <div
                          style={{
                            width: '32px',
                            height: '32px',
                            borderRadius: 'var(--radius-md)',
                            backgroundColor: 'var(--accent-slate)',
                            color: '#ffffff',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                            marginTop: '2px',
                          }}
                        >
                          <Sparkles size={16} />
                        </div>
                      )}

                      {/* Content Box */}
                      <div
                        style={{
                          flex: 1,
                          minWidth: 0,
                          backgroundColor: isUser ? 'var(--bg-muted)' : 'transparent',
                          border: isUser ? '1px solid var(--border-light)' : 'none',
                          borderRadius: isUser ? '18px 18px 4px 18px' : '0',
                          padding: isUser ? '10px 16px' : '0',
                          color: 'var(--text-primary)',
                        }}
                      >
                        {isUser ? (
                          <div
                            style={{
                              fontSize: '14.5px',
                              lineHeight: 1.55,
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word',
                            }}
                          >
                            {msg.content}
                          </div>
                        ) : (
                          <div style={{ width: '100%' }}>
                            <MarkdownRenderer content={msg.content} />

                            {/* Citations Drawer (if RAG sources present) */}
                            {hasCitations && (
                              <div
                                style={{
                                  marginTop: '14px',
                                  border: '1px solid var(--border-light)',
                                  borderRadius: 'var(--radius-md)',
                                  backgroundColor: 'var(--bg-muted)',
                                  overflow: 'hidden',
                                }}
                              >
                                <button
                                  onClick={() =>
                                    setOpenCitationsTurnId(isCitationsOpen ? null : msg.id)
                                  }
                                  style={{
                                    width: '100%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    padding: '8px 12px',
                                    fontSize: '12px',
                                    fontWeight: 500,
                                    color: 'var(--text-secondary)',
                                    cursor: 'pointer',
                                  }}
                                >
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    <BookOpen size={14} color="var(--accent-primary)" />
                                    <span>
                                      Grounded in {msg.sources!.length} verified source
                                      {msg.sources!.length > 1 ? 's' : ''}
                                    </span>
                                  </div>
                                  {isCitationsOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                </button>

                                {isCitationsOpen && (
                                  <div
                                    style={{
                                      padding: '8px 12px 12px 12px',
                                      borderTop: '1px solid var(--border-light)',
                                      display: 'flex',
                                      flexDirection: 'column',
                                      gap: '8px',
                                    }}
                                  >
                                    {msg.sources!.map((s, sIdx) => (
                                      <div
                                        key={sIdx}
                                        style={{
                                          padding: '8px 10px',
                                          backgroundColor: '#ffffff',
                                          border: '1px solid var(--border-light)',
                                          borderRadius: 'var(--radius-sm)',
                                          fontSize: '12px',
                                        }}
                                      >
                                        <div
                                          style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            fontWeight: 500,
                                            color: 'var(--text-primary)',
                                          }}
                                        >
                                          <span>{s.source}</span>
                                          {s.score !== undefined && (
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
                                              {(s.score * 100).toFixed(0)}% Match
                                            </span>
                                          )}
                                        </div>
                                        {s.section && (
                                          <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>
                                            Section: {s.section}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Assistant message action buttons */}
                            <div
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px',
                                marginTop: '10px',
                                color: 'var(--text-muted)',
                              }}
                            >
                              <button
                                onClick={() => handleCopyMessage(msg.id, msg.content)}
                                title="Copy response"
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '4px',
                                  padding: '4px 6px',
                                  borderRadius: 'var(--radius-xs)',
                                  fontSize: '11.5px',
                                  color: 'var(--text-muted)',
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.backgroundColor = 'var(--bg-muted)';
                                  e.currentTarget.style.color = 'var(--text-primary)';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.backgroundColor = 'transparent';
                                  e.currentTarget.style.color = 'var(--text-muted)';
                                }}
                              >
                                {copiedId === msg.id ? (
                                  <>
                                    <Check size={13} color="var(--status-success)" />
                                    <span style={{ color: 'var(--status-success)' }}>Copied</span>
                                  </>
                                ) : (
                                  <>
                                    <Copy size={13} />
                                    <span>Copy</span>
                                  </>
                                )}
                              </button>

                              <button
                                onClick={() => onFeedback?.(msg.id, 'up')}
                                title="Good response"
                                style={{
                                  padding: '4px 6px',
                                  borderRadius: 'var(--radius-xs)',
                                  color: msg.feedback === 'up' ? 'var(--status-success)' : 'inherit',
                                }}
                                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-muted)')}
                                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                              >
                                <ThumbsUp size={13} />
                              </button>

                              <button
                                onClick={() => onFeedback?.(msg.id, 'down')}
                                title="Bad response"
                                style={{
                                  padding: '4px 6px',
                                  borderRadius: 'var(--radius-xs)',
                                  color: msg.feedback === 'down' ? 'var(--status-danger)' : 'inherit',
                                }}
                                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-muted)')}
                                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                              >
                                <ThumbsDown size={13} />
                              </button>

                              {index === messages.length - 1 && onRegenerateLast && (
                                <button
                                  onClick={onRegenerateLast}
                                  title="Regenerate response"
                                  style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                    padding: '4px 6px',
                                    borderRadius: 'var(--radius-xs)',
                                    fontSize: '11.5px',
                                    marginLeft: '4px',
                                  }}
                                  onMouseEnter={(e) => {
                                    e.currentTarget.style.backgroundColor = 'var(--bg-muted)';
                                    e.currentTarget.style.color = 'var(--text-primary)';
                                  }}
                                  onMouseLeave={(e) => {
                                    e.currentTarget.style.backgroundColor = 'transparent';
                                    e.currentTarget.style.color = 'var(--text-muted)';
                                  }}
                                >
                                  <RotateCcw size={13} />
                                  <span>Regenerate</span>
                                </button>
                              )}

                              {msg.latency_ms && (
                                <span
                                  style={{
                                    fontSize: '11px',
                                    color: 'var(--text-light)',
                                    marginLeft: 'auto',
                                  }}
                                >
                                  {(msg.latency_ms / 1000).toFixed(2)}s
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Loading / Generating Indicator */}
              {isLoading && (
                <div
                  className="animate-fade-in"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    width: '100%',
                  }}
                >
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: 'var(--accent-slate)',
                      color: '#ffffff',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <Sparkles size={16} />
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '5px',
                      padding: '8px 12px',
                      backgroundColor: 'var(--bg-muted)',
                      borderRadius: 'var(--radius-lg)',
                      fontSize: '13px',
                      color: 'var(--text-secondary)',
                    }}
                  >
                    <span>Synthesizing answer from grounded sources</span>
                    <span style={{ display: 'inline-flex', gap: '3px', marginLeft: '4px' }}>
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Modern Fixed Bottom Message Composer */}
      <MessageComposer onSendMessage={onSendMessage} disabled={isLoading} />
    </div>
  );
}
