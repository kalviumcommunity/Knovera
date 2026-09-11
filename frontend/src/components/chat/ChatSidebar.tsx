'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Conversation, ConversationGroup } from '@/lib/types';
import {
  Plus,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Trash2,
  Check,
  X,
  PanelLeftClose,
  PanelLeft,
  ShieldCheck,
  Sparkles,
  Bot,
  Layers,
  ChevronRight,
  LogOut,
} from 'lucide-react';

interface ChatSidebarProps {
  conversations: Conversation[];
  activeConversationId: string;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  onDeleteConversation: (id: string) => void;
  isOpen: boolean;
  onToggleOpen: () => void;
  onSwitchToAdmin?: () => void;
  onLogout?: () => void;
  userName?: string;
  userEmail?: string;
}

export default function ChatSidebar({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onRenameConversation,
  onDeleteConversation,
  isOpen,
  onToggleOpen,
  onSwitchToAdmin,
  onLogout,
  userName = 'Sarah Jenkins',
  userEmail = 'user@knovera.ai',
}: ChatSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const editInputRef = useRef<HTMLInputElement>(null);

  // Close menus when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.conversation-action-menu') && !target.closest('.conversation-menu-trigger')) {
        setMenuOpenId(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingId]);

  const handleStartRename = (conv: Conversation) => {
    setEditingId(conv.id);
    setEditingTitle(conv.title);
    setMenuOpenId(null);
  };

  const handleSaveRename = (id: string) => {
    if (editingTitle.trim()) {
      onRenameConversation(id, editingTitle.trim());
    }
    setEditingId(null);
  };

  // Group conversations by Today, Yesterday, Previous 7 Days, Older
  const groupConversations = () => {
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const yesterdayStart = todayStart - 86400000;
    const sevenDaysAgo = todayStart - 7 * 86400000;

    const groups: Record<ConversationGroup, Conversation[]> = {
      Today: [],
      Yesterday: [],
      'Previous 7 Days': [],
      Older: [],
    };

    conversations.forEach((conv) => {
      const date = new Date(conv.updatedAt || conv.createdAt).getTime();
      if (date >= todayStart) {
        groups.Today.push(conv);
      } else if (date >= yesterdayStart) {
        groups.Yesterday.push(conv);
      } else if (date >= sevenDaysAgo) {
        groups['Previous 7 Days'].push(conv);
      } else {
        groups.Older.push(conv);
      }
    });

    return groups;
  };

  const grouped = groupConversations();
  const groupKeys: ConversationGroup[] = ['Today', 'Yesterday', 'Previous 7 Days', 'Older'];

  if (!isOpen) {
    return (
      <div
        style={{
          position: 'fixed',
          top: '14px',
          left: '14px',
          zIndex: 40,
        }}
      >
        <button
          onClick={onToggleOpen}
          title="Open sidebar"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '38px',
            height: '38px',
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-sm)',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-muted)';
            e.currentTarget.style.color = 'var(--text-primary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-surface)';
            e.currentTarget.style.color = 'var(--text-secondary)';
          }}
        >
          <PanelLeft size={18} />
        </button>
      </div>
    );
  }

  return (
    <aside
      style={{
        width: '260px',
        height: '100vh',
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-light)',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        position: 'relative',
        zIndex: 30,
      }}
    >
      {/* Top Header: Logo + Collapse Sidebar */}
      <div
        style={{
          padding: '16px 14px 12px 16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              width: '28px',
              height: '28px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'var(--accent-slate)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
            }}
          >
            <Sparkles size={16} />
          </div>
          <span
            style={{
              fontWeight: 600,
              fontSize: '15px',
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
            }}
          >
            Knovera AI
          </span>
          <span
            style={{
              fontSize: '10px',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              padding: '1px 5px',
              borderRadius: 'var(--radius-xs)',
              backgroundColor: 'var(--bg-muted)',
              color: 'var(--text-muted)',
              fontWeight: 600,
            }}
          >
            RAG
          </span>
        </div>

        <button
          onClick={onToggleOpen}
          title="Close sidebar"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '28px',
            height: '28px',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text-muted)',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-sidebar-hover)')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
        >
          <PanelLeftClose size={17} />
        </button>
      </div>

      {/* Prominent New Chat Button */}
      <div style={{ padding: '0 12px 10px 12px' }}>
        <button
          onClick={onNewChat}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 12px',
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-xs)',
            fontSize: '13.5px',
            fontWeight: 500,
            color: 'var(--text-primary)',
            cursor: 'pointer',
            transition: 'background 0.15s ease, border-color 0.15s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-sidebar-hover)';
            e.currentTarget.style.borderColor = 'var(--border-medium)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-surface)';
            e.currentTarget.style.borderColor = 'var(--border-light)';
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Plus size={16} color="var(--accent-primary)" />
            <span>New chat</span>
          </div>
          <kbd
            style={{
              fontSize: '10px',
              padding: '1px 5px',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--border-light)',
              color: 'var(--text-light)',
              fontFamily: 'inherit',
            }}
          >
            ⌘K
          </kbd>
        </button>
      </div>

      {/* Scrollable Conversation History */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '4px 10px',
        }}
      >
        {conversations.length === 0 ? (
          <div
            style={{
              padding: '32px 12px',
              textAlign: 'center',
              color: 'var(--text-muted)',
              fontSize: '13px',
            }}
          >
            No conversations yet. Start a new chat above!
          </div>
        ) : (
          groupKeys.map((group) => {
            const list = grouped[group];
            if (list.length === 0) return null;

            return (
              <div key={group} style={{ marginBottom: '14px' }}>
                <div
                  style={{
                    fontSize: '11px',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                    color: 'var(--text-light)',
                    padding: '4px 8px 6px 8px',
                  }}
                >
                  {group}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  {list.map((conv) => {
                    const isActive = conv.id === activeConversationId;
                    const isEditing = conv.id === editingId;
                    const isMenuOpen = conv.id === menuOpenId;

                    return (
                      <div
                        key={conv.id}
                        style={{
                          position: 'relative',
                          display: 'flex',
                          alignItems: 'center',
                          borderRadius: 'var(--radius-md)',
                          backgroundColor: isActive ? 'var(--bg-sidebar-active)' : 'transparent',
                          transition: 'background 0.12s ease',
                        }}
                        onMouseEnter={(e) => {
                          if (!isActive) e.currentTarget.style.backgroundColor = 'var(--bg-sidebar-hover)';
                          const btn = e.currentTarget.querySelector('.three-dot-btn') as HTMLElement;
                          if (btn && !isMenuOpen) btn.style.opacity = '1';
                        }}
                        onMouseLeave={(e) => {
                          if (!isActive) e.currentTarget.style.backgroundColor = 'transparent';
                          const btn = e.currentTarget.querySelector('.three-dot-btn') as HTMLElement;
                          if (btn && !isMenuOpen) btn.style.opacity = '0';
                        }}
                      >
                        {isEditing ? (
                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px',
                              width: '100%',
                              padding: '4px 6px',
                            }}
                          >
                            <input
                              ref={editInputRef}
                              value={editingTitle}
                              onChange={(e) => setEditingTitle(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') handleSaveRename(conv.id);
                                if (e.key === 'Escape') setEditingId(null);
                              }}
                              style={{
                                flex: 1,
                                padding: '4px 8px',
                                fontSize: '13px',
                                border: '1px solid var(--border-focus)',
                                borderRadius: 'var(--radius-xs)',
                                backgroundColor: '#ffffff',
                                color: 'var(--text-primary)',
                              }}
                            />
                            <button
                              onClick={() => handleSaveRename(conv.id)}
                              style={{ padding: '4px', color: 'var(--status-success)' }}
                              title="Save title"
                            >
                              <Check size={14} />
                            </button>
                            <button
                              onClick={() => setEditingId(null)}
                              style={{ padding: '4px', color: 'var(--text-muted)' }}
                              title="Cancel"
                            >
                              <X size={14} />
                            </button>
                          </div>
                        ) : (
                          <>
                            <button
                              onClick={() => onSelectConversation(conv.id)}
                              style={{
                                flex: 1,
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                padding: '7px 8px',
                                textAlign: 'left',
                                fontSize: '13px',
                                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                                fontWeight: isActive ? 500 : 400,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                              }}
                            >
                              <MessageSquare
                                size={14}
                                color={isActive ? 'var(--accent-primary)' : 'var(--text-muted)'}
                                style={{ flexShrink: 0 }}
                              />
                              <span
                                style={{
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                }}
                              >
                                {conv.title}
                              </span>
                            </button>

                            {/* Three-dot action button */}
                            <button
                              className="three-dot-btn conversation-menu-trigger"
                              onClick={(e) => {
                                e.stopPropagation();
                                setMenuOpenId(isMenuOpen ? null : conv.id);
                              }}
                              style={{
                                opacity: isMenuOpen ? 1 : 0,
                                padding: '6px',
                                marginRight: '4px',
                                borderRadius: 'var(--radius-xs)',
                                color: 'var(--text-muted)',
                                transition: 'opacity 0.12s ease',
                              }}
                              title="Conversation actions"
                            >
                              <MoreHorizontal size={14} />
                            </button>

                            {/* Action Menu Popover */}
                            {isMenuOpen && (
                              <div
                                className="conversation-action-menu animate-slide-down"
                                style={{
                                  position: 'absolute',
                                  right: '6px',
                                  top: '100%',
                                  marginTop: '2px',
                                  width: '130px',
                                  backgroundColor: 'var(--bg-surface)',
                                  border: '1px solid var(--border-light)',
                                  borderRadius: 'var(--radius-md)',
                                  boxShadow: 'var(--shadow-md)',
                                  zIndex: 50,
                                  overflow: 'hidden',
                                  padding: '4px',
                                }}
                              >
                                <button
                                  onClick={() => handleStartRename(conv)}
                                  style={{
                                    width: '100%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    padding: '6px 8px',
                                    fontSize: '12.5px',
                                    borderRadius: 'var(--radius-xs)',
                                    color: 'var(--text-primary)',
                                    textAlign: 'left',
                                  }}
                                  onMouseEnter={(e) =>
                                    (e.currentTarget.style.backgroundColor = 'var(--bg-muted)')
                                  }
                                  onMouseLeave={(e) =>
                                    (e.currentTarget.style.backgroundColor = 'transparent')
                                  }
                                >
                                  <Pencil size={13} color="var(--text-muted)" />
                                  <span>Rename</span>
                                </button>
                                <button
                                  onClick={() => {
                                    onDeleteConversation(conv.id);
                                    setMenuOpenId(null);
                                  }}
                                  style={{
                                    width: '100%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    padding: '6px 8px',
                                    fontSize: '12.5px',
                                    borderRadius: 'var(--radius-xs)',
                                    color: 'var(--status-danger)',
                                    textAlign: 'left',
                                  }}
                                  onMouseEnter={(e) =>
                                    (e.currentTarget.style.backgroundColor = 'var(--status-danger-bg)')
                                  }
                                  onMouseLeave={(e) =>
                                    (e.currentTarget.style.backgroundColor = 'transparent')
                                  }
                                >
                                  <Trash2 size={13} />
                                  <span>Delete</span>
                                </button>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Bottom Footer: User Profile & Session */}
      <div
        style={{
          padding: '12px 14px',
          borderTop: '1px solid var(--border-light)',
          backgroundColor: 'var(--bg-surface)',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '8px',
            padding: '4px 4px 0 4px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0, flex: 1 }}>
            <div
              style={{
                width: '28px',
                height: '28px',
                borderRadius: 'var(--radius-full)',
                backgroundColor: 'var(--bg-muted)',
                border: '1px solid var(--border-light)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '11px',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                flexShrink: 0,
              }}
            >
              {userName
                .split(' ')
                .map((n) => n[0])
                .join('')
                .slice(0, 2)
                .toUpperCase() || 'U'}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {userName}
              </div>
              <div
                style={{
                  fontSize: '10.5px',
                  color: 'var(--text-muted)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {userEmail}
              </div>
            </div>
          </div>

          {onLogout && (
            <button
              onClick={onLogout}
              title="Sign out of Knovera AI"
              style={{
                padding: '5px',
                borderRadius: 'var(--radius-xs)',
                color: 'var(--text-muted)',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--status-danger-bg)';
                e.currentTarget.style.color = 'var(--status-danger)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.color = 'var(--text-muted)';
              }}
            >
              <LogOut size={14} />
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}
