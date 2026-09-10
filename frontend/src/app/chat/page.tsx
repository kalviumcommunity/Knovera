'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredAuth, logout, AuthUser, DEMO_CREDENTIALS } from '@/lib/auth';
import { Conversation, ChatTurn } from '@/lib/types';
import { api } from '@/lib/api';

import ChatSidebar from '@/components/chat/ChatSidebar';
import ChatArea from '@/components/chat/ChatArea';

export default function UserChatPage() {
  const router = useRouter();
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);

  // Check user session
  useEffect(() => {
    const auth = getStoredAuth();
    if (!auth) {
      // Default to standard demo user session
      const guest: AuthUser = {
        id: 'usr_guest',
        name: DEMO_CREDENTIALS.user.name,
        email: DEMO_CREDENTIALS.user.email,
        role: 'user',
        organization: DEMO_CREDENTIALS.user.organization,
        avatarInitials: DEMO_CREDENTIALS.user.avatarInitials,
      };
      setCurrentUser(guest);
    } else {
      setCurrentUser(auth);
    }
  }, []);

  // Fetch persistent conversations scoped to this specific user from MongoDB
  const loadConversationsFromDb = useCallback(async () => {
    if (!currentUser) return;
    setIsInitializing(true);
    const userKey = currentUser.email || currentUser.id;

    try {
      const dbConversations = await api.getConversations(userKey);
      if (dbConversations && dbConversations.length > 0) {
        setConversations(dbConversations);
        setActiveConvId(dbConversations[0].id);
      } else {
        // Create initial clean conversation in MongoDB for this new user
        const initial = await api.createConversation('New chat', undefined, userKey);
        setConversations([initial]);
        setActiveConvId(initial.id);
      }
    } catch (err) {
      console.error('Failed to load conversations from backend MongoDB:', err);
      const fallbackId = `conv_${Date.now()}`;
      const fallbackConv: Conversation = {
        id: fallbackId,
        title: 'New chat',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      setConversations([fallbackConv]);
      setActiveConvId(fallbackId);
    } finally {
      setIsInitializing(false);
    }
  }, [currentUser]);

  useEffect(() => {
    if (currentUser) {
      loadConversationsFromDb();
    }
  }, [currentUser, loadConversationsFromDb]);

  // Keyboard shortcut Ctrl+K / Cmd+K for New Chat
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        handleNewChat();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentUser]);

  const activeConversation = conversations.find((c) => c.id === activeConvId);

  // Create New Chat in MongoDB
  const handleNewChat = async () => {
    const userKey = currentUser?.email || currentUser?.id || 'user@knovera.ai';
    try {
      const newConv = await api.createConversation('New chat', undefined, userKey);
      setConversations((prev) => [newConv, ...prev]);
      setActiveConvId(newConv.id);
    } catch (err) {
      console.error('Failed to create new conversation in MongoDB:', err);
      const localId = `conv_${Date.now()}`;
      const localConv: Conversation = {
        id: localId,
        title: 'New chat',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      setConversations((prev) => [localConv, ...prev]);
      setActiveConvId(localId);
    }
  };

  // Rename Conversation in MongoDB
  const handleRenameConversation = async (id: string, newTitle: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title: newTitle, updatedAt: new Date().toISOString() } : c))
    );

    try {
      await api.renameConversation(id, newTitle);
    } catch (err) {
      console.error('Failed to persist conversation rename:', err);
    }
  };

  // Delete Conversation from MongoDB
  const handleDeleteConversation = async (id: string) => {
    const filtered = conversations.filter((c) => c.id !== id);
    setConversations(filtered);
    if (activeConvId === id) {
      if (filtered.length > 0) {
        setActiveConvId(filtered[0].id);
      } else {
        handleNewChat();
      }
    }

    try {
      await api.deleteConversation(id);
    } catch (err) {
      console.error('Failed to delete conversation from MongoDB:', err);
    }
  };

  // Send message and persist turns to MongoDB
  const handleSendMessage = async (
    text: string,
    attachedFile?: { name: string; size: number }
  ) => {
    if (!text && !attachedFile) return;

    let convId = activeConvId;
    let targetConv = conversations.find((c) => c.id === convId);
    const userKey = currentUser?.email || currentUser?.id || 'user@knovera.ai';

    if (!targetConv) {
      try {
        const created = await api.createConversation(text ? text.slice(0, 32) : 'New chat', undefined, userKey);
        targetConv = created;
        convId = created.id;
        setConversations((prev) => [created, ...prev]);
        setActiveConvId(convId);
      } catch {
        convId = `conv_${Date.now()}`;
        targetConv = {
          id: convId,
          title: text ? text.slice(0, 32) : 'New chat',
          messages: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        setConversations((prev) => [targetConv!, ...prev]);
        setActiveConvId(convId);
      }
    }

    // Auto-rename from "New chat" to initial query snippet
    if (targetConv.title === 'New chat' && text) {
      const autoTitle = text.length > 36 ? `${text.slice(0, 33)}...` : text;
      handleRenameConversation(convId, autoTitle);
    }

    const userMessageContent = attachedFile
      ? `${text ? `${text}\n\n` : ''}[Attached: ${attachedFile.name} (${(attachedFile.size / 1024).toFixed(1)} KB)]`
      : text;

    const userTurn: ChatTurn = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: userMessageContent,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Update UI immediately
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: [...c.messages, userTurn],
              updatedAt: new Date().toISOString(),
            }
          : c
      )
    );

    // Save user message to database asynchronously
    api.saveChatMessage(convId, userTurn).catch((err) =>
      console.warn('Failed to save user message turn to MongoDB:', err)
    );

    setIsLoading(true);

    try {
      // Build conversation history for API payload
      const historyPayload = (targetConv.messages || []).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Submit chat query with dynamic backend thresholding
      const res = await api.sendChatMessage({
        session_id: convId,
        message: text,
        history: historyPayload,
        k: 4,
      });

      const assistantTurn: ChatTurn = {
        id: `asst_${Date.now()}`,
        role: 'assistant',
        content: res.message,
        sources: res.sources,
        latency_ms: res.latency_ms || 240,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      // Save assistant message to database asynchronously
      api.saveChatMessage(convId, assistantTurn).catch((err) =>
        console.warn('Failed to save assistant turn to MongoDB:', err)
      );

      // Update state
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: [...c.messages, assistantTurn],
                updatedAt: new Date().toISOString(),
              }
            : c
        )
      );
    } catch (err: any) {
      console.error('Chat API request error:', err);
      const errorTurn: ChatTurn = {
        id: `err_${Date.now()}`,
        role: 'assistant',
        content: `⚠️ Failed to generate response: ${err.message || 'Cannot reach server'}. Please ensure FastAPI backend is running.`,
        latency_ms: 100,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setConversations((prev) =>
        prev.map((c) => (c.id === convId ? { ...c, messages: [...c.messages, errorTurn] } : c))
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = (turnId: string, type: 'up' | 'down') => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeConvId
          ? {
              ...c,
              messages: c.messages.map((m) =>
                m.id === turnId ? { ...m, feedback: m.feedback === type ? undefined : type } : m
              ),
            }
          : c
      )
    );
  };

  const handleClearHistory = () => {
    setConversations([]);
    handleNewChat();
  };

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        backgroundColor: '#ffffff',
      }}
    >
      {/* ChatGPT-Style Sidebar */}
      <ChatSidebar
        conversations={conversations}
        activeConversationId={activeConvId}
        onSelectConversation={(id) => setActiveConvId(id)}
        onNewChat={handleNewChat}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={handleDeleteConversation}
        isOpen={sidebarOpen}
        onToggleOpen={() => setSidebarOpen(!sidebarOpen)}
        onSwitchToAdmin={() => router.push('/admin')}
        onLogout={handleLogout}
        userName={currentUser?.name || DEMO_CREDENTIALS.user.name}
        userEmail={currentUser?.email || DEMO_CREDENTIALS.user.email}
      />

      {/* Main Chat Workspace Area */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflow: 'hidden',
          backgroundColor: '#ffffff',
          position: 'relative',
        }}
      >
        <ChatArea
          messages={activeConversation?.messages || []}
          isLoading={isLoading}
          onSendMessage={handleSendMessage}
          onFeedback={handleFeedback}
        />
      </div>
    </div>
  );
}
