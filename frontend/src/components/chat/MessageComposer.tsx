'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  ArrowUp,
  Paperclip,
  Mic,
  MicOff,
  X,
  FileText,
  Sparkles,
} from 'lucide-react';

interface MessageComposerProps {
  onSendMessage: (text: string, attachedFile?: { name: string; size: number }) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function MessageComposer({
  onSendMessage,
  disabled = false,
  placeholder = 'Message Knovera AI...',
}: MessageComposerProps) {
  const [text, setText] = useState('');
  const [attachedFile, setAttachedFile] = useState<{ name: string; size: number } | null>(null);
  const [isListening, setIsListening] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const newHeight = Math.min(textareaRef.current.scrollHeight, 180);
      textareaRef.current.style.height = `${Math.max(48, newHeight)}px`;
    }
  }, [text]);

  const handleSend = () => {
    if ((!text.trim() && !attachedFile) || disabled) return;
    onSendMessage(text.trim(), attachedFile || undefined);
    setText('');
    setAttachedFile(null);
    if (textareaRef.current) {
      textareaRef.current.style.height = '48px';
      textareaRef.current.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAttachedFile({
        name: file.name,
        size: file.size,
      });
      // reset input value so re-selecting same file triggers change
      e.target.value = '';
    }
  };

  const toggleListening = () => {
    if (isListening) {
      setIsListening(false);
    } else {
      setIsListening(true);
      // Simulate speech-to-text transcript after brief pause
      setTimeout(() => {
        setText((prev) => (prev ? `${prev} Summarize the retrieval metrics.` : 'Summarize the retrieval metrics.'));
        setIsListening(false);
      }, 2400);
    }
  };

  const canSend = (text.trim().length > 0 || attachedFile !== null) && !disabled;

  return (
    <div
      style={{
        position: 'sticky',
        bottom: 0,
        width: '100%',
        maxWidth: '820px',
        margin: '0 auto',
        padding: '0 16px 20px 16px',
        backgroundColor: 'var(--bg-surface)',
        zIndex: 10,
      }}
    >
      {/* File preview badge if attached */}
      {attachedFile && (
        <div
          className="animate-fade-in"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            marginBottom: '8px',
            backgroundColor: 'var(--bg-muted)',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-md)',
            fontSize: '13px',
            color: 'var(--text-secondary)',
          }}
        >
          <FileText size={15} color="var(--accent-primary)" />
          <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{attachedFile.name}</span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            ({(attachedFile.size / 1024).toFixed(1)} KB)
          </span>
          <button
            onClick={() => setAttachedFile(null)}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '2px',
              borderRadius: 'var(--radius-full)',
              color: 'var(--text-muted)',
              cursor: 'pointer',
            }}
            title="Remove attachment"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Listening status banner */}
      {isListening && (
        <div
          className="animate-fade-in"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            marginBottom: '8px',
            backgroundColor: '#eff6ff',
            border: '1px solid #bfdbfe',
            borderRadius: 'var(--radius-md)',
            fontSize: '12.5px',
            color: 'var(--accent-primary)',
          }}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#ef4444',
              display: 'inline-block',
              animation: 'pulseDot 1s infinite',
            }}
          />
          <span>Listening to speech input... Speak now</span>
        </div>
      )}

      {/* Main Composer Box */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-light)',
          borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-composer)',
          transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
          padding: '8px 12px 8px 14px',
        }}
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          style={{
            width: '100%',
            border: 'none',
            outline: 'none',
            resize: 'none',
            backgroundColor: 'transparent',
            fontFamily: 'inherit',
            fontSize: '15px',
            lineHeight: 1.5,
            color: 'var(--text-primary)',
            padding: '4px 0',
            maxHeight: '180px',
          }}
        />

        {/* Action controls row */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginTop: '4px',
            paddingTop: '4px',
          }}
        >
          {/* Left tools: Attach file, Mic voice */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              style={{ display: 'none' }}
              accept=".pdf,.docx,.txt,.csv,.md,.json"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              title="Attach document or source (PDF, DOCX, TXT, CSV)"
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '32px',
                height: '32px',
                borderRadius: 'var(--radius-full)',
                color: 'var(--text-muted)',
                transition: 'all 0.15s ease',
                backgroundColor: attachedFile ? 'var(--bg-muted)' : 'transparent',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-muted)')}
              onMouseLeave={(e) =>
                (e.currentTarget.style.backgroundColor = attachedFile ? 'var(--bg-muted)' : 'transparent')
              }
            >
              <Paperclip size={17} />
            </button>

            <button
              type="button"
              onClick={toggleListening}
              disabled={disabled}
              title={isListening ? 'Stop listening' : 'Voice input'}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '32px',
                height: '32px',
                borderRadius: 'var(--radius-full)',
                color: isListening ? '#ef4444' : 'var(--text-muted)',
                backgroundColor: isListening ? '#fee2e2' : 'transparent',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                if (!isListening) e.currentTarget.style.backgroundColor = 'var(--bg-muted)';
              }}
              onMouseLeave={(e) => {
                if (!isListening) e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              {isListening ? <MicOff size={17} /> : <Mic size={17} />}
            </button>
          </div>

          {/* Right tool: Send button */}
          <button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            title="Send message (Enter)"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '34px',
              height: '34px',
              borderRadius: 'var(--radius-full)',
              backgroundColor: canSend ? 'var(--accent-slate)' : 'var(--bg-muted)',
              color: canSend ? '#ffffff' : 'var(--text-light)',
              cursor: canSend ? 'pointer' : 'not-allowed',
              transition: 'all 0.18s ease',
            }}
            onMouseEnter={(e) => {
              if (canSend) e.currentTarget.style.backgroundColor = 'var(--accent-slate-hover)';
            }}
            onMouseLeave={(e) => {
              if (canSend) e.currentTarget.style.backgroundColor = 'var(--accent-slate)';
            }}
          >
            <ArrowUp size={18} strokeWidth={2.4} />
          </button>
        </div>
      </div>

      {/* Subtle Disclaimer Notice */}
      <div
        style={{
          textAlign: 'center',
          marginTop: '8px',
          fontSize: '11.5px',
          color: 'var(--text-light)',
          letterSpacing: '-0.01em',
        }}
      >
        Knovera AI answers questions based on indexed enterprise sources. Verify critical facts.
      </div>
    </div>
  );
}
