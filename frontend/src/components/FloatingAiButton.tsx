'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { getStoredAuth } from '@/lib/auth';
import { Sparkles } from 'lucide-react';

interface Position {
  x: number;
  y: number;
}

const STORAGE_KEY = 'knovera_floating_ai_pos';
const DRAG_THRESHOLD = 5; // Pixels moved before initiating drag vs click
const BUTTON_SIZE = 58; // Circle diameter in px

export default function FloatingAiButton() {
  const router = useRouter();
  const pathname = usePathname();
  const [isHovered, setIsHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [position, setPosition] = useState<Position | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const dragStartRef = useRef<{
    startX: number;
    startY: number;
    elemX: number;
    elemY: number;
    hasMoved: boolean;
  }>({
    startX: 0,
    startY: 0,
    elemX: 0,
    elemY: 0,
    hasMoved: false,
  });

  // Calculate default or saved position on client mount
  useEffect(() => {
    setMounted(true);
    let initialX = Math.max(16, window.innerWidth - BUTTON_SIZE - 24);
    let initialY = Math.max(16, window.innerHeight - BUTTON_SIZE - 24);

    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (typeof parsed.x === 'number' && typeof parsed.y === 'number') {
          // Clamp safely inside current viewport
          initialX = Math.max(12, Math.min(window.innerWidth - BUTTON_SIZE - 12, parsed.x));
          initialY = Math.max(12, Math.min(window.innerHeight - BUTTON_SIZE - 12, parsed.y));
        }
      }
    } catch {
      // Ignore storage errors
    }

    setPosition({ x: initialX, y: initialY });
  }, []);

  // Keep button within window bounds on browser resize
  useEffect(() => {
    const handleResize = () => {
      setPosition((prev) => {
        if (!prev) return prev;
        return {
          x: Math.max(12, Math.min(window.innerWidth - BUTTON_SIZE - 12, prev.x)),
          y: Math.max(12, Math.min(window.innerHeight - BUTTON_SIZE - 12, prev.y)),
        };
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Don't show on the main chat page since user is already in the full chatbot interface
  if (!mounted || pathname === '/chat' || !position) {
    return null;
  }

  const handleClick = () => {
    const auth = getStoredAuth();
    if (auth && auth.role) {
      router.push('/chat');
    } else {
      router.push('/login');
    }
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Only primary mouse button or touch
    if (e.button !== 0) return;

    dragStartRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      elemX: position.x,
      elemY: position.y,
      hasMoved: false,
    };

    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;

    const dx = e.clientX - dragStartRef.current.startX;
    const dy = e.clientY - dragStartRef.current.startY;

    if (!dragStartRef.current.hasMoved && Math.hypot(dx, dy) >= DRAG_THRESHOLD) {
      dragStartRef.current.hasMoved = true;
      setIsDragging(true);
    }

    if (dragStartRef.current.hasMoved) {
      const rawX = dragStartRef.current.elemX + dx;
      const rawY = dragStartRef.current.elemY + dy;

      const clampedX = Math.max(10, Math.min(window.innerWidth - BUTTON_SIZE - 10, rawX));
      const clampedY = Math.max(10, Math.min(window.innerHeight - BUTTON_SIZE - 10, rawY));

      setPosition({ x: clampedX, y: clampedY });
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // Ignore if pointer capture already released
    }

    if (dragStartRef.current.hasMoved) {
      setIsDragging(false);
      // Persist new position
      try {
        const finalX = Math.max(10, Math.min(window.innerWidth - BUTTON_SIZE - 10, position.x));
        const finalY = Math.max(10, Math.min(window.innerHeight - BUTTON_SIZE - 10, position.y));
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ x: finalX, y: finalY }));
      } catch {
        // Ignore storage errors
      }
      return;
    }

    // Normal click without movement
    setIsDragging(false);
    handleClick();
  };

  const handlePointerCancel = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // Ignore
      }
    }
    setIsDragging(false);
  };

  // Determine tooltip orientation based on screen position
  const isNearRightEdge = position.x > window.innerWidth - 180;

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerCancel}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => !isDragging && setIsHovered(false)}
      title="Knovera AI • Drag to move • Click to chat"
      role="button"
      tabIndex={0}
      aria-label="Open Knovera AI Assistant"
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
      style={{
        position: 'fixed',
        left: `${position.x}px`,
        top: `${position.y}px`,
        width: `${BUTTON_SIZE}px`,
        height: `${BUTTON_SIZE}px`,
        zIndex: 9999,
        touchAction: 'none',
        userSelect: 'none',
        WebkitUserSelect: 'none',
        cursor: isDragging ? 'grabbing' : 'grab',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '50%',
        background: isDragging
          ? 'radial-gradient(circle at 30% 30%, #1e293b 0%, #090d16 100%)'
          : isHovered
          ? 'radial-gradient(circle at 35% 35%, #2563eb 0%, #0f172a 85%)'
          : 'radial-gradient(circle at 35% 35%, #1e293b 0%, #0f172a 100%)',
        color: '#ffffff',
        boxShadow: isDragging
          ? '0 24px 44px -6px rgba(15, 23, 42, 0.65), 0 0 0 3px rgba(96, 165, 250, 0.65), 0 0 32px rgba(59, 130, 246, 0.5)'
          : isHovered
          ? '0 16px 36px -4px rgba(37, 99, 235, 0.45), 0 0 0 2px rgba(147, 197, 253, 0.5), 0 0 24px rgba(59, 130, 246, 0.4)'
          : '0 10px 25px -4px rgba(15, 23, 42, 0.35), 0 4px 10px -2px rgba(15, 23, 42, 0.18), 0 0 16px -2px rgba(59, 130, 246, 0.25)',
        border: isDragging
          ? '1.5px solid rgba(147, 197, 253, 0.8)'
          : isHovered
          ? '1.5px solid rgba(255, 255, 255, 0.4)'
          : '1.5px solid rgba(255, 255, 255, 0.18)',
        transform: isDragging
          ? 'scale(1.12)'
          : isHovered
          ? 'scale(1.08)'
          : 'scale(1)',
        transition: isDragging
          ? 'transform 0.1s ease, box-shadow 0.15s ease'
          : 'transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease, background 0.3s ease, border-color 0.2s ease',
      }}
    >
      {/* Centered AI Sparkles Icon */}
      <div
        style={{
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          pointerEvents: 'none',
        }}
      >
        <Sparkles
          size={24}
          color={isHovered ? '#ffffff' : '#60a5fa'}
          style={{
            transform: isHovered || isDragging ? 'rotate(15deg) scale(1.1)' : 'rotate(0) scale(1)',
            transition: 'transform 0.25s ease, color 0.2s ease',
            filter: isHovered
              ? 'drop-shadow(0 0 8px rgba(255, 255, 255, 0.8))'
              : 'drop-shadow(0 0 6px rgba(96, 165, 250, 0.6))',
          }}
        />

        {/* Pulsing online indicator dot */}
        <span
          style={{
            position: 'absolute',
            top: '-5px',
            right: '-5px',
            width: '9px',
            height: '9px',
            borderRadius: '50%',
            backgroundColor: '#10b981',
            boxShadow: '0 0 8px #10b981',
            border: '1.5px solid #0f172a',
          }}
        />
      </div>

      {/* Floating Tooltip Pill on Hover (only when not dragging) */}
      {isHovered && !isDragging && (
        <div
          style={{
            position: 'absolute',
            ...(isNearRightEdge
              ? { right: `${BUTTON_SIZE + 10}px`, left: 'auto' }
              : { left: `${BUTTON_SIZE + 10}px`, right: 'auto' }),
            top: '50%',
            transform: 'translateY(-50%)',
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
            padding: '6px 12px',
            backgroundColor: '#0f172a',
            color: '#ffffff',
            borderRadius: '9999px',
            fontSize: '12px',
            fontWeight: 600,
            letterSpacing: '-0.01em',
            boxShadow: '0 8px 20px -2px rgba(15, 23, 42, 0.35)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            animation: 'fadeIn 0.15s ease-out forwards',
          }}
        >
          <span>Knovera AI</span>
          <span
            style={{
              fontSize: '9.5px',
              padding: '1px 5px',
              borderRadius: '4px',
              backgroundColor: 'rgba(59, 130, 246, 0.25)',
              color: '#93c5fd',
              fontWeight: 700,
              letterSpacing: '0.04em',
            }}
          >
            RAG
          </span>
        </div>
      )}
    </div>
  );
}
