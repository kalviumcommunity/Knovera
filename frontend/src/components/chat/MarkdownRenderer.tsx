'use client';

import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const handleCopy = (code: string, index: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIndex(index);
    setTimeout(() => {
      setCopiedIndex(null);
    }, 2000);
  };

  // Parse lines into tokens: code blocks, tables, headers, lists, paragraphs
  const renderFormattedText = (text: string) => {
    // Process inline code, bold, italic, links
    const parts = [];
    let remaining = text;
    let keyIdx = 0;

    while (remaining.length > 0) {
      // Inline code: `code`
      const codeMatch = remaining.match(/^`([^`]+)`/);
      if (codeMatch) {
        parts.push(
          <code key={`code-${keyIdx++}`}>{codeMatch[1]}</code>
        );
        remaining = remaining.slice(codeMatch[0].length);
        continue;
      }

      // Bold: **text** or __text__
      const boldMatch = remaining.match(/^(\*\*|__)(.*?)\1/);
      if (boldMatch) {
        parts.push(
          <strong key={`bold-${keyIdx++}`} style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {boldMatch[2]}
          </strong>
        );
        remaining = remaining.slice(boldMatch[0].length);
        continue;
      }

      // Italic: *text* or _text_
      const italicMatch = remaining.match(/^(\*|_)(.*?)\1/);
      if (italicMatch && !boldMatch) {
        parts.push(
          <em key={`italic-${keyIdx++}`}>{italicMatch[2]}</em>
        );
        remaining = remaining.slice(italicMatch[0].length);
        continue;
      }

      // Link: [text](url)
      const linkMatch = remaining.match(/^\[(.*?)\]\((.*?)\)/);
      if (linkMatch) {
        parts.push(
          <a
            key={`link-${keyIdx++}`}
            href={linkMatch[2]}
            target="_blank"
            rel="noopener noreferrer"
          >
            {linkMatch[1]}
          </a>
        );
        remaining = remaining.slice(linkMatch[0].length);
        continue;
      }

      // Standard text up to next special char
      const nextSpecial = remaining.search(/[`*_\[]/);
      if (nextSpecial === -1) {
        parts.push(<React.Fragment key={`text-${keyIdx++}`}>{remaining}</React.Fragment>);
        break;
      } else if (nextSpecial === 0) {
        parts.push(<React.Fragment key={`char-${keyIdx++}`}>{remaining[0]}</React.Fragment>);
        remaining = remaining.slice(1);
      } else {
        parts.push(<React.Fragment key={`text-${keyIdx++}`}>{remaining.slice(0, nextSpecial)}</React.Fragment>);
        remaining = remaining.slice(nextSpecial);
      }
    }

    return parts;
  };

  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];

  let i = 0;
  let blockIndex = 0;

  while (i < lines.length) {
    const line = lines[i];

    // 1. Code block detection: ```lang
    if (line.trim().startsWith('```')) {
      const lang = line.trim().slice(3).trim() || 'text';
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      const fullCode = codeLines.join('\n');
      const currentIndex = blockIndex++;

      elements.push(
        <div key={`code-block-${currentIndex}`} className="code-block-wrapper">
          <div className="code-block-header">
            <span>{lang.toLowerCase()}</span>
            <button
              onClick={() => handleCopy(fullCode, currentIndex)}
              className="code-block-copy-btn"
              title="Copy code to clipboard"
            >
              {copiedIndex === currentIndex ? (
                <>
                  <Check size={12} color="#059669" />
                  <span style={{ color: '#059669', fontWeight: 500 }}>Copied!</span>
                </>
              ) : (
                <>
                  <Copy size={12} />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
          <pre className="code-block-content">
            <code>{fullCode}</code>
          </pre>
        </div>
      );
      continue;
    }

    // 2. Table detection: lines starting with '|'
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
        tableLines.push(lines[i].trim());
        i++;
      }

      if (tableLines.length >= 2) {
        const headerRow = tableLines[0]
          .slice(1, -1)
          .split('|')
          .map((c) => c.trim());
        // Check if second line is separator |---|---|
        const isSeparator = /^\|?(\s*:?-+:?\s*\|?)+$/.test(tableLines[1]);
        const dataRows = isSeparator
          ? tableLines.slice(2).map((r) =>
              r
                .slice(1, -1)
                .split('|')
                .map((c) => c.trim())
            )
          : tableLines.slice(1).map((r) =>
              r
                .slice(1, -1)
                .split('|')
                .map((c) => c.trim())
            );

        elements.push(
          <div key={`table-${blockIndex++}`} style={{ overflowX: 'auto', margin: '0.85rem 0' }}>
            <table>
              <thead>
                <tr>
                  {headerRow.map((h, colIdx) => (
                    <th key={`th-${colIdx}`}>{renderFormattedText(h)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataRows.map((row, rIdx) => (
                  <tr key={`tr-${rIdx}`}>
                    {row.map((cell, cIdx) => (
                      <td key={`td-${rIdx}-${cIdx}`}>{renderFormattedText(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }
    }

    // 3. Headers: #, ##, ###
    if (line.startsWith('### ')) {
      elements.push(<h3 key={`h3-${i}`}>{renderFormattedText(line.slice(4))}</h3>);
      i++;
      continue;
    }
    if (line.startsWith('## ')) {
      elements.push(<h2 key={`h2-${i}`}>{renderFormattedText(line.slice(3))}</h2>);
      i++;
      continue;
    }
    if (line.startsWith('# ')) {
      elements.push(<h1 key={`h1-${i}`}>{renderFormattedText(line.slice(2))}</h1>);
      i++;
      continue;
    }

    // 4. Blockquotes: >
    if (line.startsWith('> ')) {
      elements.push(
        <blockquote key={`quote-${i}`}>
          {renderFormattedText(line.slice(2))}
        </blockquote>
      );
      i++;
      continue;
    }

    // 5. Unordered list: - or *
    if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      const listItems: string[] = [];
      while (
        i < lines.length &&
        (lines[i].trim().startsWith('- ') || lines[i].trim().startsWith('* '))
      ) {
        listItems.push(lines[i].trim().slice(2));
        i++;
      }
      elements.push(
        <ul key={`ul-${blockIndex++}`}>
          {listItems.map((item, idx) => (
            <li key={`li-${idx}`}>{renderFormattedText(item)}</li>
          ))}
        </ul>
      );
      continue;
    }

    // 6. Ordered list: 1. 2.
    if (/^\d+\.\s/.test(line.trim())) {
      const listItems: string[] = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i].trim())) {
        listItems.push(lines[i].trim().replace(/^\d+\.\s/, ''));
        i++;
      }
      elements.push(
        <ol key={`ol-${blockIndex++}`}>
          {listItems.map((item, idx) => (
            <li key={`oli-${idx}`}>{renderFormattedText(item)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // 7. Regular paragraph or empty line
    if (!line.trim()) {
      elements.push(<div key={`spacer-${i}`} style={{ height: '0.45rem' }} />);
    } else {
      elements.push(
        <p key={`p-${i}`}>
          {renderFormattedText(line)}
        </p>
      );
    }
    i++;
  }

  return <div className="markdown-body">{elements}</div>;
}
