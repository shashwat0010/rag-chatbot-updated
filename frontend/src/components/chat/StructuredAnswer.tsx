"use client";

import type { ReactNode } from "react";

/**
 * Renders backend markdown-lite answers: **Section:** headers and - bullets.
 */
export function StructuredAnswer({ text }: { text: string }) {
  if (!text) return null;
  const lines = text.split("\n");

  if (lines.length === 1 && !lines[0].startsWith("**") && !lines[0].startsWith("- ")) {
    return <p className="text-sm leading-relaxed text-foreground">{lines[0]}</p>;
  }

  const blocks: ReactNode[] = [];
  let listItems: string[] = [];

  const flushList = () => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="ml-1 list-disc space-y-2 pl-5 text-sm text-foreground">
        {listItems.map((item, i) => (
          <li key={i} className="leading-relaxed">
            {renderInline(item)}
          </li>
        ))}
      </ul>
    );
    listItems = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      continue;
    }

    const headerMatch = trimmed.match(/^\*\*(.+?):\*\*\s*(.*)$/);
    if (headerMatch) {
      flushList();
      const [, title, rest] = headerMatch;
      blocks.push(
        <div key={`h-${blocks.length}`} className="space-y-1">
          <p className="text-sm font-semibold text-foreground">{title}</p>
          {rest ? (
            <p className="text-sm leading-relaxed text-foreground">{renderInline(rest)}</p>
          ) : null}
        </div>
      );
      continue;
    }

    if (trimmed.startsWith("- ")) {
      listItems.push(trimmed.slice(2));
      continue;
    }

    flushList();
    blocks.push(
      <p key={`p-${blocks.length}`} className="text-sm leading-relaxed text-foreground">
        {renderInline(trimmed)}
      </p>
    );
  }
  flushList();

  return <div className="space-y-3">{blocks}</div>;
}

function renderInline(text: string): ReactNode {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) =>
    /^\[\d+\]$/.test(part) ? (
      <span key={i} className="font-medium text-primary">
        {part}
      </span>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}
