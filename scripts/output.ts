// File outputs of the scripts: calculation logs as Markdown and tables as CSV
import fs from 'fs';
import type { CalcStep } from '../shared/cashflow.ts';

// Numbered sections «## 1. Title» with one bullet per line of the step
export function stepsToMarkdown(steps: CalcStep[], level = 2): string[] {
  const h = '#'.repeat(level);
  // A line indented with two spaces continues the previous one: a nested bullet
  return steps.flatMap((s, i) => [`${h} ${i + 1}. ${s.title}`, '', ...s.lines.map(l => (l.startsWith('  ') ? `  - ${l.trim()}` : `- ${l}`)), '']);
}

// Semicolon-separated like the case files; the BOM makes Excel read the Cyrillic text as UTF-8
export function writeCsv(file: string, header: string[], rows: Array<Array<string | number | null>>): void {
  const cell = (v: string | number | null) => {
    if (v === null) return '';
    const s = String(v);
    return /[;"\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  fs.writeFileSync(file, '﻿' + [header, ...rows].map(r => r.map(cell).join(';')).join('\r\n') + '\r\n');
}
