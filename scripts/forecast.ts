// Forecast for every company of statement files in the case format, computed exactly as the site does it
// (the same parser, cleaning and analyzeStatement), with the step-by-step calculation log of each company.
//
// Run:  node scripts/forecast.ts --out-dir <folder> file.csv [more.csv ...]
// Output in the folder:
//   forecast_companies.csv  one row per company: balance, gap probability, status, worst P10 / P50, risky dates
//   forecast_days.csv       the 30 forecast days of every company: P10 / P50 / P90 and the share of scenarios below zero
//   calc_log.md             step-by-step log of every company (the same text as the «Журнал расчёта» tab of the site)
//   forecast.json           all of the above for the HTML report (scripts/report.ts)

import fs from 'fs';
import path from 'path';
import { analyzeStatement, explainForecast, formatRubExact } from '../shared/cashflow.ts';
import type { CalcStep, ForecastPoint, GapAlert } from '../shared/cashflow.ts';
import { parseStatementCsv, cleanCompanyOperations, explainParsing, plural } from '../shared/statement.ts';
import type { ParsedOperation, ParsedStatement } from '../shared/statement.ts';
import { stepsToMarkdown, writeCsv } from './output.ts';

const args = process.argv.slice(2);
const outAt = args.indexOf('--out-dir');
const OUT_DIR = outAt >= 0 ? args[outAt + 1] : 'reports/forecast';
const files = outAt >= 0 ? args.filter((_, i) => i !== outAt && i !== outAt + 1) : args;
if (files.length === 0) {
  console.error('Укажите файл с выписками: node scripts/forecast.ts --out-dir папка файл.csv');
  process.exit(1);
}

const METHOD_RU: Record<string, string> = {
  analog_periods: 'похожие периоды истории',
  block_bootstrap: 'недельные блоки истории',
  day_bootstrap: 'отдельные дни истории',
};
const ruDate = (iso: string) => iso.split('-').reverse().join('.');

interface Input {
  id: string;
  file: string;
  parsed: ParsedStatement;
  ops: ParsedOperation[];
}
const inputs = new Map<string, Input>();
const fileEnd = new Map<string, string>(); // last operation date of each file
for (const file of files) {
  const name = path.basename(file);
  const parsed = parseStatementCsv(fs.readFileSync(file, 'utf8'));
  if (parsed.missing.length > 0) {
    console.error(`${name}: не найдены колонки ${parsed.missing.join(', ')}`);
    process.exit(1);
  }
  for (const op of parsed.operations) {
    // Without a company column the file is one company, named after the file as on the site
    const id = op.company || path.basename(file, path.extname(file)).toUpperCase();
    const item = inputs.get(id) ?? { id, file: name, parsed, ops: [] };
    item.ops.push(op);
    inputs.set(id, item);
    if (op.date > (fileEnd.get(name) ?? '')) fileEnd.set(name, op.date);
  }
}

export interface CompanyForecast {
  id: string;
  file: string;
  period: { start: string; end: string };
  daysBeforeFileEnd: number; // the statement ends this many days before the last date of the file
  operations: number;
  duplicatesRemoved: number;
  balanceBreaks: number;
  hasClosingBalance: boolean;
  currentBalance: number;
  probability: number;
  gapPaths: number;
  scenarios: number;
  method: string;
  status: string;
  statusLabel: string;
  minP10: number;
  minP10Date: string;
  minP50: number;
  minP50Date: string;
  firstRiskDate: string | null; // first day with at least 10% of scenarios below zero or P50 below zero
  alerts: GapAlert[];
  forecast: ForecastPoint[];
  log: CalcStep[];
}

const results: CompanyForecast[] = [];
for (const c of [...inputs.values()].sort((a, b) => a.id.localeCompare(b.id, 'ru', { numeric: true }))) {
  const clean = cleanCompanyOperations(c.ops);
  if (clean.operations.length === 0) continue;
  // What the site sends to /api/analyze-statement for this company; without a balance column the opening balance is 0
  const a = analyzeStatement(clean.operations, { closingBalance: clean.closingBalance });
  const f = a.sim.forecast;
  const argMin = (pick: (p: ForecastPoint) => number) => f.reduce((best, p) => (pick(p) < pick(best) ? p : best), f[0]);
  const risky = f.find(p => p.gap_probability_pct >= 10 || p.balance_p50 < 0);
  const end = fileEnd.get(c.file)!;
  results.push({
    id: c.id,
    file: c.file,
    period: a.period,
    daysBeforeFileEnd: Math.round((Date.parse(end) - Date.parse(a.period.end)) / 86_400_000),
    operations: clean.operations.length,
    duplicatesRemoved: clean.duplicatesRemoved,
    balanceBreaks: clean.balanceBreaks.length,
    hasClosingBalance: a.hasClosingBalance,
    currentBalance: Math.round(a.currentBalance),
    probability: a.sim.gapProbabilityPct,
    gapPaths: a.sim.gapPaths,
    scenarios: a.sim.paths.length,
    method: a.sim.method,
    status: a.status.level,
    statusLabel: a.status.label,
    minP10: a.minP10,
    minP10Date: argMin(p => p.balance_p10).date,
    minP50: a.minP50,
    minP50Date: argMin(p => p.balance_p50).date,
    firstRiskDate: risky?.date ?? null,
    alerts: a.alerts,
    forecast: f,
    log: [...explainParsing(c.file, c.parsed, c.id, clean), ...explainForecast(a)],
  });
}
if (results.length === 0) {
  console.error('В файлах нет ни одной операции с корректной датой и суммой');
  process.exit(1);
}

fs.mkdirSync(OUT_DIR, { recursive: true });
writeCsv(
  path.join(OUT_DIR, 'forecast_companies.csv'),
  ['company_id', 'file', 'period_start', 'period_end', 'operations', 'current_balance', 'gap_probability_pct', 'gap_scenarios', 'scenarios',
    'method', 'status', 'min_p10', 'min_p10_date', 'min_p50', 'min_p50_date', 'first_risk_date', 'alerts', 'duplicates_removed',
    'balance_break_days', 'days_before_file_end'],
  results.map(r => [
    r.id, r.file, r.period.start, r.period.end, r.operations, r.currentBalance, r.probability, r.gapPaths, r.scenarios, METHOD_RU[r.method],
    r.statusLabel, r.minP10, r.minP10Date, r.minP50, r.minP50Date, r.firstRiskDate, r.alerts.length, r.duplicatesRemoved, r.balanceBreaks,
    r.daysBeforeFileEnd,
  ])
);
writeCsv(
  path.join(OUT_DIR, 'forecast_days.csv'),
  ['company_id', 'date', 'p10', 'p50', 'p90', 'below_zero_pct', 'mean_inflow', 'mean_outflow', 'scheduled_payments'],
  results.flatMap(r => r.forecast.map(p => [r.id, p.date, p.balance_p10, p.balance_p50, p.balance_p90, p.gap_probability_pct, p.inflow, p.outflow, p.sched_description]))
);

const counts = { CRITICAL: 0, WARNING: 0, STABLE: 0 } as Record<string, number>;
for (const r of results) counts[r.status]++;
const companiesText = `${results.length} ${plural(results.length, 'компания', 'компании', 'компаний')}`;
fs.writeFileSync(path.join(OUT_DIR, 'calc_log.md'), [
  '# Журнал расчёта прогноза',
  '',
  `Файлы: ${files.map(f => path.basename(f)).join(', ')}; ${companiesText}. ` +
    `Критический статус — ${counts.CRITICAL}, повышенный риск — ${counts.WARNING}, стабильно — ${counts.STABLE}. ` +
    'Для каждой компании — те же шаги, что на вкладке «Журнал расчёта» сайта.',
  '',
  ...results.flatMap(r => [
    `## ${r.id}: ${r.statusLabel}, вероятность разрыва ${r.probability}%`,
    '',
    ...stepsToMarkdown(r.log, 3),
  ]),
].join('\n'));
fs.writeFileSync(path.join(OUT_DIR, 'forecast.json'), JSON.stringify({
  generated_at: new Date().toISOString(),
  files: files.map(f => path.basename(f)),
  status_counts: counts,
  companies: results,
}));

console.log(`Прогноз построен: ${companiesText}; критический статус — ${counts.CRITICAL}, повышенный риск — ${counts.WARNING}, стабильно — ${counts.STABLE}`);
const top = [...results].sort((a, b) => b.probability - a.probability).slice(0, 5);
for (const r of top) {
  console.log(`  ${r.id.padEnd(8)} вероятность ${String(r.probability).padStart(5)}%  остаток ${formatRubExact(r.currentBalance).padStart(16)}  ` +
    `худший P10 ${formatRubExact(r.minP10).padStart(16)} (${ruDate(r.minP10Date)})  ${r.statusLabel}`);
}
console.log(`Сохранено в ${OUT_DIR}: forecast_companies.csv, forecast_days.csv, calc_log.md`);
