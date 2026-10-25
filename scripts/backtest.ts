// Backtest of the cash-flow forecast on statement files in the case format.
//
// For every company and every cutoff date the forecast is built by the same function as the site
// (analyzeStatement), only from the operations up to the cutoff, and compared with the actual end-of-day
// balances of the next 30 days:
//   - balance: MAE, MSE, RMSE, WAPE and bias of the median P50 (and of two simple baselines) by horizon
//     day, and how often the fact falls inside the P10–P90 band;
//   - cash gap = a negative end-of-day balance on any day of the horizon: confusion matrix, accuracy,
//     balanced accuracy, precision, recall, specificity and F1 for every probability threshold and for the
//     site statuses («Повышенный риск» from 10%, «Критический» from 50%); ROC-AUC, average precision,
//     Brier score and calibration of the probability; lead time = days from the forecast to the first
//     negative day. Only forecasts made before the gap count: if the balance is already negative at the
//     cutoff, the gap is not a prediction.
// 95% intervals: Wilson for shares, bootstrap over companies for the main metrics.
//
// Run:  node scripts/backtest.ts [--out metrics.json] [statement.csv ...]   (default: data/raw/train.csv data/raw/test.csv)
// Output: reports/backtest_metrics.json or the --out path, and next to it backtest_cases.csv (every forecast
// against the fact) and backtest_log.md (step-by-step calculation of the metrics). Charts are drawn by
// scripts/plot_backtest.py; for a new file everything is done by: npm run evaluate -- file.csv

import fs from 'fs';
import path from 'path';
import { analyzeStatement, buildDailySeries, addDaysIso, formatRubExact, HORIZON_DAYS } from '../shared/cashflow.ts';
import type { CalcStep, RiskLevel } from '../shared/cashflow.ts';
import { parseStatementCsv, cleanCompanyOperations, plural } from '../shared/statement.ts';
import type { ParsedOperation } from '../shared/statement.ts';
import { stepsToMarkdown, writeCsv } from './output.ts';

const CUTOFF_COUNT = 6;
const MIN_HISTORY_DAYS = 90;
const THRESHOLDS = Array.from({ length: 19 }, (_, i) => (i + 1) * 5); // 5% .. 95%
const BOOTSTRAP_ROUNDS = 1000;
const GAP_DEFINITION = 'остаток на конец дня ниже 0 ₽ хотя бы в один день горизонта';
const GAP_EVAL_RULE = 'только прогнозы с неотрицательным остатком на дату среза (разрыв ещё не наступил)';

const args = process.argv.slice(2);
const outAt = args.indexOf('--out');
const OUT_FILE = outAt >= 0 ? args[outAt + 1] : 'reports/backtest_metrics.json';
const OUT_DIR = path.dirname(OUT_FILE);
const inputFiles = outAt >= 0 ? args.filter((_, i) => i !== outAt && i !== outAt + 1) : args;
const files = inputFiles.length > 0 ? inputFiles : ['data/raw/train.csv', 'data/raw/test.csv'];

// ---------------------------------------------------------------- formatting for the log
const rub = formatRubExact;
const ruDate = (iso: string) => iso.split('-').reverse().join('.');
const count = (n: number, one: string, few: string, many: string) => `${n.toLocaleString('ru-RU')} ${plural(n, one, few, many)}`;
const fixed = (v: number | null, digits = 3) =>
  v === null || !Number.isFinite(v) ? '—' : v.toLocaleString('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits });
const pct = (v: number | null, digits = 1) => (v === null || !Number.isFinite(v) ? '—' : `${fixed(v * 100, digits)}%`);
const ci = (range: [number, number] | null, f: (v: number) => string) => (range ? ` (95% ДИ ${f(range[0])} – ${f(range[1])})` : '');
const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹';
// 2.31e+12 -> «2,31·10¹²»: MSE is in squared rubles and too long to read in full
function sci(v: number): string {
  const [mantissa, exp] = v.toExponential(2).split('e');
  const power = String(Number(exp)).split('').map(d => (d === '-' ? '⁻' : SUPERSCRIPT[Number(d)])).join('');
  return `${mantissa.replace('.', ',')}·10${power}`;
}

// ---------------------------------------------------------------- data
interface Company {
  id: string;
  source: string; // file the statement came from
  ops: ParsedOperation[];
  actual: Map<string, number>; // actual end-of-day balance for every calendar day
}

// Actual end-of-day balance for every calendar day, from the balance column (days without operations keep
// the previous balance). A statement that ends a few days before the file end is extended to it.
function actualEodByDay(ops: ParsedOperation[], datasetEnd: string): Map<string, number> {
  const firstDay = ops.filter(op => op.date === ops[0].date);
  const firstEod = [...firstDay].reverse().find(op => op.balance !== null)?.balance ?? 0;
  const opening = firstEod - firstDay.reduce((s, op) => s + (op.isCredit ? op.amount : -op.amount), 0);
  const days = buildDailySeries(ops, opening);
  const out = new Map(days.map(d => [d.date, d.eod]));
  const last = days[days.length - 1];
  if (addDaysIso(last.date, 7) >= datasetEnd) {
    for (let d = addDaysIso(last.date, 1); d <= datasetEnd; d = addDaysIso(d, 1)) out.set(d, last.eod);
  }
  return out;
}

const readLines: string[] = [];
const grouped = new Map<string, { source: string; ops: ParsedOperation[] }>();
for (const file of files) {
  const name = path.basename(file);
  const parsed = parseStatementCsv(fs.readFileSync(file, 'utf8'));
  if (parsed.missing.length > 0) {
    console.error(`${name}: не найдены колонки ${parsed.missing.join(', ')}`);
    process.exit(1);
  }
  readLines.push(
    `«${name}»: ${count(parsed.rows, 'строка', 'строки', 'строк')} с операциями, ${count(parsed.companies.length, 'компания', 'компании', 'компаний')}; ` +
      `пропущено строк с некорректной датой или суммой: ${parsed.skipped}.`
  );
  for (const op of parsed.operations) {
    const id = op.company || path.basename(file, path.extname(file)); // no company column: the file is one company
    const item = grouped.get(id) ?? { source: name, ops: [] };
    item.ops.push(op);
    grouped.set(id, item);
  }
}

let duplicatesRemoved = 0;
let breakDays = 0;
const companies: Company[] = [];
for (const [id, { source, ops }] of grouped) {
  const clean = cleanCompanyOperations(ops);
  if (clean.operations.length === 0) continue;
  duplicatesRemoved += clean.duplicatesRemoved;
  breakDays += clean.balanceBreaks.length;
  companies.push({ id, source, ops: clean.operations, actual: new Map() });
}
if (companies.length === 0) {
  console.error('В файлах нет ни одной операции в формате кейса (company_id;operation_id;operation_date;amount;direction;...)');
  process.exit(1);
}
if (!companies.some(c => c.ops.some(op => op.balance !== null))) {
  console.error('Для бэктеста нужна колонка остатка на конец дня (balance_eod): без неё неизвестен фактический остаток');
  process.exit(1);
}
const datasetStart = companies.map(c => c.ops[0].date).sort()[0];
const datasetEnd = companies.map(c => c.ops[c.ops.length - 1].date).sort().at(-1)!;
for (const c of companies) c.actual = actualEodByDay(c.ops, datasetEnd);

// The last `count` month ends that have HORIZON_DAYS of facts after them and at least MIN_HISTORY_DAYS of data before
function monthEndCutoffs(start: string, end: string, count: number): string[] {
  const cutoffs: string[] = [];
  let year = Number(end.slice(0, 4));
  let month = Number(end.slice(5, 7)); // 1..12
  while (cutoffs.length < count) {
    const monthEnd = new Date(Date.UTC(year, month, 0)).toISOString().slice(0, 10);
    if (monthEnd < addDaysIso(start, MIN_HISTORY_DAYS)) break;
    if (addDaysIso(monthEnd, HORIZON_DAYS) <= end) cutoffs.unshift(monthEnd);
    month -= 1;
    if (month === 0) { month = 12; year -= 1; }
  }
  return cutoffs;
}
const CUTOFFS = monthEndCutoffs(datasetStart, datasetEnd, CUTOFF_COUNT);
const dayDiff = (from: string, to: string) => Math.round((Date.parse(to) - Date.parse(from)) / 86_400_000);

// ---------------------------------------------------------------- forecasts against the fact
interface Case {
  company: string;
  source: string;
  cutoff: string;
  startBalance: number; // actual balance at the cutoff = the balance the forecast starts from
  prob: number; // forecast probability of a gap within the horizon, %
  status: RiskLevel;
  gapPaths: number;
  scenarios: number;
  minP10: number;
  minP50: number;
  p10: number[];
  p50: number[];
  p90: number[];
  naive: number[];
  trend: number[];
  facts: number[];
  firstNegDay: number | null; // 1..HORIZON_DAYS: first day of the horizon with a negative actual balance
}

const cases: Case[] = [];
const skipped = { no_history: 0, short_history: 0, no_facts: 0 };
for (const c of companies) {
  for (const cutoff of CUTOFFS) {
    const hist = c.ops.filter(op => op.date <= cutoff);
    const horizon = Array.from({ length: HORIZON_DAYS }, (_, i) => addDaysIso(cutoff, i + 1));
    if (hist.length === 0) { skipped.no_history++; continue; }
    if (!c.actual.has(horizon[HORIZON_DAYS - 1])) { skipped.no_facts++; continue; }
    if (dayDiff(hist[0].date, cutoff) + 1 < MIN_HISTORY_DAYS) { skipped.short_history++; continue; }
    const startBalance = c.actual.get(cutoff)!;
    // The same function as the site; the balance at the cutoff plays the closing balance of the statement
    const a = analyzeStatement(hist, { closingBalance: startBalance, asOf: cutoff });
    const recent = a.days.slice(-90);
    const net90 = recent.reduce((s, d) => s + d.inflow - d.outflow, 0) / recent.length;
    const facts = horizon.map(d => c.actual.get(d)!);
    const neg = facts.findIndex(v => v < 0);
    cases.push({
      company: c.id,
      source: c.source,
      cutoff,
      startBalance,
      prob: a.sim.gapProbabilityPct,
      status: a.status.level,
      gapPaths: a.sim.gapPaths,
      scenarios: a.sim.paths.length,
      minP10: a.minP10,
      minP50: a.minP50,
      p10: a.sim.forecast.map(f => f.balance_p10),
      p50: a.sim.forecast.map(f => f.balance_p50),
      p90: a.sim.forecast.map(f => f.balance_p90),
      naive: horizon.map(() => startBalance),
      trend: horizon.map((_, i) => startBalance + net90 * (i + 1)),
      facts,
      firstNegDay: neg >= 0 ? neg + 1 : null,
    });
  }
}
if (cases.length === 0) {
  console.error(
    `Ни одного прогноза: данные ${datasetStart} .. ${datasetEnd}, даты среза: ${CUTOFFS.join(', ') || 'нет'}. ` +
      `Нужно не меньше ${MIN_HISTORY_DAYS} дней истории до среза и ${HORIZON_DAYS} дней факта после него.`
  );
  process.exit(1);
}

// ---------------------------------------------------------------- metrics
const sum = (xs: number[]) => xs.reduce((s, x) => s + x, 0);
const mean = (xs: number[]) => (xs.length > 0 ? sum(xs) / xs.length : null);
const ratio = (a: number, b: number) => (b > 0 ? a / b : null);
function quantile(sorted: number[], p: number): number {
  const pos = (sorted.length - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
const isGap = (c: Case) => c.firstNegDay !== null;

// 95% Wilson interval for a share k / n: stays inside [0, 1] and is honest about small samples
function wilson(k: number, n: number): [number, number] | null {
  if (n === 0) return null;
  const z = 1.96;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / d;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

const MODELS = ['analog', 'naive', 'trend'] as const;
type Model = (typeof MODELS)[number];
const MODEL_NAMES: Record<Model, string> = {
  analog: 'Аналоговые периоды (текущая модель)',
  naive: 'Наивная: остаток не меняется',
  trend: 'Линейный тренд за 90 дней',
};
const predictions = (c: Case, m: Model) => (m === 'analog' ? c.p50 : m === 'naive' ? c.naive : c.trend);

// Sums of errors (forecast − fact) over all cases and horizon days
function errorSums(cs: Case[], m: Model) {
  let abs = 0, sq = 0, signed = 0, fact = 0;
  for (const c of cs) {
    const p = predictions(c, m);
    for (let h = 0; h < HORIZON_DAYS; h++) {
      const e = p[h] - c.facts[h];
      abs += Math.abs(e);
      sq += e * e;
      signed += e;
      fact += Math.abs(c.facts[h]);
    }
  }
  const n = cs.length * HORIZON_DAYS;
  return { n, abs, sq, signed, fact, mae: abs / n, mse: sq / n, rmse: Math.sqrt(sq / n), wape: fact > 0 ? abs / fact : null, bias: signed / n };
}

function regression(cs: Case[], m: Model) {
  const byH = (pick: (e: number) => number) =>
    Array.from({ length: HORIZON_DAYS }, (_, h) => sum(cs.map(c => pick(predictions(c, m)[h] - c.facts[h]))) / cs.length);
  const s = errorSums(cs, m);
  const absErrors = cs.flatMap(c => predictions(c, m).map((p, h) => Math.abs(p - c.facts[h]))).sort((a, b) => a - b);
  const mseByH = byH(e => e * e);
  return {
    name: MODEL_NAMES[m],
    points: s.n,
    sum_abs_error: s.abs,
    sum_sq_error: s.sq,
    sum_abs_fact: s.fact,
    mae: s.mae,
    mse: s.mse,
    rmse: s.rmse,
    wape: s.wape,
    bias: s.bias,
    median_ae: quantile(absErrors, 0.5),
    mae_by_horizon: byH(Math.abs),
    mse_by_horizon: mseByH,
    rmse_by_horizon: mseByH.map(Math.sqrt),
    bias_by_horizon: byH(e => e),
  };
}

// How often the actual balance falls inside the P10–P90 band (about 80% if the band is honest)
function coverage(cs: Case[]) {
  const inside = new Array(HORIZON_DAYS).fill(0);
  let below = 0, above = 0;
  for (const c of cs) {
    for (let h = 0; h < HORIZON_DAYS; h++) {
      if (c.facts[h] < c.p10[h]) below++;
      else if (c.facts[h] > c.p90[h]) above++;
      else inside[h]++;
    }
  }
  const n = cs.length * HORIZON_DAYS;
  return { points: n, inside: sum(inside) / n, below_p10: below / n, above_p90: above / n, inside_by_horizon: inside.map(v => v / cs.length) };
}

interface Confusion { tp: number; fp: number; fn: number; tn: number }
function confusion(cs: Case[], flagged: (c: Case) => boolean): Confusion {
  const cm = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (const c of cs) {
    if (flagged(c)) isGap(c) ? cm.tp++ : cm.fp++;
    else isGap(c) ? cm.fn++ : cm.tn++;
  }
  return cm;
}
function scores(cm: Confusion) {
  const n = cm.tp + cm.fp + cm.fn + cm.tn;
  const recall = ratio(cm.tp, cm.tp + cm.fn);
  const specificity = ratio(cm.tn, cm.tn + cm.fp);
  return {
    ...cm,
    n,
    signals: cm.tp + cm.fp,
    accuracy: ratio(cm.tp + cm.tn, n),
    accuracy_ci95: wilson(cm.tp + cm.tn, n),
    balanced_accuracy: recall !== null && specificity !== null ? (recall + specificity) / 2 : null,
    precision: ratio(cm.tp, cm.tp + cm.fp),
    precision_ci95: wilson(cm.tp, cm.tp + cm.fp),
    recall,
    recall_ci95: wilson(cm.tp, cm.tp + cm.fn),
    specificity,
    specificity_ci95: wilson(cm.tn, cm.tn + cm.fp),
    f1: ratio(2 * cm.tp, 2 * cm.tp + cm.fp + cm.fn),
  };
}

// Site statuses: «Повышенный риск» = probability ≥ 10% or P10 below zero, «Критический» = ≥ 50% or P50 below zero
const STATUS = {
  warning: { name: '«Повышенный риск» и выше', threshold: 10, flagged: (c: Case) => c.status !== 'STABLE' },
  critical: { name: '«Критический»', threshold: 50, flagged: (c: Case) => c.status === 'CRITICAL' },
} as const;
type StatusKey = keyof typeof STATUS;
const STATUS_KEYS: StatusKey[] = ['warning', 'critical'];

// ROC-AUC = probability that a random forecast with a gap got a higher probability than one without (ties count half)
function rocAuc(cs: Case[]): number | null {
  const pos = cs.filter(isGap).length;
  const neg = cs.length - pos;
  if (pos === 0 || neg === 0) return null;
  const sorted = [...cs].sort((a, b) => a.prob - b.prob);
  let rankSum = 0;
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j < sorted.length && sorted[j].prob === sorted[i].prob) j++;
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) if (isGap(sorted[k])) rankSum += avgRank;
    i = j;
  }
  return (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
}

const thresholdsDesc = (cs: Case[]) => [...new Set(cs.map(c => c.prob))].sort((a, b) => b - a);

function rocCurve(cs: Case[]) {
  const pos = cs.filter(isGap).length;
  const neg = cs.length - pos;
  if (pos === 0 || neg === 0) return [];
  return [
    { threshold: null as number | null, fpr: 0, tpr: 0 },
    ...thresholdsDesc(cs).map(t => {
      const cm = confusion(cs, c => c.prob >= t);
      return { threshold: t, fpr: cm.fp / neg, tpr: cm.tp / pos };
    }),
  ];
}

// Area under the precision–recall curve, step-wise over all thresholds
function averagePrecision(cs: Case[]): number | null {
  const pos = cs.filter(isGap).length;
  if (pos === 0) return null;
  let ap = 0;
  let prevRecall = 0;
  for (const t of thresholdsDesc(cs)) {
    const cm = confusion(cs, c => c.prob >= t);
    const recall = cm.tp / pos;
    ap += (recall - prevRecall) * (cm.tp / (cm.tp + cm.fp));
    prevRecall = recall;
  }
  return ap;
}

// Brier score of the probability against the outcome 0/1, and of the constant forecast «probability = share of gaps»
function brier(cs: Case[]) {
  const y = (c: Case) => (isGap(c) ? 1 : 0);
  const score = mean(cs.map(c => (c.prob / 100 - y(c)) ** 2));
  const base = mean(cs.map(y));
  if (score === null || base === null) return { brier: null, base_rate: null, reference: null, skill: null };
  const reference = base * (1 - base);
  return { brier: score, base_rate: base, reference, skill: reference > 0 ? 1 - score / reference : null };
}

const CALIBRATION_BINS: Array<{ label: string; test: (p: number) => boolean }> = [
  { label: '0%', test: p => p === 0 },
  { label: '0–10%', test: p => p > 0 && p < 10 },
  { label: '10–30%', test: p => p >= 10 && p < 30 },
  { label: '30–50%', test: p => p >= 30 && p < 50 },
  { label: '50–100%', test: p => p >= 50 },
];
function calibration(cs: Case[]) {
  return CALIBRATION_BINS.map(({ label, test }) => {
    const bin = cs.filter(c => test(c.prob));
    const gaps = bin.filter(isGap).length;
    return {
      bin: label,
      forecasts: bin.length,
      gaps,
      mean_predicted: mean(bin.map(c => c.prob / 100)),
      observed: ratio(gaps, bin.length),
      observed_ci95: wilson(gaps, bin.length),
    };
  });
}

// Lead time: days from the forecast (cutoff) to the first negative day, for the gaps the status caught
function leadTime(cs: Case[], flagged: (c: Case) => boolean) {
  const detected = cs.filter(c => isGap(c) && flagged(c)).map(c => c.firstNegDay!).sort((a, b) => a - b);
  const missed = cs.filter(c => isGap(c) && !flagged(c)).map(c => c.firstNegDay!).sort((a, b) => a - b);
  const share = (d: number) => ratio(detected.filter(v => v >= d).length, detected.length);
  return {
    detected: detected.length,
    days: detected,
    median: detected.length > 0 ? quantile(detected, 0.5) : null,
    mean: mean(detected),
    min: detected[0] ?? null,
    max: detected.at(-1) ?? null,
    share_7_plus: share(7),
    share_14_plus: share(14),
    missed: missed.length,
    missed_days: missed,
    missed_median: missed.length > 0 ? quantile(missed, 0.5) : null,
  };
}

const gapEval = (cs: Case[]) => cs.filter(c => c.startBalance >= 0);

function summarize(cs: Case[]) {
  const gc = gapEval(cs);
  const newGaps = gc.filter(isGap).length;
  return {
    companies: new Set(cs.map(c => c.company)).size,
    forecasts: cs.length,
    actual_gaps: cs.filter(isGap).length,
    already_negative: cs.length - gc.length,
    gap_eval_forecasts: gc.length,
    new_gaps: newGaps,
    base_rate: ratio(newGaps, gc.length),
    regression: Object.fromEntries(MODELS.map(m => [m, regression(cs, m)])) as Record<Model, ReturnType<typeof regression>>,
    coverage_p10_p90: coverage(cs),
    status: Object.fromEntries(
      STATUS_KEYS.map(k => [k, { name: STATUS[k].name, threshold: STATUS[k].threshold, ...scores(confusion(gc, STATUS[k].flagged)) }])
    ) as Record<StatusKey, { name: string; threshold: number } & ReturnType<typeof scores>>,
    probability: { roc_auc: rocAuc(gc), average_precision: averagePrecision(gc), ...brier(gc), calibration: calibration(gc) },
    lead_time: Object.fromEntries(STATUS_KEYS.map(k => [k, leadTime(gc, STATUS[k].flagged)])) as Record<StatusKey, ReturnType<typeof leadTime>>,
  };
}

// Bootstrap over companies: the forecasts of one company are not independent, so whole companies are resampled
function bootstrap(cs: Case[], stat: (sample: Case[]) => number | null): [number, number] | null {
  const byCompany = new Map<string, Case[]>();
  for (const c of cs) {
    const list = byCompany.get(c.company);
    if (list) list.push(c);
    else byCompany.set(c.company, [c]);
  }
  const groups = [...byCompany.values()];
  let seed = 20241001;
  const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const values: number[] = [];
  for (let r = 0; r < BOOTSTRAP_ROUNDS; r++) {
    const sample: Case[] = [];
    for (let i = 0; i < groups.length; i++) sample.push(...groups[Math.floor(random() * groups.length)]);
    const v = stat(sample);
    if (v !== null && Number.isFinite(v)) values.push(v);
  }
  if (values.length < BOOTSTRAP_ROUNDS / 2) return null;
  values.sort((a, b) => a - b);
  return [quantile(values, 0.025), quantile(values, 0.975)];
}

const overall = summarize(cases);
const gapCases = gapEval(cases);
const byThreshold = THRESHOLDS.map(t => ({ threshold: t, ...scores(confusion(gapCases, c => c.prob >= t)) }));
const statusMismatches = Object.fromEntries(
  STATUS_KEYS.map(k => [k, gapCases.filter(c => STATUS[k].flagged(c) !== c.prob >= STATUS[k].threshold).length])
) as Record<StatusKey, number>;
const ci95 = {
  mae: bootstrap(cases, s => errorSums(s, 'analog').mae),
  rmse: bootstrap(cases, s => errorSums(s, 'analog').rmse),
  wape: bootstrap(cases, s => errorSums(s, 'analog').wape),
  coverage_p10_p90: bootstrap(cases, s => coverage(s).inside),
  roc_auc: bootstrap(gapCases, rocAuc),
  brier: bootstrap(gapCases, s => brier(s).brier),
  ...Object.fromEntries(
    STATUS_KEYS.flatMap(k => {
      const sc = (s: Case[]) => scores(confusion(s, STATUS[k].flagged));
      return [
        [`${k}_accuracy`, bootstrap(gapCases, s => sc(s).accuracy)],
        [`${k}_balanced_accuracy`, bootstrap(gapCases, s => sc(s).balanced_accuracy)],
        [`${k}_precision`, bootstrap(gapCases, s => sc(s).precision)],
        [`${k}_recall`, bootstrap(gapCases, s => sc(s).recall)],
        [`${k}_f1`, bootstrap(gapCases, s => sc(s).f1)],
      ];
    })
  ),
} as Record<string, [number, number] | null>;

const perCutoff = Object.fromEntries(
  CUTOFFS.map(cutoff => {
    const cs = cases.filter(c => c.cutoff === cutoff);
    const gc = gapEval(cs);
    const cm = confusion(gc, STATUS.warning.flagged);
    return [cutoff, {
      forecasts: cs.length,
      already_negative: cs.length - gc.length,
      new_gaps: gc.filter(isGap).length,
      warning_signals: cm.tp + cm.fp,
      warning_tp: cm.tp,
      mae_analog: cs.length > 0 ? errorSums(cs, 'analog').mae : null,
    }];
  })
);
const sources = [...new Set(cases.map(c => c.source))];
const bySource = sources.length > 1 ? Object.fromEntries(sources.map(s => [s, summarize(cases.filter(c => c.source === s))])) : null;
const actualFrom = addDaysIso(CUTOFFS[0], 1);
const actualTo = addDaysIso(CUTOFFS[CUTOFFS.length - 1], HORIZON_DAYS);

// ---------------------------------------------------------------- step-by-step log
const reg = overall.regression;
const st = overall.status;
const prob = overall.probability;
const totalSkipped = skipped.no_history + skipped.short_history + skipped.no_facts;
const log: CalcStep[] = [];

log.push({
  title: 'Чтение и очистка данных',
  lines: [
    ...readLines,
    `Компаний: ${companies.length}; данные с ${ruDate(datasetStart)} по ${ruDate(datasetEnd)}.`,
    `Удалено повторов строк, которые остаток не подтверждает: ${duplicatesRemoved}. Дней, где остаток не сходится с операциями: ${breakDays} — ` +
      'фактический остаток в эти дни всё равно берётся из колонки остатка на конец дня (balance_eod).',
  ],
});
log.push({
  title: 'Даты среза',
  lines: [
    `Срез — последний день месяца, после которого в данных есть ещё ${HORIZON_DAYS} дней факта, а до него — не меньше ${MIN_HISTORY_DAYS} дней истории. ` +
      `Берём последние ${CUTOFF_COUNT}: ${CUTOFFS.map(ruDate).join(', ')}.`,
    'Для каждой компании и даты среза прогноз строится функцией analyzeStatement — той же, что на сайте, — только по операциям до среза. ' +
      'Стартовый остаток — фактический остаток на дату среза.',
    `Прогноз на дни 1–${HORIZON_DAYS} после среза сравнивается с фактическим остатком на конец этих дней; факт — с ${ruDate(actualFrom)} по ${ruDate(actualTo)}.`,
  ],
});
log.push({
  title: 'Прогнозы',
  lines: [
    `Построено ${count(cases.length, 'прогноз', 'прогноза', 'прогнозов')} по ${count(overall.companies, 'компании', 'компаниям', 'компаниям')}. ` +
      `Пропущено ${totalSkipped}: нет операций до среза — ${skipped.no_history}, меньше ${MIN_HISTORY_DAYS} дней истории — ${skipped.short_history}, ` +
      `нет ${HORIZON_DAYS} дней факта после среза (выписка закончилась) — ${skipped.no_facts}.`,
    `Разрыв — ${GAP_DEFINITION}.`,
    ...CUTOFFS.map(cutoff => {
      const p = perCutoff[cutoff];
      return `Срез ${ruDate(cutoff)}: ${count(p.forecasts, 'прогноз', 'прогноза', 'прогнозов')}; остаток уже в минусе на дату среза — ${p.already_negative}; ` +
        `новый разрыв в следующие ${HORIZON_DAYS} дней — ${p.new_gaps}; статус «Повышенный риск» и выше — ${p.warning_signals}, из них разрыв случился у ${p.warning_tp}.`;
    }),
  ],
});

// Worked examples: a caught gap with a typical lead time and the closest miss
const lt = overall.lead_time.warning;
const caught = gapCases.filter(c => isGap(c) && STATUS.warning.flagged(c));
const example = lt.median !== null ? [...caught].sort((a, b) => Math.abs(a.firstNegDay! - lt.median!) - Math.abs(b.firstNegDay! - lt.median!) || b.prob - a.prob)[0] : undefined;
const nearMiss = gapCases.filter(c => isGap(c) && !STATUS.warning.flagged(c)).sort((a, b) => b.prob - a.prob)[0];
const caseMae = (c: Case) => sum(c.p50.map((p, h) => Math.abs(p - c.facts[h]))) / HORIZON_DAYS;
const inBand = (c: Case) => c.facts.filter((f, h) => f >= c.p10[h] && f <= c.p90[h]).length;
const STATUS_RU: Record<RiskLevel, string> = { STABLE: 'стабильно', WARNING: 'повышенный риск', CRITICAL: 'критический' };
const exampleLines: string[] = [];
for (const [label, c] of [['Найденный разрыв', example], ['Ближайший пропуск', nearMiss]] as const) {
  if (!c) continue;
  const d = c.firstNegDay! - 1;
  exampleLines.push(
    `${label}: ${c.company}, срез ${ruDate(c.cutoff)}. Стартовый остаток ${rub(c.startBalance)}; ниже нуля ушли ${c.gapPaths} из ${c.scenarios} сценариев → ` +
      `вероятность ${fixed(c.prob, 1)}%, статус «${STATUS_RU[c.status]}».`,
    `Факт: остаток впервые ушёл в минус ${ruDate(addDaysIso(c.cutoff, c.firstNegDay!))} — через ${count(c.firstNegDay!, 'день', 'дня', 'дней')} после прогноза: ${rub(c.facts[d])}; ` +
      `прогноз на этот день: P10 ${rub(c.p10[d])}, P50 ${rub(c.p50[d])}, P90 ${rub(c.p90[d])}.`,
    `Ошибка P50 этого прогноза за ${HORIZON_DAYS} дней: MAE ${rub(caseMae(c))}; факт внутри коридора P10–P90 в ${inBand(c)} из ${HORIZON_DAYS} дней.`
  );
}
if (exampleLines.length > 0) log.push({ title: 'Пример: прогноз против факта', lines: exampleLines });

const a = reg.analog;
const cov = overall.coverage_p10_p90;
const better = (m: Model) => 1 - a.mae / reg[m].mae;
log.push({
  title: 'Точность прогноза остатка (медиана P50)',
  lines: [
    `Точек сравнения: ${count(cases.length, 'прогноз', 'прогноза', 'прогнозов')} × ${HORIZON_DAYS} дней = ${a.points.toLocaleString('ru-RU')}. Ошибка = прогноз P50 − факт.`,
    `MAE = Σ|ошибка| / ${a.points.toLocaleString('ru-RU')} = ${rub(a.sum_abs_error)} / ${a.points.toLocaleString('ru-RU')} = ${rub(a.mae)}${ci(ci95.mae, rub)}.`,
    `MSE = Σ(ошибка)² / ${a.points.toLocaleString('ru-RU')} = ${sci(a.mse)} ₽² — в квадратных рублях, поэтому читаем RMSE = √MSE = ${rub(a.rmse)}${ci(ci95.rmse, rub)}.`,
    `WAPE = Σ|ошибка| / Σ|факт| = ${rub(a.sum_abs_error)} / ${rub(a.sum_abs_fact)} = ${pct(a.wape)}${ci(ci95.wape, v => pct(v))} — ошибка в процентах от остатка; точность прогноза остатка 1 − WAPE = ${pct(a.wape === null ? null : 1 - a.wape)}.`,
    `Смещение (bias) = Σ(ошибка) / ${a.points.toLocaleString('ru-RU')} = ${rub(a.bias)} (плюс — прогноз в среднем выше факта). Медианная абсолютная ошибка — ${rub(a.median_ae)}: половина ошибок меньше.`,
    `Коридор P10–P90: факт внутри в ${pct(cov.inside)} точек${ci(ci95.coverage_p10_p90, v => pct(v))} (для честного коридора ≈ 80%), ниже P10 — ${pct(cov.below_p10)}, выше P90 — ${pct(cov.above_p90)}.`,
    `Для сравнения: MAE наивной модели («остаток не меняется») — ${rub(reg.naive.mae)}, линейного тренда за 90 дней — ${rub(reg.trend.mae)}. ` +
      `Аналоговые периоды точнее наивной на ${pct(better('naive'))} и тренда на ${pct(better('trend'))}.`,
    `По горизонту (MAE / RMSE): ${[1, 7, 14, 30].filter(h => h <= HORIZON_DAYS).map(h => `день ${h} — ${rub(a.mae_by_horizon[h - 1])} / ${rub(a.rmse_by_horizon[h - 1])}`).join('; ')}.`,
  ],
});

const g = overall.gap_eval_forecasts;
const f3 = (v: number) => fixed(v);
const classLines = [
  `Считаем ${GAP_EVAL_RULE}: ${g} из ${cases.length}. Разрыв случился в ${overall.new_gaps} из них — доля ${pct(overall.base_rate)}.`,
  `Модель «разрыва не будет никогда» получила бы accuracy ${pct(overall.base_rate === null ? null : 1 - overall.base_rate)}, не найдя ни одного разрыва. ` +
    'Поэтому кроме accuracy смотрим recall, precision, F1 и balanced accuracy.',
];
for (const k of STATUS_KEYS) {
  const s = st[k];
  classLines.push(
    `Статус ${s.name} (вероятность ≥ ${s.threshold}%): TP = ${s.tp} (предсказан и случился), FP = ${s.fp} (ложная тревога), ` +
      `FN = ${s.fn} (пропуск), TN = ${s.tn} (спокойный прогноз и разрыва нет).`,
    `  Accuracy = (TP + TN) / всего = (${s.tp} + ${s.tn}) / ${s.n} = ${fixed(s.accuracy)}${ci(ci95[`${k}_accuracy`], f3)}.`,
    `  Precision = TP / (TP + FP) = ${s.tp} / ${s.tp + s.fp} = ${fixed(s.precision)}${ci(ci95[`${k}_precision`], f3)} — доля верных сигналов.`,
    `  Recall = TP / (TP + FN) = ${s.tp} / ${s.tp + s.fn} = ${fixed(s.recall)}${ci(ci95[`${k}_recall`], f3)} — доля найденных разрывов.`,
    `  Specificity = TN / (TN + FP) = ${s.tn} / ${s.tn + s.fp} = ${fixed(s.specificity)} — доля спокойных месяцев без ложной тревоги.`,
    `  F1 = 2 · Precision · Recall / (Precision + Recall) = 2·TP / (2·TP + FP + FN) = ${2 * s.tp} / ${2 * s.tp + s.fp + s.fn} = ${fixed(s.f1)}${ci(ci95[`${k}_f1`], f3)}.`,
    `  Balanced accuracy = (Recall + Specificity) / 2 = ${fixed(s.balanced_accuracy)}${ci(ci95[`${k}_balanced_accuracy`], f3)}.`
  );
}
classLines.push(
  statusMismatches.warning + statusMismatches.critical === 0
    ? `Статус сайта во всех ${g} прогнозах совпал с порогом вероятности 10% / 50%: «P10 ниже нуля» означает, что ниже нуля уже не меньше 10% сценариев.`
    : `Статус сайта разошёлся с порогом вероятности в ${statusMismatches.warning + statusMismatches.critical} прогнозах: матрицы выше построены по статусу сайта.`
);
log.push({ title: 'Обнаружение кассового разрыва: матрица ошибок и метрики', lines: classLines });

log.push({
  title: 'Качество вероятности разрыва',
  lines: [
    `Brier score = Σ(p − факт)² / ${g} = ${fixed(prob.brier, 4)}${ci(ci95.brier, v => fixed(v, 4))}, где p — вероятность от 0 до 1, факт — 1, если разрыв был, иначе 0. Меньше — лучше.`,
    `Эталон — постоянный прогноз «вероятность = доля разрывов ${pct(prob.base_rate)}»: Brier = b·(1 − b) = ${fixed(prob.reference, 4)}. ` +
      `Улучшение над эталоном (Brier skill score) = 1 − ${fixed(prob.brier, 4)} / ${fixed(prob.reference, 4)} = ${fixed(prob.skill)}.`,
    `ROC-AUC = ${fixed(prob.roc_auc)}${ci(ci95.roc_auc, f3)}: с такой вероятностью прогноз, после которого разрыв случился, получил вероятность выше, ` +
      'чем прогноз без разрыва (0,5 — угадывание, 1 — идеально).',
    `Average precision (площадь под кривой precision–recall) = ${fixed(prob.average_precision)}; у случайного угадывания она равна доле разрывов ${pct(prob.base_rate)}.`,
    'Калибровка — совпадает ли обещанная вероятность с тем, как часто разрыв случался:',
    ...prob.calibration.map(b =>
      b.forecasts === 0
        ? `  Вероятность ${b.bin}: прогнозов нет.`
        : `  Вероятность ${b.bin}: ${count(b.forecasts, 'прогноз', 'прогноза', 'прогнозов')}, в среднем обещано ${pct(b.mean_predicted)}, ` +
          `разрыв случился в ${b.gaps} — ${pct(b.observed)}${ci(b.observed_ci95, v => pct(v))}.`
    ),
  ],
});

const leadLines = ['Для найденных разрывов — сколько дней прошло от прогноза (даты среза) до первого дня с отрицательным остатком.'];
for (const k of STATUS_KEYS) {
  const l = overall.lead_time[k];
  leadLines.push(
    l.detected === 0
      ? `Статус ${STATUS[k].name}: найденных разрывов нет.`
      : `Статус ${STATUS[k].name}: найдено ${l.detected} из ${overall.new_gaps} разрывов; предупреждение пришло за ${l.min}–${l.max} дней, ` +
          `медиана ${fixed(l.median, 1)} дня, среднее ${fixed(l.mean, 1)}; за 7 и более дней — ${pct(l.share_7_plus, 0)}, за 14 и более — ${pct(l.share_14_plus, 0)}. ` +
          `Дни до разрыва: ${l.days.join(', ')}.`
  );
}
if (overall.lead_time.warning.missed > 0) {
  leadLines.push(`Пропущенные разрывы (FN при статусе «Повышенный риск»): ${overall.lead_time.warning.missed}, дни до разрыва: ${overall.lead_time.warning.missed_days.join(', ')}.`);
}
log.push({ title: 'Заблаговременность предупреждения', lines: leadLines });

log.push({
  title: 'Доверительные интервалы',
  lines: [
    'Для долей (accuracy, precision, recall, калибровка) рядом с числом в таблицах — 95% интервал Уилсона.',
    `В этом журнале «95% ДИ» — бутстрэп по компаниям: ${BOOTSTRAP_ROUNDS} раз выбираем ${overall.companies} компаний с возвращением, пересчитываем метрику, ` +
      'границы — 2,5% и 97,5% перцентили. Прогнозы одной компании зависимы, поэтому перевыбираются компании целиком.',
  ],
});

if (bySource) {
  log.push({
    title: 'Сравнение по файлам',
    lines: sources.map(s => {
      const b = bySource[s];
      const w = b.status.warning;
      return `«${s}»: ${b.companies} компаний, ${count(b.forecasts, 'прогноз', 'прогноза', 'прогнозов')}, новых разрывов ${b.new_gaps}; ` +
        `статус «Повышенный риск»: accuracy ${fixed(w.accuracy)}, precision ${fixed(w.precision)}, recall ${fixed(w.recall)}, F1 ${fixed(w.f1)}; ` +
        `ROC-AUC ${fixed(b.probability.roc_auc)}, Brier ${fixed(b.probability.brier, 4)}; MAE ${rub(b.regression.analog.mae)}, RMSE ${rub(b.regression.analog.rmse)}, ` +
        `WAPE ${pct(b.regression.analog.wape)}, коридор P10–P90 ${pct(b.coverage_p10_p90.inside)}.`;
    }),
  });
}

// ---------------------------------------------------------------- output
const result = {
  generated_at: new Date().toISOString(),
  description: 'Бэктест 30-дневного прогноза остатка: прогноз строится по данным до даты среза и сравнивается с фактом следующих 30 дней',
  files: files.map(f => path.basename(f)),
  data_period: { from: datasetStart, to: datasetEnd },
  cutoffs: CUTOFFS,
  horizon_days: HORIZON_DAYS,
  actual_period: { from: actualFrom, to: actualTo },
  skipped,
  gap_definition: GAP_DEFINITION,
  gap_eval_rule: GAP_EVAL_RULE,
  ...overall,
  classification: { analog: { name: MODEL_NAMES.analog, by_threshold: byThreshold } },
  roc_curve: rocCurve(gapCases),
  ci95_bootstrap: ci95,
  bootstrap_rounds: BOOTSTRAP_ROUNDS,
  status_mismatches: statusMismatches,
  per_cutoff: perCutoff,
  by_source: bySource,
  log,
};

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(OUT_FILE, JSON.stringify(result, null, 2));
const casesFile = path.join(OUT_DIR, 'backtest_cases.csv');
writeCsv(
  casesFile,
  ['company_id', 'file', 'cutoff', 'start_balance', 'gap_probability_pct', 'status', 'min_p10', 'min_p50', 'actual_gap', 'first_negative_date',
    'days_to_gap', 'min_actual_balance', 'mae_p50', 'days_inside_p10_p90', 'in_gap_metrics'],
  cases.map(c => [
    c.company, c.source, c.cutoff, Math.round(c.startBalance), c.prob, STATUS_RU[c.status], c.minP10, c.minP50, isGap(c) ? 1 : 0,
    isGap(c) ? addDaysIso(c.cutoff, c.firstNegDay!) : null, c.firstNegDay, Math.round(Math.min(...c.facts)), Math.round(caseMae(c)), inBand(c),
    c.startBalance >= 0 ? 1 : 0,
  ])
);
const logFile = path.join(OUT_DIR, 'backtest_log.md');
fs.writeFileSync(logFile, [
  '# Журнал расчёта метрик качества (бэктест)',
  '',
  `Файлы: ${result.files.join(', ')}. Каждый прогноз против факта — в backtest_cases.csv, все числа — в ${path.basename(OUT_FILE)}.`,
  '',
  ...stepsToMarkdown(log),
].join('\n'));

const w = st.warning;
const c = st.critical;
console.log(`Компаний: ${overall.companies}, прогнозов: ${cases.length} (пропущено ${totalSkipped}), срезы: ${CUTOFFS.map(ruDate).join(', ')}`);
console.log(`Остаток (P50): MAE ${rub(a.mae)}, MSE ${sci(a.mse)} ₽², RMSE ${rub(a.rmse)}, WAPE ${pct(a.wape)}, факт внутри P10–P90: ${pct(cov.inside)}`);
console.log(`  для сравнения MAE: наивная ${rub(reg.naive.mae)}, тренд ${rub(reg.trend.mae)}`);
console.log(`Разрыв: ${count(g, 'прогноз', 'прогноза', 'прогнозов')} до разрыва, новых разрывов ${overall.new_gaps} (${pct(overall.base_rate)}); уже в минусе на срезе: ${overall.already_negative}`);
for (const s of [w, c]) {
  console.log(`  ${s.name} (≥${s.threshold}%): accuracy ${fixed(s.accuracy)}, balanced ${fixed(s.balanced_accuracy)}, precision ${fixed(s.precision)}, ` +
    `recall ${fixed(s.recall)}, F1 ${fixed(s.f1)}, specificity ${fixed(s.specificity)} (TP ${s.tp}, FP ${s.fp}, FN ${s.fn}, TN ${s.tn})`);
}
console.log(`Вероятность: ROC-AUC ${fixed(prob.roc_auc)}, AP ${fixed(prob.average_precision)}, Brier ${fixed(prob.brier, 4)} (эталон ${fixed(prob.reference, 4)}, skill ${fixed(prob.skill)})`);
console.log(`Заблаговременность («Повышенный риск»): медиана ${fixed(lt.median, 1)} дн., от ${lt.min ?? '—'} до ${lt.max ?? '—'}`);
console.log(`Сохранено: ${OUT_FILE}, ${casesFile}, ${logFile}`);
