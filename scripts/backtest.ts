// Backtest of the cash-flow forecast on the case dataset.
//
// For every company and every cutoff date the forecast is built only from operations up to the
// cutoff and compared with the actual end-of-day balances of the next 30 days:
//   - balance accuracy (MAE / RMSE by horizon day) for our model and three baselines;
//   - cash-gap detection (precision / recall by probability threshold), where an actual gap is
//     a negative end-of-day balance on any day of the horizon. Only forecasts made before the gap
//     count: if the balance is already negative at the cutoff, the gap is not a prediction.
//
// Run:  node scripts/backtest.ts [--out metrics.json] [statement.csv ...]   (default: data/raw/train.csv data/raw/test.csv)
// Output: reports/backtest_metrics.json or the --out path (plotted by scripts/plot_backtest.py).
// Cutoffs are the last month ends that still have HORIZON_DAYS of facts in the data, so any file in the case
// format works; scripts/evaluate.ts runs the backtest and the charts for a new file in one command.

import fs from 'fs';
import path from 'path';
import { buildDailySeries, simulateForecast, addDaysIso, HORIZON_DAYS } from '../shared/cashflow.ts';
import type { StatementOp } from '../shared/cashflow.ts';

const CUTOFF_COUNT = 6;
const MIN_HISTORY_DAYS = 90;
const THRESHOLDS = Array.from({ length: 19 }, (_, i) => (i + 1) * 5); // 5% .. 95%
const GAP_EVAL_RULE = 'только прогнозы с неотрицательным остатком на дату среза (разрыв ещё не наступил)';
const args = process.argv.slice(2);
const outAt = args.indexOf('--out');
const OUT_FILE = outAt >= 0 ? args[outAt + 1] : 'reports/backtest_metrics.json';
const inputFiles = outAt >= 0 ? args.filter((_, i) => i !== outAt && i !== outAt + 1) : args;
const STANDARD_COLUMNS = ['company_id', 'operation_id', 'operation_date', 'amount', 'direction', 'counterparty_inn', 'counterparty_name', 'payment_purpose', 'balance_eod'];

interface Row extends StatementOp {
  company: string;
  eod: number;
}

function normalizeDate(raw: string): string {
  const s = raw.trim();
  const ru = s.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})$/);
  return ru ? `${ru[3]}-${ru[2].padStart(2, '0')}-${ru[1].padStart(2, '0')}` : s.slice(0, 10);
}

function loadRows(files: string[]): Map<string, Row[]> {
  const byCompany = new Map<string, Row[]>();
  for (const file of files) {
    const lines = fs.readFileSync(file, 'utf8').replace(/^﻿/, '').split(/\r?\n/).filter(Boolean);
    // test.csv of the case comes without a header row: fall back to the standard column order
    const hasHeader = lines[0].includes('company_id');
    const header = hasHeader ? lines[0].split(';') : STANDARD_COLUMNS;
    const col = (name: string) => header.indexOf(name);
    const [iC, iD, iA, iDir, iName, iPurp, iBal] = ['company_id', 'operation_date', 'amount', 'direction', 'counterparty_name', 'payment_purpose', 'balance_eod'].map(col);
    for (const line of hasHeader ? lines.slice(1) : lines) {
      const c = line.split(';');
      const amount = Math.abs(parseFloat(c[iA]));
      const eod = parseFloat(c[iBal]);
      if (!c[iC] || !Number.isFinite(amount) || !Number.isFinite(eod)) continue;
      const row: Row = {
        company: c[iC],
        date: normalizeDate(c[iD]),
        amount,
        isCredit: c[iDir] === 'Поступление',
        category: '',
        counterparty: c[iName] ?? '',
        purpose: c[iPurp] ?? '',
        eod
      };
      const list = byCompany.get(row.company);
      if (list) list.push(row);
      else byCompany.set(row.company, [row]);
    }
  }
  for (const list of byCompany.values()) list.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return byCompany;
}

// Actual end-of-day balance for every calendar day (carried forward over days without operations).
// A statement that ends a few days before the dataset end is extended to it: no operations = same balance.
function actualEodByDay(rows: Row[], datasetEnd: string): Map<string, number> {
  const map = new Map<string, number>();
  for (const r of rows) map.set(r.date, r.eod);
  const out = new Map<string, number>();
  let last = rows[0].eod;
  const lastOp = rows[rows.length - 1].date;
  const end = addDaysIso(lastOp, 7) >= datasetEnd ? datasetEnd : lastOp;
  for (let d = rows[0].date; d <= end; d = addDaysIso(d, 1)) {
    if (map.has(d)) last = map.get(d)!;
    out.set(d, last);
  }
  return out;
}

const MODELS = ['analog', 'naive', 'trend'] as const;
type Model = (typeof MODELS)[number];
const MODEL_NAMES: Record<Model, string> = {
  analog: 'Аналоговые периоды (текущая модель)',
  naive: 'Наивная: остаток не меняется',
  trend: 'Линейный тренд за 90 дней'
};

// The last `count` month ends that have HORIZON_DAYS of facts after them and at least MIN_HISTORY_DAYS of data before
function monthEndCutoffs(datasetStart: string, datasetEnd: string, count: number): string[] {
  const cutoffs: string[] = [];
  let year = Number(datasetEnd.slice(0, 4));
  let month = Number(datasetEnd.slice(5, 7)); // 1..12
  while (cutoffs.length < count) {
    const monthEnd = new Date(Date.UTC(year, month, 0)).toISOString().slice(0, 10);
    if (monthEnd < addDaysIso(datasetStart, MIN_HISTORY_DAYS)) break;
    if (addDaysIso(monthEnd, HORIZON_DAYS) <= datasetEnd) cutoffs.unshift(monthEnd);
    month -= 1;
    if (month === 0) { month = 12; year -= 1; }
  }
  return cutoffs;
}

const files = inputFiles.length > 0 ? inputFiles : ['data/raw/train.csv', 'data/raw/test.csv'];
const companies = loadRows(files);
if (companies.size === 0) {
  console.error('В файлах нет ни одной операции в формате кейса (company_id;operation_id;operation_date;amount;direction;...)');
  process.exit(1);
}
const datasetStart = [...companies.values()].map(rows => rows[0].date).sort()[0];
const datasetEnd = [...companies.values()].map(rows => rows[rows.length - 1].date).sort().at(-1)!;
const CUTOFFS = monthEndCutoffs(datasetStart, datasetEnd, CUTOFF_COUNT);

// Errors per model and horizon day; scores and labels for gap detection
const absErr: Record<Model, number[][]> = { analog: [], naive: [], trend: [] };
const sqErr: Record<Model, number[][]> = { analog: [], naive: [], trend: [] };
for (const m of MODELS) for (let h = 0; h < HORIZON_DAYS; h++) { absErr[m].push([]); sqErr[m].push([]); }
const cases: Array<{ company: string; cutoff: string; actualGap: boolean; startNegative: boolean; analogProb: number }> = [];
const perCutoff: Record<string, { forecasts: number; gaps: number; maeAnalog: number[] }> = {};
let skipped = 0;
let actualFrom = '9999-12-31';
let actualTo = '0000-01-01';

for (const [company, rows] of companies) {
  const actual = actualEodByDay(rows, datasetEnd);
  const opening = rows[0].eod - rows.filter(r => r.date === rows[0].date).reduce((s, r) => s + (r.isCredit ? r.amount : -r.amount), 0);
  for (const cutoff of CUTOFFS) {
    const hist = rows.filter(r => r.date <= cutoff);
    const horizon = Array.from({ length: HORIZON_DAYS }, (_, i) => addDaysIso(cutoff, i + 1));
    if (hist.length === 0 || !actual.has(horizon[HORIZON_DAYS - 1])) { skipped++; continue; }
    const days = buildDailySeries(hist, opening);
    // Pad days without operations up to the cutoff so the forecast starts on cutoff + 1, aligned with the facts
    while (days.length > 0 && days[days.length - 1].date < cutoff) {
      const last = days[days.length - 1];
      days.push({ date: addDaysIso(last.date, 1), inflow: 0, outflow: 0, sched: 0, schedNames: [], ownerOut: 0, eod: last.eod });
    }
    if (days.length < MIN_HISTORY_DAYS) { skipped++; continue; }
    // Start from the actual balance at the cutoff, exactly as the service does with closingBalance
    const balance = actual.get(cutoff) ?? days[days.length - 1].eod;
    const sim = simulateForecast(days, balance);
    const net90 = days.slice(-90).reduce((s, d) => s + d.inflow - d.outflow, 0) / Math.min(90, days.length);

    const facts = horizon.map(d => actual.get(d)!);
    const preds: Record<Model, number[]> = {
      analog: sim.forecast.map(p => p.balance_p50),
      naive: horizon.map(() => balance),
      trend: horizon.map((_, i) => balance + net90 * (i + 1))
    };
    for (const m of MODELS) {
      preds[m].forEach((p, h) => {
        const e = p - facts[h];
        absErr[m][h].push(Math.abs(e));
        sqErr[m][h].push(e * e);
      });
    }
    const actualGap = facts.some(v => v < 0);
    cases.push({ company, cutoff, actualGap, startNegative: balance < 0, analogProb: sim.gapProbabilityPct });
    perCutoff[cutoff] ??= { forecasts: 0, gaps: 0, maeAnalog: [] };
    perCutoff[cutoff].forecasts++;
    if (actualGap) perCutoff[cutoff].gaps++;
    perCutoff[cutoff].maeAnalog.push(preds.analog.reduce((s, p, h) => s + Math.abs(p - facts[h]), 0) / HORIZON_DAYS);
    if (horizon[0] < actualFrom) actualFrom = horizon[0];
    if (horizon[HORIZON_DAYS - 1] > actualTo) actualTo = horizon[HORIZON_DAYS - 1];
  }
}

if (cases.length === 0) {
  console.error(
    `Ни одного прогноза: данные ${datasetStart} .. ${datasetEnd}, даты среза: ${CUTOFFS.join(', ') || 'нет'}. ` +
    `Нужно не меньше ${MIN_HISTORY_DAYS} дней истории до среза и ${HORIZON_DAYS} дней факта после него.`
  );
  process.exit(1);
}

const mean =(xs: number[]) => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
const regression = Object.fromEntries(
  MODELS.map(m => [m, {
    name: MODEL_NAMES[m],
    mae_by_horizon: absErr[m].map(mean),
    rmse_by_horizon: sqErr[m].map(xs => Math.sqrt(mean(xs))),
    mae: mean(absErr[m].flat()),
    rmse: Math.sqrt(mean(sqErr[m].flat()))
  }])
);

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

// Gap detection is scored only on forecasts made before the gap
const gapCases = cases.filter(c => !c.startNegative);

function prMetrics(score: (c: (typeof cases)[number]) => number) {
  return THRESHOLDS.map(t => {
    const tp = gapCases.filter(c => c.actualGap && score(c) >= t).length;
    const fp = gapCases.filter(c => !c.actualGap && score(c) >= t).length;
    const fn = gapCases.filter(c => c.actualGap && score(c) < t).length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : null;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision !== null && precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { threshold: t, tp, fp, fn, signals: tp + fp, precision, precision_ci95: wilson(tp, tp + fp), recall, f1 };
  });
}

const result = {
  generated_at: new Date().toISOString(),
  description: 'Бэктест 30-дневного прогноза остатка: прогноз строится по данным до даты среза и сравнивается с фактом следующих 30 дней',
  cutoffs: CUTOFFS,
  horizon_days: HORIZON_DAYS,
  actual_period: { from: actualFrom, to: actualTo },
  companies: new Set(cases.map(c => c.company)).size,
  forecasts: cases.length,
  skipped,
  gap_definition: 'остаток на конец дня ниже 0 ₽ хотя бы в один день горизонта',
  actual_gaps: cases.filter(c => c.actualGap).length,
  gap_eval_rule: GAP_EVAL_RULE,
  already_negative: cases.length - gapCases.length,
  gap_eval_forecasts: gapCases.length,
  new_gaps: gapCases.filter(c => c.actualGap).length,
  regression,
  classification: {
    analog: { name: MODEL_NAMES.analog, by_threshold: prMetrics(c => c.analogProb) }
  },
  per_cutoff: Object.fromEntries(Object.entries(perCutoff).map(([k, v]) => [k, { forecasts: v.forecasts, gaps: v.gaps, mae_analog: mean(v.maeAnalog) }]))
};

fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
fs.writeFileSync(OUT_FILE, JSON.stringify(result, null, 2));

const fmt = (v: number) => Math.round(v).toLocaleString('ru-RU');
console.log(`Компаний: ${result.companies}, прогнозов: ${result.forecasts}, пропущено: ${skipped}, фактических разрывов: ${result.actual_gaps}`);
console.log(`Факт: ${actualFrom} .. ${actualTo}`);
console.log(`Метрики разрыва: ${result.gap_eval_forecasts} прогнозов до разрыва, новых разрывов ${result.new_gaps}; исключено уже отрицательных на срезе: ${result.already_negative}`);
for (const m of MODELS) console.log(`${MODEL_NAMES[m].padEnd(38)} MAE ${fmt(regression[m].mae).padStart(12)}  RMSE ${fmt(regression[m].rmse).padStart(12)}`);
for (const t of [10, 20, 30, 50]) {
  const a = result.classification.analog.by_threshold.find(x => x.threshold === t)!;
  const ci = a.precision_ci95 ? ` [${a.precision_ci95[0].toFixed(2)}; ${a.precision_ci95[1].toFixed(2)}]` : '';
  console.log(`порог ${t}%: precision ${a.precision?.toFixed(2) ?? '—'}${ci}, recall ${a.recall.toFixed(2)} (TP ${a.tp}, FP ${a.fp}, FN ${a.fn})`);
}
console.log(`Сохранено: ${OUT_FILE}`);
