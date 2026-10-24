// Backtest of the cash-flow forecast on the case dataset.
//
// For every company and every cutoff date the forecast is built only from operations up to the
// cutoff and compared with the actual end-of-day balances of the next 30 days:
//   - balance accuracy (MAE / RMSE by horizon day) for our model and three baselines;
//   - cash-gap detection (precision / recall by probability threshold), where an actual gap is
//     a negative end-of-day balance on any day of the horizon. Only forecasts made before the gap
//     count: if the balance is already negative at the cutoff, the gap is not a prediction.
//
// Run:  node scripts/backtest.ts [statement.csv ...]   (default: data/raw/train.csv data/raw/test.csv)
// Output: reports/backtest_metrics.json (plotted by scripts/plot_backtest.py)

import fs from 'fs';
import path from 'path';
import { buildDailySeries, simulateForecast, addDaysIso, HORIZON_DAYS } from '../shared/cashflow.ts';
import type { StatementOp } from '../shared/cashflow.ts';

const CUTOFFS = ['2026-03-31', '2026-04-30', '2026-05-31', '2026-06-30', '2026-07-31', '2026-08-31'];
const MIN_HISTORY_DAYS = 90;
const THRESHOLDS = Array.from({ length: 19 }, (_, i) => (i + 1) * 5); // 5% .. 95%
const GAP_EVAL_RULE = 'только прогнозы с неотрицательным остатком на дату среза (разрыв ещё не наступил)';
const OUT_FILE = 'reports/backtest_metrics.json';
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

// The forecast the service used before this model: sine/cosine around turnover per operation
function legacySyntheticForecast(hist: Row[], balance: number, cutoff: string) {
  const totalIn = hist.filter(r => r.isCredit).reduce((s, r) => s + r.amount, 0);
  const totalOut = hist.filter(r => !r.isCredit).reduce((s, r) => s + r.amount, 0);
  const a = (totalIn + totalOut) / (Math.max(1, hist.length) * 1.5);
  const balances: number[] = [];
  let cum = balance;
  let maxProb = 0;
  for (let i = 1; i <= HORIZON_DAYS; i++) {
    const d = new Date(addDaysIso(cutoff, i) + 'T00:00:00Z');
    const weekend = d.getUTCDay() === 0 || d.getUTCDay() === 6;
    const dom = d.getUTCDate();
    const predIn = weekend ? 0 : a * (0.85 + Math.sin(i * 0.7) * 0.25);
    let predOut = weekend ? 0 : a * (0.8 + Math.cos(i * 0.7) * 0.2);
    if (!weekend) {
      if (dom === 10) predOut += a * 3.5;
      else if (dom === 25) predOut += a * 2.8;
      else if (dom === 28) predOut += a * 4.5;
      else if (dom <= 3) predOut += a * 1.8;
    }
    cum += predIn - predOut;
    balances.push(cum);
    const spread = Math.sqrt(i) * (a * 0.75 + 15000);
    const p10 = cum - spread * 1.28;
    let prob = 5;
    if (cum < 0) prob = Math.min(98, Math.round(75 + (Math.abs(cum) / (spread + 1)) * 20));
    else if (p10 < 0) prob = Math.min(68, Math.round((-p10 / (spread * 2.56 + 1)) * 60 + 10));
    maxProb = Math.max(maxProb, prob);
  }
  return { balances, gapProb: maxProb };
}

const MODELS = ['analog', 'legacy', 'naive', 'trend'] as const;
type Model = (typeof MODELS)[number];
const MODEL_NAMES: Record<Model, string> = {
  analog: 'Аналоговые периоды (текущая модель)',
  legacy: 'Синтетика sin/cos (прежняя версия)',
  naive: 'Наивная: остаток не меняется',
  trend: 'Линейный тренд за 90 дней'
};

const files = process.argv.slice(2).length > 0 ? process.argv.slice(2) : ['data/raw/train.csv', 'data/raw/test.csv'];
const companies = loadRows(files);
const datasetEnd = [...companies.values()].map(rows => rows[rows.length - 1].date).sort().at(-1)!;

// Errors per model and horizon day; scores and labels for gap detection
const absErr: Record<Model, number[][]> = { analog: [], legacy: [], naive: [], trend: [] };
const sqErr: Record<Model, number[][]> = { analog: [], legacy: [], naive: [], trend: [] };
for (const m of MODELS) for (let h = 0; h < HORIZON_DAYS; h++) { absErr[m].push([]); sqErr[m].push([]); }
const cases: Array<{ company: string; cutoff: string; actualGap: boolean; startNegative: boolean; analogProb: number; legacyProb: number }> = [];
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
    const legacy = legacySyntheticForecast(hist, balance, cutoff);
    const net90 = days.slice(-90).reduce((s, d) => s + d.inflow - d.outflow, 0) / Math.min(90, days.length);

    const facts = horizon.map(d => actual.get(d)!);
    const preds: Record<Model, number[]> = {
      analog: sim.forecast.map(p => p.balance_p50),
      legacy: legacy.balances,
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
    cases.push({ company, cutoff, actualGap, startNegative: balance < 0, analogProb: sim.gapProbabilityPct, legacyProb: legacy.gapProb });
    perCutoff[cutoff] ??= { forecasts: 0, gaps: 0, maeAnalog: [] };
    perCutoff[cutoff].forecasts++;
    if (actualGap) perCutoff[cutoff].gaps++;
    perCutoff[cutoff].maeAnalog.push(preds.analog.reduce((s, p, h) => s + Math.abs(p - facts[h]), 0) / HORIZON_DAYS);
    if (horizon[0] < actualFrom) actualFrom = horizon[0];
    if (horizon[HORIZON_DAYS - 1] > actualTo) actualTo = horizon[HORIZON_DAYS - 1];
  }
}

const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
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
    analog: { name: MODEL_NAMES.analog, by_threshold: prMetrics(c => c.analogProb) },
    legacy: { name: MODEL_NAMES.legacy, by_threshold: prMetrics(c => c.legacyProb) }
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
