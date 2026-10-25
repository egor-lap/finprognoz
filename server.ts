import express from 'express';
import type { Request, Response, NextFunction } from 'express';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { analyzeStatement, explainForecast } from './shared/cashflow.ts';
import { categorizeOperation } from './shared/statement.ts';
import type { StatementOp } from './shared/cashflow.ts';

dotenv.config({ quiet: true });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
// --production lets 'npm start' run in production mode on any OS without setting NODE_ENV
const IS_PROD = process.env.NODE_ENV === 'production' || process.argv.includes('--production');

// Server-only data (demo statement). Not under public/, so it is never served as a static file.
const DATA_DIR = path.join(__dirname, 'data');

// Basic authentication: credentials come from AUTH_USER / AUTH_PASSWORD.
const AUTH_USER = process.env.AUTH_USER || '';
const AUTH_PASSWORD = process.env.AUTH_PASSWORD || '';
const AUTH_ENABLED = Boolean(AUTH_USER && AUTH_PASSWORD);

if (!AUTH_ENABLED && IS_PROD) {
  console.error('[auth] AUTH_USER и AUTH_PASSWORD обязательны в production. Сервер не запущен.');
  process.exit(1);
}
if (!AUTH_ENABLED) {
  console.warn('[auth] AUTH_USER/AUTH_PASSWORD не заданы: dev-режим без аутентификации');
}

// Compare fixed-length hashes so the check does not leak length or prefix via timing
function safeEqual(a: string, b: string) {
  const ha = crypto.createHash('sha256').update(a).digest();
  const hb = crypto.createHash('sha256').update(b).digest();
  return crypto.timingSafeEqual(ha, hb);
}

app.get('/api/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok' });
});

app.use((req: Request, res: Response, next: NextFunction) => {
  if (!AUTH_ENABLED) return next();
  const [scheme, encoded] = (req.headers.authorization || '').split(' ');
  if (scheme === 'Basic' && encoded) {
    const decoded = Buffer.from(encoded, 'base64').toString('utf-8');
    const sep = decoded.indexOf(':');
    if (sep >= 0) {
      const userOk = safeEqual(decoded.slice(0, sep), AUTH_USER);
      const passOk = safeEqual(decoded.slice(sep + 1), AUTH_PASSWORD);
      if (userOk && passOk) return next();
    }
  }
  res.setHeader('WWW-Authenticate', 'Basic realm="FinPrognoz", charset="UTF-8"');
  res.status(401).json({ error: 'Требуется аутентификация' });
});

app.use(express.json({ limit: '10mb' }));

// Log only the error type and stack frames: the message may contain statement data
function sendServerError(res: Response, err: unknown) {
  const name = err instanceof Error ? err.name : 'UnknownError';
  const frames = err instanceof Error && err.stack ? err.stack.split('\n').slice(1, 3).join(' | ') : '';
  console.error(`[api] ${name} ${frames}`);
  res.status(500).json({ error: 'Внутренняя ошибка сервера' });
}

// Accepts YYYY-MM-DD and DD.MM.YYYY; returns YYYY-MM-DD or null for invalid input (e.g. 31.02)
function normalizeDate(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const s = raw.trim();
  let iso: string;
  let m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (m) {
    iso = `${m[1]}-${m[2]}-${m[3]}`;
  } else if ((m = s.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})$/))) {
    iso = `${m[3]}-${m[2].padStart(2, '0')}-${m[1].padStart(2, '0')}`;
  } else {
    return null;
  }
  const t = new Date(iso + 'T00:00:00Z');
  if (isNaN(t.getTime()) || t.toISOString().slice(0, 10) !== iso) return null;
  return iso;
}

// Bank statement upload & analysis
app.post('/api/analyze-statement', (req: Request, res: Response) => {
  try {
    const { companyName, operations, initialBalance, closingBalance } = req.body ?? {};
    if (!Array.isArray(operations) || operations.length === 0) {
      return res.status(400).json({ error: 'Не переданы операции для анализа' });
    }

    // Validate and normalize: rows without a valid date or numeric amount are skipped, not fatal
    const sortedOps: StatementOp[] = [];
    for (const op of operations) {
      if (!op || typeof op !== 'object') continue;
      const date = normalizeDate(op.date);
      const rawAmount = typeof op.amount === 'string' ? Number(op.amount.replace(/\s/g, '').replace(',', '.')) : Number(op.amount);
      if (!date || op.amount === null || op.amount === '' || !Number.isFinite(rawAmount)) continue;
      let isCredit = true;
      if (op.type === 'debit' || op.type === 'outflow' || op.direction === 'Списание' || rawAmount < 0) {
        isCredit = false;
      }
      sortedOps.push({
        date,
        amount: Math.abs(rawAmount),
        isCredit,
        category: typeof op.category === 'string' && op.category ? op.category : categorizeOperation(typeof op.purpose === 'string' ? op.purpose : '', isCredit),
        counterparty: typeof op.counterparty === 'string' && op.counterparty ? op.counterparty : (isCredit ? 'Покупатель' : 'Контрагент'),
        purpose: typeof op.purpose === 'string' ? op.purpose : '',
        // End-of-day balance from the file: the daily history is taken from it
        balance: typeof op.balance === 'number' && Number.isFinite(op.balance) ? op.balance : null
      });
    }
    const skippedOperations = operations.length - sortedOps.length;
    if (sortedOps.length === 0) {
      return res.status(400).json({ error: 'Нет ни одной операции с корректной датой и суммой', skipped_operations: skippedOperations });
    }

    // Forecast, alerts, insights and status (shared/cashflow.ts: the same code the scripts use)
    const a = analyzeStatement(sortedOps, { closingBalance, initialBalance });
    const f = a.sim.forecast;

    res.json({
      company_id: companyName || 'Загруженная компания',
      industry: 'Пользовательские данные',
      split: 'upload',
      current_balance: Math.round(a.currentBalance),
      min_forecast_p50: a.minP50,
      min_forecast_p10: a.minP10,
      max_gap_probability: a.sim.gapProbabilityPct,
      forecast_risk_level: a.status.level,
      forecast_status_label: a.status.label,
      opening_balance: Math.round(a.openingBalance),
      period_start: a.period.start,
      period_end: a.period.end,
      total_operations: a.operations.length,
      skipped_operations: skippedOperations,
      total_inflow: Math.round(a.totals.inflow),
      total_outflow: Math.round(a.totals.outflow),
      net_cash_flow: Math.round(a.totals.inflow - a.totals.outflow),
      inflow_operations: a.totals.inflowCount,
      categories: a.categories,
      counterparties: a.counterparties,
      recurring_items: a.recurring,
      // Daily end-of-day balance for the last 60 calendar days (one point per date for the chart axis)
      history: a.days.slice(-60).map(d => ({
        date: d.date,
        balance: Math.round(d.eod),
        inflow: Math.round(d.inflow),
        outflow: Math.round(d.outflow)
      })),
      forecast: f,
      alerts: a.alerts,
      insights: a.insights,
      // Cumulative net flow of every simulated path: lets the client re-run what-if scenarios on the same distribution
      paths: a.sim.paths,
      // Mean reversion of the median and the stricter status rule: the client applies them in what-if scenarios
      p50_shift: a.sim.p50Shift,
      calm_year: a.calmYear,
      model: { method: a.sim.method, scenarios: a.sim.paths.length, gap_scenarios: a.sim.gapPaths, horizon_days: f.length },
      // Step-by-step log of the calculation, shown on the «Журнал расчёта» tab
      calc_log: explainForecast(a)
    });
  } catch (err) {
    sendServerError(res, err);
  }
});

// Demo statement C052 for the one-click demo
app.get('/api/demo-c052.csv', (_req: Request, res: Response) => {
  const filePath = path.join(DATA_DIR, 'demo_statement_c052.csv');
  if (fs.existsSync(filePath)) {
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.sendFile(filePath);
  } else {
    res.status(404).send('Demo C052 file not found');
  }
});

// Unknown API routes get JSON 404 instead of the SPA index.html
app.use('/api', (_req: Request, res: Response) => {
  res.status(404).json({ error: 'Not found' });
});

// Body parser and route errors: never echo the request body or the raw error message
app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
  if (err?.type === 'entity.parse.failed') {
    return res.status(400).json({ error: 'Некорректный JSON в теле запроса' });
  }
  if (err?.type === 'entity.too.large') {
    return res.status(413).json({ error: 'Слишком большой запрос' });
  }
  sendServerError(res, err);
});

// Setup Vite in development or serve static in production
async function startServer() {
  if (!IS_PROD) {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
  });
}

startServer();
