import express from 'express';
import type { Request, Response, NextFunction } from 'express';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config({ quiet: true });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
// --production lets 'npm start' run in production mode on any OS without setting NODE_ENV
const IS_PROD = process.env.NODE_ENV === 'production' || process.argv.includes('--production');

// Server-only data artifacts. Not under public/, so they are never served as static files.
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

// Helper to load JSON artifacts from data/
function loadJson(fileName: string) {
  const fullPath = path.join(DATA_DIR, fileName);
  if (fs.existsSync(fullPath)) {
    return JSON.parse(fs.readFileSync(fullPath, 'utf-8'));
  }
  return null;
}

// 1. EDA Summary Endpoint
app.get('/api/eda', (req: Request, res: Response) => {
  try {
    const data = loadJson('eda_summary.json');
    if (!data) {
      return res.status(404).json({ error: 'EDA data not generated yet' });
    }
    res.json(data);
  } catch (err) {
    sendServerError(res, err);
  }
});

// 2. Models Benchmark & Libraries Evaluation Endpoint
app.get('/api/benchmark', (req: Request, res: Response) => {
  try {
    const data = loadJson('models_benchmark.json');
    if (!data) {
      return res.status(404).json({ error: 'Benchmark data not found' });
    }
    res.json(data);
  } catch (err) {
    sendServerError(res, err);
  }
});

// 3. Companies Index Endpoint
app.get('/api/companies', (req: Request, res: Response) => {
  try {
    const data = loadJson('companies_index.json');
    if (!data) {
      return res.status(404).json({ error: 'Companies data not found' });
    }
    res.json(data);
  } catch (err) {
    sendServerError(res, err);
  }
});

// 4. Single Company Detailed Forecast Endpoint
app.get('/api/company/:id', (req: Request, res: Response) => {
  try {
    const cid = req.params.id;
    const sampleForecasts = loadJson('sample_forecasts.json') || {};
    
    if (sampleForecasts[cid]) {
      return res.json(sampleForecasts[cid]);
    }

    const companies = loadJson('companies_index.json') || [];
    const compInfo = companies.find((c: any) => c.company_id === cid);

    if (!compInfo) {
      return res.status(404).json({ error: `Company ${cid} not found` });
    }

    const lastBal = compInfo.last_balance;
    const avgDailyFlow = (compInfo.total_inflow + compInfo.total_outflow) / (compInfo.total_operations * 2);
    const futureDates = [];
    let cumBal = lastBal;
    const alerts = [];

    const now = new Date('2026-10-01');
    for (let i = 1; i <= 30; i++) {
      const d = new Date(now);
      d.setDate(d.getDate() + i);
      const isWeekend = d.getDay() === 0 || d.getDay() === 6;
      const dom = d.getDate();

      const predIn = isWeekend ? 0 : avgDailyFlow * (0.8 + Math.sin(i) * 0.3);
      let predOut = isWeekend ? 0 : avgDailyFlow * (0.85 + Math.cos(i) * 0.25);
      let schedOut = 0;
      let schedDesc = 'Текущие операционные расходы';

      if (dom === 10 && !isWeekend) {
        schedOut = avgDailyFlow * 4;
        schedDesc = 'Зарплата: Окончательный расчет (10-е число)';
      } else if (dom === 25 && !isWeekend) {
        schedOut = avgDailyFlow * 3;
        schedDesc = 'Зарплата: Аванс (25-е число)';
      } else if (dom === 28 && !isWeekend) {
        schedOut = avgDailyFlow * 5;
        schedDesc = 'Единый налоговый платеж (ЕНП 28-го числа)';
      }

      predOut += schedOut;
      cumBal += (predIn - predOut);

      const spread = Math.sqrt(i) * avgDailyFlow * 0.8;
      const p10 = cumBal - spread;
      const p90 = cumBal + spread;
      const gapProb = cumBal < 0 ? 80 : (p10 < 0 ? Math.min(65, Math.round((-p10 / spread) * 50)) : 5);

      futureDates.push({
        date: d.toISOString().split('T')[0],
        balance_p50: Math.round(cumBal),
        balance_p10: Math.round(p10),
        balance_p90: Math.round(p90),
        inflow: Math.round(predIn),
        outflow: Math.round(predOut),
        sched_outflow: Math.round(schedOut),
        sched_description: schedDesc,
        gap_probability_pct: gapProb,
        is_forecast: true
      });

      if (gapProb >= 25 || cumBal < 0) {
        const def = Math.max(0, -Math.round(p10), -Math.round(cumBal));
        const isTax = dom === 28;
        const isSalary = dom === 10 || dom === 25;
        const isRent = dom <= 5;
        const defStr = def.toLocaleString('ru-RU') + ' ₽';
        const bufferStr = Math.round(def * 1.15).toLocaleString('ru-RU') + ' ₽';
        const splitStr = Math.round(def * 0.5).toLocaleString('ru-RU') + ' ₽';

        const recs: string[] = [];
        if (isSalary && isTax) {
          recs.push(`1. Приоритет обязательств: Выплатить зарплату (ст. 142 ТК РФ). По ЕНП подать заявление в ФНС о рассрочке (ст. 64 НК РФ) либо внести аванс ${splitStr}, перенеся остаток на 5 дней.`);
        } else if (isTax) {
          recs.push(`1. Налоговый маневр: Обеспечить бронь средств ${defStr} под ЕНП за 48 часов до списания ФНС, либо согласовать налоговый овердрафт.`);
        } else if (isSalary) {
          recs.push(`1. Зарплатный транш: Для бесперебойной выплаты персоналу подключить целевой зарплатный овердрафт со льготным грейс-периодом на ${defStr}.`);
        } else if (isRent) {
          recs.push(`1. Перенос аренды: Направить арендодателю гарантийное письмо о сдвиге оплаты на 7 дней после инкассации поступлений от покупателей.`);
        } else {
          recs.push(`1. Сдвиг кредиторской задолженности: Согласовать с поставщиками частичную оплату 40% и перенос оставшихся ${splitStr} на 5-7 рабочих дней.`);
        }
        recs.push(`2. Экспресс-факторинг: Передать счета дебиторов Банку Центр-инвест — финансирование до 90% (${defStr}) за 2 часа без залога.`);
        recs.push(`3. Возобновляемая кредитная линия (ВКЛ): Открыть лимит в Банке Центр-инвест на ${bufferStr} (дефицит ${defStr} + 15% буфер надежности).`);

        alerts.push({
          date: d.toLocaleDateString('ru-RU'),
          severity: cumBal < 0 || gapProb > 50 ? 'CRITICAL' : 'WARNING',
          probability_pct: gapProb,
          deficit_rub: def,
          balance_p50: Math.round(cumBal),
          balance_p10: Math.round(p10),
          cause: `Пиковая нагрузка обязательств: ${schedDesc} при дефиците расчетного счета ${defStr}`,
          recommendation: recs.join('\n')
        });
      }
    }

    res.json({
      company_id: cid,
      industry: compInfo.industry,
      split: compInfo.split,
      current_balance: lastBal,
      min_forecast_p50: Math.min(...futureDates.map(f => f.balance_p50)),
      min_forecast_p10: Math.min(...futureDates.map(f => f.balance_p10)),
      max_gap_probability: Math.max(...futureDates.map(f => f.gap_probability_pct)),
      recurring_items: [
        { category_name: 'Зарплата: Окончательный расчет', counterparty: 'Сотрудники компании', typical_day: 10, typical_amount: Math.round(avgDailyFlow * 4) },
        { category_name: 'Зарплата: Аванс', counterparty: 'Сотрудники компании', typical_day: 25, typical_amount: Math.round(avgDailyFlow * 3) },
        { category_name: 'Налоги и взносы (ЕНП)', counterparty: 'ФНС России (ЕНП)', typical_day: 28, typical_amount: Math.round(avgDailyFlow * 5) }
      ],
      history: [],
      forecast: futureDates,
      alerts
    });
  } catch (err) {
    sendServerError(res, err);
  }
});

// 5. Custom Bank Statement Upload & Analysis Endpoint
app.post('/api/analyze-statement', (req: Request, res: Response) => {
  try {
    const { companyName, operations, initialBalance, closingBalance } = req.body ?? {};
    if (!Array.isArray(operations) || operations.length === 0) {
      return res.status(400).json({ error: 'Не переданы операции для анализа' });
    }

    // Validate and normalize: rows without a valid date or numeric amount are skipped, not fatal
    const sortedOps: Array<{ date: string; amount: number; isCredit: boolean; category: string; counterparty: string }> = [];
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
        category: typeof op.category === 'string' && op.category ? op.category : (isCredit ? 'Выручка от покупателей' : 'Оплата поставщикам'),
        counterparty: typeof op.counterparty === 'string' && op.counterparty ? op.counterparty : (isCredit ? 'Покупатель' : 'Контрагент')
      });
    }
    const skippedOperations = operations.length - sortedOps.length;
    if (sortedOps.length === 0) {
      return res.status(400).json({ error: 'Нет ни одной операции с корректной датой и суммой', skipped_operations: skippedOperations });
    }

    // Sort operations by date (ISO strings compare chronologically; sort is stable for same-day rows)
    sortedOps.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));

    // closingBalance is balance_eod at the end of the statement; the opening balance is
    // reconstructed from it so the history is not added on top of the final balance twice
    const netHistory = sortedOps.reduce((s, op) => s + (op.isCredit ? op.amount : -op.amount), 0);
    const hasClosing = typeof closingBalance === 'number' && Number.isFinite(closingBalance);
    const openingBalance = hasClosing ? closingBalance - netHistory : (Number(initialBalance) || 0);

    let runningBal = openingBalance;
    let totalInflow = 0;
    let totalOutflow = 0;
    const catMap: Record<string, { count: number; total: number; direction: string }> = {};
    const counterpartyMap: Record<string, { count: number; total: number; direction: string }> = {};
    const historyPoints: Array<{ date: string; balance: number; inflow: number; outflow: number }> = [];

    for (const op of sortedOps) {
      const { amount, isCredit, category, counterparty } = op;

      if (isCredit) {
        totalInflow += amount;
        runningBal += amount;
      } else {
        totalOutflow += amount;
        runningBal -= amount;
      }

      if (!catMap[category]) {
        catMap[category] = { count: 0, total: 0, direction: isCredit ? 'Поступление' : 'Списание' };
      }
      catMap[category].count += 1;
      catMap[category].total += amount;

      if (!counterpartyMap[counterparty]) {
        counterpartyMap[counterparty] = { count: 0, total: 0, direction: isCredit ? 'Поступление' : 'Списание' };
      }
      counterpartyMap[counterparty].count += 1;
      counterpartyMap[counterparty].total += amount;

      historyPoints.push({
        date: op.date,
        balance: Math.round(runningBal),
        inflow: isCredit ? amount : 0,
        outflow: isCredit ? 0 : amount
      });
    }

    const avgDailyFlow = (totalInflow + totalOutflow) / (Math.max(1, sortedOps.length) * 1.5);
    const futureDates = [];
    let cumBal = runningBal;
    const alerts: Array<any> = [];

    // Dates are UTC midnight, so use UTC getters: results must not depend on the server timezone
    const lastDate = new Date(sortedOps[sortedOps.length - 1].date + 'T00:00:00Z');
    for (let i = 1; i <= 30; i++) {
      const d = new Date(lastDate);
      d.setUTCDate(d.getUTCDate() + i);
      const isWeekend = d.getUTCDay() === 0 || d.getUTCDay() === 6;
      const dom = d.getUTCDate();

      const predIn = isWeekend ? 0 : avgDailyFlow * (0.85 + Math.sin(i * 0.7) * 0.25);
      let predOut = isWeekend ? 0 : avgDailyFlow * (0.80 + Math.cos(i * 0.7) * 0.20);
      let schedOut = 0;
      let schedDesc = '';

      if (dom === 10 && !isWeekend) {
        schedOut = avgDailyFlow * 3.5;
        schedDesc = 'Зарплата: Окончательный расчет (10-е число)';
      } else if (dom === 25 && !isWeekend) {
        schedOut = avgDailyFlow * 2.8;
        schedDesc = 'Зарплата: Аванс (25-е число)';
      } else if (dom === 28 && !isWeekend) {
        schedOut = avgDailyFlow * 4.5;
        schedDesc = 'Единый налоговый платеж (ЕНП 28-го числа)';
      } else if (dom <= 3 && !isWeekend) {
        schedOut = avgDailyFlow * 1.8;
        schedDesc = 'Аренда помещений и сервисов';
      }

      predOut += schedOut;
      cumBal += (predIn - predOut);

      const spread = Math.sqrt(i) * (avgDailyFlow * 0.75 + 15000);
      const p10 = cumBal - spread * 1.28;
      const p90 = cumBal + spread * 1.28;
      let gapProb = 5;
      if (cumBal < 0) {
        gapProb = Math.min(98, Math.round(75 + (Math.abs(cumBal) / (spread + 1)) * 20));
      } else if (p10 < 0) {
        gapProb = Math.min(68, Math.round((-p10 / (spread * 2.56 + 1)) * 60 + 10));
      }

      futureDates.push({
        date: d.toISOString().split('T')[0],
        balance_p50: Math.round(cumBal),
        balance_p10: Math.round(p10),
        balance_p90: Math.round(p90),
        inflow: Math.round(predIn),
        outflow: Math.round(predOut),
        sched_outflow: Math.round(schedOut),
        sched_description: schedDesc || 'Текущая операционная деятельность',
        gap_probability_pct: gapProb,
        is_forecast: true
      });

      if (gapProb >= 25 || cumBal < 0) {
        const def = Math.max(0, -Math.round(p10), -Math.round(cumBal));
        const defStr = def.toLocaleString('ru-RU') + ' ₽';
        const bufferStr = Math.round(def * 1.15).toLocaleString('ru-RU') + ' ₽';
        const splitStr = Math.round(def * 0.5).toLocaleString('ru-RU') + ' ₽';

        const recs: string[] = [];
        if (schedDesc.includes('Зарплата') && schedDesc.includes('ЕНП')) {
          recs.push(`1. Приоритет обязательств: Выплатить зарплату. По ЕНП подать заявление в ФНС о рассрочке (ст. 64 НК РФ) либо внести аванс ${splitStr}.`);
        } else if (schedDesc.includes('ЕНП')) {
          recs.push(`1. Налоговый маневр: Обеспечить бронь средств ${defStr} под ЕНП за 48 часов до списания ФНС, либо согласовать налоговый овердрафт.`);
        } else if (schedDesc.includes('Зарплата')) {
          recs.push(`1. Зарплатный транш: Подключить целевой зарплатный овердрафт со льготным грейс-периодом на ${defStr}.`);
        } else if (schedDesc.includes('Аренда')) {
          recs.push(`1. Перенос аренды: Направить арендодателю гарантийное письмо о сдвиге оплаты на 7 дней после инкассации выручки.`);
        } else {
          recs.push(`1. Сдвиг кредиторской задолженности: Согласовать с поставщиками частичную оплату 40% и перенос ${splitStr} на 5-7 дней.`);
        }
        recs.push(`2. Экспресс-факторинг: Передать дебиторскую задолженность Банку Центр-инвест — финансирование до 90% (${defStr}) за 2 часа.`);
        recs.push(`3. Кредитная линия: Открыть лимит в Банке Центр-инвест на ${bufferStr} (дефицит ${defStr} + 15% буфер ликвидности).`);

        alerts.push({
          date: d.toLocaleDateString('ru-RU', { timeZone: 'UTC' }),
          severity: cumBal < 0 || gapProb > 50 ? 'CRITICAL' : 'WARNING',
          probability_pct: gapProb,
          deficit_rub: def,
          balance_p50: Math.round(cumBal),
          balance_p10: Math.round(p10),
          cause: schedDesc ? `Совпадение критических выплат: ${schedDesc} при дефиците остатка ${defStr}` : `Превышение текущих операционных расходов над выручкой (дефицит: ${defStr})`,
          recommendation: recs.join('\n')
        });
      }
    }

    const minP50 = Math.min(...futureDates.map(f => f.balance_p50));
    const minP10 = Math.min(...futureDates.map(f => f.balance_p10));
    const maxGapProb = Math.max(...futureDates.map(f => f.gap_probability_pct));

    let riskLevel: 'CRITICAL' | 'WARNING' | 'STABLE' = 'STABLE';
    let statusLabel = 'Стабильное состояние (В норме)';
    if (maxGapProb >= 50 || minP50 < 0) {
      riskLevel = 'CRITICAL';
      statusLabel = 'Критический кассовый разрыв';
    } else if (maxGapProb >= 20 || minP10 < 0) {
      riskLevel = 'WARNING';
      statusLabel = 'Повышенный риск кассового разрыва';
    }

    if (alerts.length === 0) {
      alerts.push({
        date: '30-дневный горизонт',
        severity: 'STABLE',
        probability_pct: maxGapProb,
        deficit_rub: 0,
        balance_p50: minP50,
        balance_p10: minP10,
        cause: 'Кассовых разрывов не прогнозируется: операционные поступления стабильно превышают календарные обязательства.',
        recommendation: 'Поддерживать плановую платежную дисциплину. Риски дефицита ликвидности отсутствуют.'
      });
    }

    res.json({
      company_id: companyName || 'Загруженная компания',
      industry: 'Пользовательские данные',
      split: 'upload',
      current_balance: Math.round(runningBal),
      min_forecast_p50: minP50,
      min_forecast_p10: minP10,
      max_gap_probability: maxGapProb,
      forecast_risk_level: riskLevel,
      forecast_status_label: statusLabel,
      opening_balance: Math.round(openingBalance),
      total_operations: sortedOps.length,
      skipped_operations: skippedOperations,
      total_inflow: Math.round(totalInflow),
      total_outflow: Math.round(totalOutflow),
      net_cash_flow: Math.round(totalInflow - totalOutflow),
      categories: Object.entries(catMap).map(([name, data]) => ({
        name,
        total_amount: Math.round(data.total),
        direction: data.direction,
        count: data.count
      })).sort((a, b) => b.total_amount - a.total_amount),
      counterparties: Object.entries(counterpartyMap).map(([name, data]) => ({
        name,
        total: Math.round(data.total),
        direction: data.direction,
        count: data.count
      })).sort((a, b) => b.total - a.total).slice(0, 8),
      recurring_items: [
        { category_name: 'Зарплата: Окончательный расчет', counterparty: 'Сотрудники компании', typical_day: 10, typical_amount: Math.round(avgDailyFlow * 3.5) },
        { category_name: 'Зарплата: Аванс', counterparty: 'Сотрудники компании', typical_day: 25, typical_amount: Math.round(avgDailyFlow * 2.8) },
        { category_name: 'Налоги и взносы (ЕНП)', counterparty: 'ФНС России (ЕНП)', typical_day: 28, typical_amount: Math.round(avgDailyFlow * 4.5) }
      ],
      history: historyPoints.slice(-60),
      forecast: futureDates,
      alerts
    });
  } catch (err) {
    sendServerError(res, err);
  }
});

// 6. Template C052 Download Endpoint
app.get('/api/template-c052.csv', (_req: Request, res: Response) => {
  const filePath = path.join(DATA_DIR, 'template_c052.csv');
  if (fs.existsSync(filePath)) {
    res.setHeader('Content-Disposition', 'attachment; filename="template_c052.csv"');
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.sendFile(filePath);
  } else {
    res.status(404).send('Template file not found');
  }
});

// 7. Demo C052 Download / Fetch Endpoint
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
