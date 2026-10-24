// Empirical cash-flow forecast shared by the server (forecast + insights) and the client (what-if).
//
// Method: "analog periods" Monte Carlo. Each simulated path replays the real daily cash flows of a
// past 30-day window that starts on (almost) the same day of month as the forecast, applied to
// today's balance. P10/P50/P90 and the cash-gap probability are therefore shares of real historical
// scenarios, not a parametric guess. With too little history, single days are bootstrapped instead.

export const HORIZON_DAYS = 30;

export interface StatementOp {
  date: string; // YYYY-MM-DD
  amount: number; // absolute value
  isCredit: boolean;
  category: string;
  counterparty: string;
  purpose: string;
}

export interface ForecastPoint {
  date: string;
  balance_p50: number;
  balance_p10: number;
  balance_p90: number;
  inflow: number;
  outflow: number;
  sched_outflow: number;
  sched_description: string;
  gap_probability_pct: number;
  is_forecast: boolean;
}

export interface GapAlert {
  date: string;
  severity: string;
  probability_pct: number;
  deficit_rub: number;
  balance_p50: number;
  balance_p10: number;
  cause: string;
  recommendation: string;
}

export interface Insight {
  severity: 'CRITICAL' | 'WARNING' | 'INFO';
  title: string;
  detail: string;
  recommendation: string;
}

export interface RecurringItem {
  category_name: string;
  counterparty: string;
  typical_day: number;
  typical_amount: number;
  count: number;
}

// Calendar obligations: salary, taxes (ЕНП), rent. Matches the client's category names.
const SCHEDULED_CATEGORY = /Зарплата|Налоги|ЕНП|Аренда/;
// Owner withdrawals: transfers to a personal account, dividends, accountable sums
const OWNER_WITHDRAWAL = /личн[а-я]* сч|собственных средств|дивиденд|подотч[её]т/i;

export const formatRubExact = (v: number) => Math.round(v).toLocaleString('ru-RU') + ' ₽';
const isoToRu = (iso: string) => iso.split('-').reverse().join('.');

export function addDaysIso(iso: string, n: number): string {
  const d = new Date(iso + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

const dayOfMonth = (iso: string) => Number(iso.slice(8, 10));
const isWeekendIso = (iso: string) => {
  const wd = new Date(iso + 'T00:00:00Z').getUTCDay();
  return wd === 0 || wd === 6;
};

function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

const median = (values: number[]) => quantile([...values].sort((a, b) => a - b), 0.5);

// Deterministic PRNG so the same statement always yields the same forecast
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface HistoryDay {
  date: string;
  inflow: number;
  outflow: number;
  sched: number;
  schedNames: string[];
  ownerOut: number;
  eod: number;
}

// Contiguous daily series (days without operations included) with end-of-day balance
export function buildDailySeries(ops: StatementOp[], openingBalance: number): HistoryDay[] {
  if (ops.length === 0) return [];
  const byDate = new Map<string, StatementOp[]>();
  for (const op of ops) {
    const list = byDate.get(op.date);
    if (list) list.push(op);
    else byDate.set(op.date, [op]);
  }
  const days: HistoryDay[] = [];
  let bal = openingBalance;
  const last = ops[ops.length - 1].date;
  for (let date = ops[0].date; date <= last; date = addDaysIso(date, 1)) {
    const day: HistoryDay = { date, inflow: 0, outflow: 0, sched: 0, schedNames: [], ownerOut: 0, eod: 0 };
    for (const op of byDate.get(date) ?? []) {
      if (op.isCredit) {
        day.inflow += op.amount;
      } else {
        day.outflow += op.amount;
        if (SCHEDULED_CATEGORY.test(op.category)) {
          day.sched += op.amount;
          if (!day.schedNames.includes(op.category)) day.schedNames.push(op.category);
        }
        if (OWNER_WITHDRAWAL.test(op.purpose)) day.ownerOut += op.amount;
      }
    }
    bal += day.inflow - day.outflow;
    day.eod = bal;
    days.push(day);
  }
  return days;
}

export interface SimulationResult {
  method: 'analog_periods' | 'block_bootstrap' | 'day_bootstrap';
  paths: number[][]; // cumulative net flow per path and day
  forecast: ForecastPoint[]; // base forecast (no scenario), quantiles from paths
  gapPaths: number;
  gapProbabilityPct: number; // probability of a gap anywhere within the horizon
}

export function simulateForecast(days: HistoryDay[], currentBalance: number): SimulationResult {
  const lastDate = days[days.length - 1].date;
  const futureDates = Array.from({ length: HORIZON_DAYS }, (_, i) => addDaysIso(lastDate, i + 1));
  const targetDom = dayOfMonth(futureDates[0]);
  const domDistance = (a: number, b: number) => Math.min(Math.abs(a - b), 31 - Math.abs(a - b));

  // Windows of HORIZON_DAYS history days starting near the same day of month
  const starts: number[] = [];
  for (let s = 0; s + HORIZON_DAYS <= days.length; s++) {
    if (domDistance(dayOfMonth(days[s].date), targetDom) <= 3) starts.push(s);
  }
  const recent = starts.filter(s => s >= days.length - 365);
  const chosen = recent.length >= 12 ? recent : starts;

  // Each sample is a sequence of HORIZON_DAYS history days
  let samples: HistoryDay[][];
  let method: SimulationResult['method'];
  if (chosen.length >= 5) {
    method = 'analog_periods';
    samples = chosen.map(s => days.slice(s, s + HORIZON_DAYS));
  } else if (days.length >= 14) {
    // Short history: glue random 7-day blocks, which keeps big inflows and payments of the same week together
    method = 'block_bootstrap';
    const rng = mulberry32(days.length * 7919 + Math.round(Math.abs(currentBalance)));
    samples = Array.from({ length: 300 }, () => {
      const seq: HistoryDay[] = [];
      while (seq.length < HORIZON_DAYS) {
        const s = Math.floor(rng() * (days.length - 6));
        seq.push(...days.slice(s, s + 7));
      }
      return seq.slice(0, HORIZON_DAYS);
    });
  } else {
    method = 'day_bootstrap';
    const rng = mulberry32(days.length * 7919 + Math.round(Math.abs(currentBalance)));
    const weekdays = days.filter(d => !isWeekendIso(d.date));
    const weekends = days.filter(d => isWeekendIso(d.date));
    samples = Array.from({ length: 300 }, () =>
      futureDates.map(fd => {
        const pool = isWeekendIso(fd) && weekends.length > 0 ? weekends : weekdays.length > 0 ? weekdays : days;
        return pool[Math.floor(rng() * pool.length)];
      })
    );
  }

  const paths = samples.map(sample => {
    let cum = 0;
    return sample.map(d => (cum += d.inflow - d.outflow));
  });
  const summary = summarizePaths(currentBalance, paths, new Array(HORIZON_DAYS).fill(0), 0);

  const forecast: ForecastPoint[] = futureDates.map((date, i) => {
    const mean = (pick: (d: HistoryDay) => number) => samples.reduce((s, smp) => s + pick(smp[i]), 0) / samples.length;
    // Calendar obligations that happen on this day in at least 40% of analog periods
    const nameCounts = new Map<string, number>();
    for (const smp of samples) for (const n of smp[i].schedNames) nameCounts.set(n, (nameCounts.get(n) ?? 0) + 1);
    const typical = [...nameCounts.entries()].filter(([, c]) => c / samples.length >= 0.4).map(([n]) => n);
    return {
      date,
      balance_p50: Math.round(summary.p50[i]),
      balance_p10: Math.round(summary.p10[i]),
      balance_p90: Math.round(summary.p90[i]),
      inflow: Math.round(mean(d => d.inflow)),
      outflow: Math.round(mean(d => d.outflow)),
      sched_outflow: Math.round(mean(d => d.sched)),
      sched_description: typical.length > 0 ? typical.join(', ') : 'Текущая операционная деятельность',
      gap_probability_pct: summary.gapProb[i],
      is_forecast: true
    };
  });

  return {
    method,
    paths: paths.map(p => p.map(v => Math.round(v))),
    forecast,
    gapPaths: summary.gapPaths,
    gapProbabilityPct: summary.overallGapProb
  };
}

// Applies a deterministic scenario shift (cumulative, per day) to every path and recomputes quantiles.
// Without paths (manual-only mode) a single flat path is used, so the result is deterministic.
export function summarizePaths(currentBalance: number, paths: number[][], cumDelta: number[], buffer: number) {
  const usable = paths.length > 0 ? paths : [new Array(cumDelta.length).fill(0)];
  const n = cumDelta.length;
  const p10: number[] = [], p50: number[] = [], p90: number[] = [], gapProb: number[] = [];
  const pathMin = new Array(usable.length).fill(Infinity);
  for (let d = 0; d < n; d++) {
    const vals = usable.map((p, k) => {
      const v = currentBalance + (p[d] ?? p[p.length - 1] ?? 0) + cumDelta[d];
      if (v < pathMin[k]) pathMin[k] = v;
      return v;
    });
    const below = vals.filter(v => v < buffer).length;
    vals.sort((a, b) => a - b);
    p10.push(quantile(vals, 0.1));
    p50.push(quantile(vals, 0.5));
    p90.push(quantile(vals, 0.9));
    gapProb.push(Math.round((below / usable.length) * 1000) / 10);
  }
  const gapPaths = pathMin.filter(m => m < buffer).length;
  return { p10, p50, p90, gapProb, gapPaths, pathCount: usable.length, overallGapProb: Math.round((gapPaths / usable.length) * 1000) / 10 };
}

export function gapRecommendation(schedDesc: string, deficit: number): string {
  const defStr = formatRubExact(deficit);
  const bufferStr = formatRubExact(deficit * 1.15);
  const splitStr = formatRubExact(deficit * 0.5);
  const recs: string[] = [];
  if (schedDesc.includes('Зарплата') && schedDesc.includes('ЕНП')) {
    recs.push(`1. Приоритет обязательств: Выплатить зарплату. По ЕНП подать заявление в ФНС о рассрочке (ст. 64 НК РФ) либо внести аванс ${splitStr}.`);
  } else if (schedDesc.includes('ЕНП')) {
    recs.push(`1. Налоговый маневр: Обеспечить бронь средств ${defStr} под ЕНП за 48 часов до списания ФНС, либо согласовать налоговый овердрафт.`);
  } else if (schedDesc.includes('Зарплата')) {
    recs.push(`1. Зарплатный транш: Подключить целевой зарплатный овердрафт Банка Центр-инвест со льготным грейс-периодом на ${defStr}.`);
  } else if (schedDesc.includes('Аренда')) {
    recs.push(`1. Перенос аренды: Направить арендодателю гарантийное письмо о сдвиге оплаты на 7 дней после инкассации выручки.`);
  } else {
    recs.push(`1. Сдвиг кредиторской задолженности: Согласовать с поставщиками частичную оплату 40% и перенос ${splitStr} на 5-7 дней.`);
  }
  recs.push(`2. Экспресс-факторинг: Передать дебиторскую задолженность Банку Центр-инвест — финансирование до 90% (${defStr}) за 2 часа.`);
  recs.push(`3. Кредитная линия: Открыть лимит в Банке Центр-инвест на ${bufferStr} (дефицит ${defStr} + 15% буфер ликвидности).`);
  return recs.join('\n');
}

// One alert per contiguous risk episode (not one per day): first risky day, worst P10, peak probability
export function buildGapAlerts(points: ForecastPoint[], buffer: number): GapAlert[] {
  const alerts: GapAlert[] = [];
  let i = 0;
  while (i < points.length) {
    // A day is risky when at least 10% of scenarios (P10) or the median fall below the buffer
    const risky = (p: ForecastPoint) => p.gap_probability_pct >= 10 || p.balance_p50 < buffer;
    if (!risky(points[i])) {
      i++;
      continue;
    }
    const start = i;
    while (i < points.length && risky(points[i])) i++;
    const episode = points.slice(start, i);
    const worst = episode.reduce((w, p) => (p.balance_p10 < w.balance_p10 ? p : w), episode[0]);
    const peakProb = Math.max(...episode.map(p => p.gap_probability_pct));
    const p50Breach = episode.some(p => p.balance_p50 < buffer);
    const deficit = Math.max(0, Math.round(buffer - worst.balance_p10));
    const obligations = [...new Set(episode.map(p => p.sched_description).filter(s => s && s !== 'Текущая операционная деятельность'))];
    const period = episode.length > 1 ? ` (период риска ${isoToRu(episode[0].date)}–${isoToRu(episode[episode.length - 1].date)})` : '';
    alerts.push({
      date: isoToRu(episode[0].date),
      severity: p50Breach || peakProb > 50 ? 'CRITICAL' : 'WARNING',
      probability_pct: peakProb,
      deficit_rub: deficit,
      balance_p50: worst.balance_p50,
      balance_p10: worst.balance_p10,
      cause:
        (obligations.length > 0
          ? `Совпадение крупных платежей: ${obligations.join(' | ')}.`
          : 'Расходы превышают поступления.') +
        ` Медианный остаток ${formatRubExact(worst.balance_p50)}, пессимистичный (P10) ${formatRubExact(worst.balance_p10)} на ${isoToRu(worst.date)}${period}.`,
      recommendation: gapRecommendation(obligations.join(' '), deficit)
    });
  }
  return alerts;
}

// Liquidity diagnostics that matter even when no gap is forecast (thin cushion, negative trend, ...)
export function buildInsights(days: HistoryDay[], currentBalance: number, sim: SimulationResult): Insight[] {
  const insights: Insight[] = [];
  const lastDays = (n: number) => days.slice(-n);
  const sum = (list: HistoryDay[], pick: (d: HistoryDay) => number) => list.reduce((s, d) => s + pick(d), 0);
  const last90 = lastDays(90);
  const span90 = Math.max(1, last90.length);
  const months90 = span90 / 30.4;
  const avgDailyOut = sum(last90, d => d.outflow) / span90;
  const net90 = sum(last90, d => d.inflow - d.outflow);
  const owner90 = sum(last90, d => d.ownerOut);
  const minP10 = Math.min(...sim.forecast.map(p => p.balance_p10));

  // 1. Historical replay: how many analog periods end in a gap
  if (sim.gapPaths > 0) {
    const total = sim.paths.length;
    const need = Math.max(0, -minP10);
    insights.push({
      severity: sim.gapProbabilityPct >= 50 ? 'CRITICAL' : 'WARNING',
      title: `Риск разрыва ${sim.gapProbabilityPct}%: ${sim.gapPaths} из ${total} похожих периодов истории уводят остаток в минус`,
      detail:
        sim.method === 'analog_periods'
          ? `Если ближайшие 30 дней повторят один из ${total} прошлых периодов, начинавшихся около того же числа месяца, в ${sim.gapPaths} случаях остаток опустится ниже нуля. Пессимистичный остаток (P10): ${formatRubExact(minP10)}.`
          : `Истории меньше двух месяцев, поэтому прогноз собран из случайных дней выписки (${total} сценариев). Пессимистичный остаток (P10): ${formatRubExact(minP10)}.`,
      recommendation:
        need > 0
          ? `Заранее согласовать овердрафт или кредитную линию не меньше ${formatRubExact(need * 1.15)} (нехватка в пессимистичном сценарии + 15%).`
          : `Заранее согласовать овердрафт на сумму недельных расходов (${formatRubExact(avgDailyOut * 7)}) и не ставить крупные платежи на одни и те же дни.`
    });
  }

  // 2. Cash cushion in days of spending
  if (avgDailyOut > 0) {
    const coverage = currentBalance / avgDailyOut;
    const target = avgDailyOut * 14;
    if (coverage < 14) {
      insights.push({
        severity: coverage < 5 ? 'CRITICAL' : 'WARNING',
        title: `Тонкая подушка ликвидности: остатка хватит на ${Math.max(0, coverage).toFixed(1)} дн. расходов`,
        detail: `Средний расход за последние 90 дней: ${formatRubExact(avgDailyOut)} в день. Текущий остаток ${formatRubExact(currentBalance)}. Любая задержка поступлений больше чем на ${Math.max(0, Math.floor(coverage))} дн. приведёт к разрыву.`,
        recommendation: `Довести неснижаемый остаток до 2 недель расходов (${formatRubExact(target)}): не хватает ${formatRubExact(target - currentBalance)}. До этого держать открытым овердрафт на эту сумму.`
      });
    }
  }

  // 3. Near-miss in the past year
  const lastYear = lastDays(365);
  const minDay = lastYear.reduce((m, d) => (d.eod < m.eod ? d : m), lastYear[0]);
  if (minDay && avgDailyOut > 0 && minDay.eod < avgDailyOut * 2) {
    const month = minDay.date.slice(0, 7);
    const monthNet = sum(days.filter(d => d.date.startsWith(month)), d => d.inflow - d.outflow);
    insights.push({
      severity: minDay.eod < 0 ? 'CRITICAL' : 'WARNING',
      title: `${isoToRu(minDay.date)} остаток опускался до ${formatRubExact(minDay.eod)}: компания уже была на грани разрыва`,
      detail: `Чистый поток за ${month.split('-').reverse().join('.')}: ${formatRubExact(monthNet)}. Похожий месяц сейчас начнётся с остатка ${formatRubExact(currentBalance)}.`,
      recommendation: 'Разобрать, какие платежи совпали в тот период, и заранее разнести их по датам или обеспечить резерв на этот месяц.'
    });
  }

  // 4. Negative recent trend
  if (net90 < 0) {
    const perMonth = net90 / months90;
    const monthsLeft = currentBalance > 0 ? currentBalance / -perMonth : 0;
    insights.push({
      severity: monthsLeft < 3 ? 'WARNING' : 'INFO',
      title: `Отрицательный денежный поток: ${formatRubExact(perMonth)} в месяц за последние 3 месяца`,
      detail: `За 90 дней расходы превысили поступления на ${formatRubExact(-net90)}. При таком темпе текущего остатка хватит примерно на ${monthsLeft.toFixed(1)} мес.`,
      recommendation: 'Ускорить сбор дебиторской задолженности, пересмотреть необязательные расходы и сроки оплаты поставщикам.'
    });
  }

  // 5. Owner withdrawals eat the free cash flow
  if (owner90 > 0) {
    const ownerPerMonth = owner90 / months90;
    const netExOwner = (net90 + owner90) / months90;
    if (net90 < 0 && netExOwner > 0) {
      // Leave room to rebuild a two-week cash reserve within three months
      const reserveGap = Math.max(0, avgDailyOut * 14 - currentBalance);
      const limit = Math.max(0, netExOwner - reserveGap / 3);
      // Same runway rule as the negative-trend insight: a warning only when the balance lasts less than 3 months
      const monthsLeft = currentBalance > 0 ? currentBalance / (-net90 / months90) : 0;
      insights.push({
        severity: monthsLeft < 3 ? 'WARNING' : 'INFO',
        title: `Выводы владельца и переводы на свои счета (${formatRubExact(ownerPerMonth)}/мес) превышают свободный денежный поток`,
        detail: `Без выводов на личный счёт бизнес генерирует ${formatRubExact(netExOwner)} в месяц, с ними поток отрицательный. При таком темпе текущего остатка хватит примерно на ${monthsLeft.toFixed(1)} мес.`,
        recommendation:
          `Ограничить вывод суммой до ${formatRubExact(limit)} в месяц` +
          (reserveGap > 0 ? ` (свободный поток минус ${formatRubExact(reserveGap / 3)} на пополнение резерва за 3 месяца)` : '') +
          ' и делать его после крупных поступлений, а не перед зарплатой и ЕНП.'
      });
    } else {
      insights.push({
        severity: 'INFO',
        title: `Выводы владельца и переводы на свои счета: ${formatRubExact(ownerPerMonth)} в месяц`,
        detail: `Это ${Math.round((owner90 / Math.max(1, sum(last90, d => d.outflow))) * 100)}% всех списаний за 90 дней.`,
        recommendation: 'Планировать выводы после крупных поступлений, чтобы не снижать остаток перед обязательными платежами.'
      });
    }
  }

  const order = { CRITICAL: 0, WARNING: 1, INFO: 2 };
  return insights.sort((a, b) => order[a.severity] - order[b.severity]);
}

// Salary, taxes, rent detected from the last 6 months of actual payments
export function detectRecurring(ops: StatementOp[]): RecurringItem[] {
  if (ops.length === 0) return [];
  const since = addDaysIso(ops[ops.length - 1].date, -183);
  const groups = new Map<string, StatementOp[]>();
  for (const op of ops) {
    if (op.isCredit || op.date < since || !SCHEDULED_CATEGORY.test(op.category)) continue;
    const list = groups.get(op.category);
    if (list) list.push(op);
    else groups.set(op.category, [op]);
  }
  const items: RecurringItem[] = [];
  for (const [category, list] of groups) {
    const perMonth = new Map<string, number>();
    for (const op of list) perMonth.set(op.date.slice(0, 7), (perMonth.get(op.date.slice(0, 7)) ?? 0) + op.amount);
    if (perMonth.size < 3) continue;
    const parties = new Map<string, number>();
    for (const op of list) parties.set(op.counterparty, (parties.get(op.counterparty) ?? 0) + 1);
    items.push({
      category_name: category,
      counterparty: [...parties.entries()].sort((a, b) => b[1] - a[1])[0][0],
      typical_day: Math.round(median(list.map(op => dayOfMonth(op.date)))),
      typical_amount: Math.round(median([...perMonth.values()])),
      count: list.length
    });
  }
  return items.sort((a, b) => a.typical_day - b.typical_day);
}
