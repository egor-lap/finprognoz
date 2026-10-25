// One self-contained HTML report of npm run evaluate: the forecast of every company with its calculation log,
// the backtest metrics with confusion matrices, the charts and the step-by-step metrics log. Charts are embedded
// as base64 and there are no external links, so the file opens on any computer without internet.
//
// Run:  node scripts/report.ts <folder with forecast.json, backtest_metrics.json and the PNG charts>
// Output: <folder>/report.html

import fs from 'fs';
import path from 'path';
import type { CalcStep, ForecastPoint } from '../shared/cashflow.ts';
import type { CompanyForecast } from './forecast.ts';

const dir = process.argv[2];
if (!dir || !fs.existsSync(path.join(dir, 'forecast.json'))) {
  console.error('Укажите папку с результатами прогноза: node scripts/report.ts reports/evaluate/<имя>');
  process.exit(1);
}
const readJson = (name: string) => {
  const file = path.join(dir, name);
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
};
const fc: { generated_at: string; files: string[]; status_counts: Record<string, number>; companies: CompanyForecast[] } = readJson('forecast.json');
const m = readJson('backtest_metrics.json'); // null when the backtest could not run on this data
const backtestError = fs.existsSync(path.join(dir, 'backtest_error.txt')) ? fs.readFileSync(path.join(dir, 'backtest_error.txt'), 'utf8').trim() : '';

// ---------------------------------------------------------------- formatting
const esc = (s: unknown) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const ruDate = (iso: string | null | undefined) => (iso ? iso.split('-').reverse().join('.') : '—');
const rub = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : `${Math.round(v).toLocaleString('ru-RU')} ₽`);
const fixed = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toLocaleString('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits });
const pct = (v: number | null | undefined, digits = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : `${fixed(v * 100, digits)}%`);
const ciText = (range: [number, number] | null | undefined, f: (v: number) => string) => (range ? `95% ДИ ${f(range[0])} – ${f(range[1])}` : '');
const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹';
function sci(v: number): string {
  const [mantissa, exp] = v.toExponential(2).split('e');
  return `${mantissa.replace('.', ',')}·10${String(Number(exp)).split('').map(d => (d === '-' ? '⁻' : SUPERSCRIPT[Number(d)])).join('')}`;
}
function plural(n: number, one: string, few: string, many: string) {
  const m10 = Math.abs(n) % 10;
  const m100 = Math.abs(n) % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
const STATUS_CLASS: Record<string, string> = { CRITICAL: 'crit', WARNING: 'warn', STABLE: 'ok' };
const STATUS_SHORT: Record<string, string> = { CRITICAL: 'Критический', WARNING: 'Повышенный риск', STABLE: 'Стабильно' };
const chip = (level: string) => `<span class="chip ${STATUS_CLASS[level]}">${STATUS_SHORT[level]}</span>`;

function stepsHtml(steps: CalcStep[]): string {
  return `<ol class="steps">${steps.map(s => {
    const items: string[] = [];
    for (const line of s.lines) {
      // A line indented with two spaces continues the previous one
      if (line.startsWith('  ') && items.length > 0) items[items.length - 1] += `<div class="sub">${esc(line.trim())}</div>`;
      else items.push(esc(line));
    }
    return `<li><h4>${esc(s.title)}</h4><ul>${items.map(i => `<li>${i}</li>`).join('')}</ul></li>`;
  }).join('')}</ol>`;
}

const mln = (v: number) => `${(v / 1e6).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} млн ₽`;

// Fan chart of one company: P10–P90 band, P50 line and the zero line
function fanChart(f: ForecastPoint[], current: number): string {
  const W = 640, H = 170, L = 8, R = 8, T = 10, B = 22;
  const values = [current, 0, ...f.flatMap(p => [p.balance_p10, p.balance_p90])];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const x = (i: number) => L + ((W - L - R) * i) / f.length;
  const y = (v: number) => T + ((H - T - B) * (hi - v)) / span;
  const pts = (pick: (p: ForecastPoint) => number) => f.map((p, i) => `${x(i + 1).toFixed(1)},${y(pick(p)).toFixed(1)}`);
  const band = [`${x(0).toFixed(1)},${y(current).toFixed(1)}`, ...pts(p => p.balance_p90), ...pts(p => p.balance_p10).reverse()].join(' ');
  const median = [`${x(0).toFixed(1)},${y(current).toFixed(1)}`, ...pts(p => p.balance_p50)].join(' ');
  return `<svg class="fan" viewBox="0 0 ${W} ${H}" role="img" aria-label="Прогноз остатка: коридор P10–P90 и медиана P50">
<polygon points="${band}" class="band"/><line x1="${L}" x2="${W - R}" y1="${y(0).toFixed(1)}" y2="${y(0).toFixed(1)}" class="zero"/>
<polyline points="${median}" class="median"/>
<text x="${L}" y="${H - 6}">${esc(ruDate(f[0].date))}</text><text x="${W - R}" y="${H - 6}" text-anchor="end">${esc(ruDate(f[f.length - 1].date))}</text>
<text x="${W - R}" y="${(y(0) - 4).toFixed(1)}" text-anchor="end" class="zl">0 ₽</text>
<text x="${L + 2}" y="${T + 10}">${esc(mln(hi))}</text><text x="${L + 2}" y="${H - B - 3}">${esc(mln(lo))}</text></svg>`;
}

const tile = (label: string, value: string, sub = '') => `<div class="tile"><div class="tl">${esc(label)}</div><div class="tv">${esc(value)}</div>${sub ? `<div class="ts">${esc(sub)}</div>` : ''}</div>`;

// ---------------------------------------------------------------- forecast part
const companies = fc.companies;
const counts = fc.status_counts;
const allDates = companies.flatMap(c => [c.period.start, c.period.end]).sort();
// Statements that ended long before the file end (closed accounts) are not «the most risky now»
const current = companies.filter(c => c.daysBeforeFileEnd <= 7);
const byRisk = [...current].sort((a, b) => b.probability - a.probability);
const endedEarly = companies.length - current.length;

const forecastRows = companies.map(c => `<tr data-status="${c.status}">
<td data-v="${esc(c.id)}"><a href="#c-${esc(c.id)}" class="cid">${esc(c.id)}</a></td>
<td data-v="${{ CRITICAL: 2, WARNING: 1, STABLE: 0 }[c.status]}">${chip(c.status)}</td>
<td class="num" data-v="${c.probability}">${fixed(c.probability, 1)}%</td>
<td class="num" data-v="${c.currentBalance}">${rub(c.currentBalance)}</td>
<td class="num" data-v="${c.minP10}">${rub(c.minP10)} <span class="muted">${ruDate(c.minP10Date)}</span></td>
<td class="num" data-v="${c.minP50}">${rub(c.minP50)}</td>
<td data-v="${c.firstRiskDate ?? '9999'}">${ruDate(c.firstRiskDate)}</td>
<td data-v="${c.period.end}" title="Выписка с ${ruDate(c.period.start)}">${ruDate(c.period.end)}${c.daysBeforeFileEnd > 7 ? `<div class="muted">за ${c.daysBeforeFileEnd} дн. до конца файла</div>` : ''}</td>
<td class="num" data-v="${c.operations}">${c.operations.toLocaleString('ru-RU')}</td>
</tr>`).join('\n');

const companyDetails = companies.map(c => `<details id="c-${esc(c.id)}" class="company">
<summary><b>${esc(c.id)}</b> ${chip(c.status)} вероятность разрыва ${fixed(c.probability, 1)}% · остаток ${rub(c.currentBalance)} · худший P10 ${rub(c.minP10)}</summary>
<div class="cbody">
${fanChart(c.forecast, c.currentBalance)}
<div class="cols">
<div><h4>Журнал расчёта</h4>${stepsHtml(c.log)}</div>
<div><h4>Прогноз по дням</h4><table class="days"><thead><tr><th>Дата</th><th>P10</th><th>P50</th><th>P90</th><th>Ниже 0</th></tr></thead><tbody>
${c.forecast.map(p => `<tr class="${p.gap_probability_pct >= 10 || p.balance_p50 < 0 ? 'risk' : ''}"><td>${ruDate(p.date)}</td><td class="num">${rub(p.balance_p10)}</td><td class="num">${rub(p.balance_p50)}</td><td class="num">${rub(p.balance_p90)}</td><td class="num">${fixed(p.gap_probability_pct, 1)}%</td></tr>`).join('')}
</tbody></table></div>
</div></div></details>`).join('\n');

// ---------------------------------------------------------------- metrics part
function metricsHtml(): string {
  if (!m) {
    return `<p class="note">Бэктест не выполнен${backtestError ? `: ${esc(backtestError)}` : ''}. Для метрик нужна история не меньше 90 дней до даты среза и 30 дней факта после неё.</p>`;
  }
  const ci = m.ci95_bootstrap ?? {};
  const st = m.status;
  const keys = ['warning', 'critical'] as const;
  const a = m.regression.analog;
  const prob = m.probability;
  const lt = m.lead_time;
  const cm = (k: 'warning' | 'critical') => {
    const s = st[k];
    return `<div class="cmwrap"><h4>Статус ${esc(s.name)} <span class="muted">(вероятность ≥ ${s.threshold}%)</span></h4>
<table class="cm"><tr><th></th><th>Сигнал</th><th>Нет сигнала</th></tr>
<tr><th>Разрыв был</th><td class="good"><b>${s.tp}</b><span>TP — предсказан</span></td><td class="bad"><b>${s.fn}</b><span>FN — пропуск</span></td></tr>
<tr><th>Разрыва не было</th><td class="bad"><b>${s.fp}</b><span>FP — ложная тревога</span></td><td class="good"><b>${s.tn}</b><span>TN — спокойно</span></td></tr></table></div>`;
  };
  const scoreRow = (label: string, key: string, formula: string) => `<tr><th>${label}<div class="muted">${formula}</div></th>${keys.map(k => {
    const v = st[k][key];
    return `<td class="num"><b>${fixed(v)}</b><div class="muted">${ciText(ci[`${k}_${key}`] ?? st[k][`${key}_ci95`], v2 => fixed(v2))}</div></td>`;
  }).join('')}</tr>`;
  const regRows = (['analog', 'naive', 'trend'] as const).map(k => {
    const r = m.regression[k];
    return `<tr><th>${esc(r.name)}</th><td class="num">${rub(r.mae)}</td><td class="num">${sci(r.mse)} ₽²</td><td class="num">${rub(r.rmse)}</td><td class="num">${pct(r.wape)}</td><td class="num">${rub(r.bias)}</td><td class="num">${rub(r.median_ae)}</td></tr>`;
  }).join('');
  const horizonDays = [1, 7, 14, 21, 30].filter(h => h <= m.horizon_days);
  const thresholdRows = m.classification.analog.by_threshold.map((r: Record<string, number | null>) =>
    `<tr${r.threshold === 10 || r.threshold === 50 ? ' class="hl"' : ''}><td class="num">${r.threshold}%</td><td class="num">${r.signals}</td><td class="num">${r.tp}</td><td class="num">${r.fp}</td><td class="num">${r.fn}</td><td class="num">${r.tn}</td>` +
    `<td class="num">${fixed(r.accuracy)}</td><td class="num">${fixed(r.balanced_accuracy)}</td><td class="num">${fixed(r.precision)}</td><td class="num">${fixed(r.recall)}</td><td class="num">${fixed(r.specificity)}</td><td class="num">${fixed(r.f1)}</td></tr>`
  ).join('');
  const calRows = prob.calibration.map((b: { bin: string; forecasts: number; gaps: number; mean_predicted: number | null; observed: number | null; observed_ci95: [number, number] | null }) =>
    `<tr><td>${esc(b.bin)}</td><td class="num">${b.forecasts}</td><td class="num">${pct(b.mean_predicted)}</td><td class="num">${b.gaps}</td><td class="num">${pct(b.observed)}</td><td class="num muted">${ciText(b.observed_ci95, v => pct(v))}</td></tr>`
  ).join('');
  const sourceRows = m.by_source
    ? Object.entries(m.by_source).map(([name, b]: [string, any]) =>
      `<tr><th>${esc(name)}</th><td class="num">${b.companies}</td><td class="num">${b.forecasts}</td><td class="num">${b.new_gaps}</td><td class="num">${fixed(b.status.warning.accuracy)}</td><td class="num">${fixed(b.status.warning.precision)}</td><td class="num">${fixed(b.status.warning.recall)}</td><td class="num">${fixed(b.status.warning.f1)}</td><td class="num">${fixed(b.probability.roc_auc)}</td><td class="num">${fixed(b.probability.brier, 4)}</td><td class="num">${rub(b.regression.analog.mae)}</td><td class="num">${rub(b.regression.analog.rmse)}</td><td class="num">${pct(b.coverage_p10_p90.inside)}</td></tr>`
    ).join('')
    : '';
  const cutoffRows = Object.entries(m.per_cutoff).map(([cutoff, p]: [string, any]) =>
    `<tr><td>${ruDate(cutoff)}</td><td class="num">${p.forecasts}</td><td class="num">${p.already_negative}</td><td class="num">${p.new_gaps}</td><td class="num">${p.warning_signals}</td><td class="num">${p.warning_tp}</td><td class="num">${rub(p.mae_analog)}</td></tr>`
  ).join('');
  const leadText = (k: 'warning' | 'critical') => {
    const l = lt[k];
    return l.detected === 0
      ? `Статус ${esc(st[k].name)}: найденных разрывов нет.`
      : `Статус ${esc(st[k].name)}: найдено ${l.detected} из ${m.new_gaps}; предупреждение за ${l.min}–${l.max} дн., медиана <b>${fixed(l.median, 1)} дня</b>, среднее ${fixed(l.mean, 1)}; за 7+ дней — ${pct(l.share_7_plus, 0)}, за 14+ — ${pct(l.share_14_plus, 0)}.`;
  };

  return `
<p>Прогноз строится той же функцией, что на сайте, только по операциям до даты среза, и сравнивается с фактическим остатком следующих ${m.horizon_days} дней.
Срезы: ${m.cutoffs.map(ruDate).join(', ')}; факт с ${ruDate(m.actual_period.from)} по ${ruDate(m.actual_period.to)}.
${m.companies} ${plural(m.companies, 'компания', 'компании', 'компаний')}, ${m.forecasts} ${plural(m.forecasts, 'прогноз', 'прогноза', 'прогнозов')}.
Разрыв — ${esc(m.gap_definition)}; метрики разрыва — по ${m.gap_eval_forecasts} прогнозам, сделанным до разрыва (новых разрывов ${m.new_gaps}, доля ${pct(m.base_rate)}).</p>

<h3 id="confusion">Обнаружение кассового разрыва: матрица ошибок</h3>
<div class="cmgrid">${cm('warning')}${cm('critical')}</div>
<table class="metrics"><thead><tr><th>Метрика</th>${keys.map(k => `<th>${esc(st[k].name)}<div class="muted">≥ ${st[k].threshold}%</div></th>`).join('')}</tr></thead><tbody>
${scoreRow('Accuracy', 'accuracy', '(TP + TN) / всего')}
${scoreRow('Balanced accuracy', 'balanced_accuracy', '(Recall + Specificity) / 2')}
${scoreRow('Precision', 'precision', 'TP / (TP + FP) — доля верных сигналов')}
${scoreRow('Recall', 'recall', 'TP / (TP + FN) — доля найденных разрывов')}
${scoreRow('Specificity', 'specificity', 'TN / (TN + FP) — спокойные месяцы без тревоги')}
${scoreRow('F1', 'f1', '2·TP / (2·TP + FP + FN)')}
</tbody></table>
<p class="note">Разрывы редкие (${pct(m.base_rate)}): прогноз «разрыва не будет никогда» дал бы accuracy ${pct(m.base_rate === null ? null : 1 - m.base_rate)}, не найдя ни одного разрыва.
Поэтому смотрим recall, precision, F1 и balanced accuracy. Интервалы — бутстрэп по компаниям (${m.bootstrap_rounds} повторов).</p>

<h3>Вероятность разрыва</h3>
<table class="metrics"><tbody>
<tr><th>ROC-AUC<div class="muted">вероятность, что прогноз перед разрывом получил оценку выше, чем прогноз без разрыва; 0,5 — угадывание</div></th><td class="num"><b>${fixed(prob.roc_auc)}</b><div class="muted">${ciText(ci.roc_auc, v => fixed(v))}</div></td></tr>
<tr><th>Average precision<div class="muted">площадь под кривой precision–recall; у угадывания = доля разрывов ${pct(prob.base_rate)}</div></th><td class="num"><b>${fixed(prob.average_precision)}</b></td></tr>
<tr><th>Brier score<div class="muted">Σ(p − факт)² / N, меньше — лучше</div></th><td class="num"><b>${fixed(prob.brier, 4)}</b><div class="muted">${ciText(ci.brier, v => fixed(v, 4))}</div></td></tr>
<tr><th>Brier эталона<div class="muted">постоянная вероятность = доля разрывов</div></th><td class="num">${fixed(prob.reference, 4)}</td></tr>
<tr><th>Brier skill score<div class="muted">1 − Brier / эталон; выше 0 — лучше эталона</div></th><td class="num">${fixed(prob.skill)}</td></tr>
</tbody></table>
<table class="grid"><thead><tr><th>Вероятность в прогнозе</th><th>Прогнозов</th><th>Обещано в среднем</th><th>Разрывов</th><th>Случилось</th><th>95% интервал</th></tr></thead><tbody>${calRows}</tbody></table>

<h3>Прогноз остатка</h3>
<table class="grid"><thead><tr><th>Модель</th><th>MAE</th><th>MSE</th><th>RMSE</th><th>WAPE</th><th>Смещение</th><th>Медианная ошибка</th></tr></thead><tbody>${regRows}</tbody></table>
<p>Для текущей модели: MAE ${rub(a.mae)} <span class="muted">(${ciText(ci.mae, rub)})</span>, RMSE ${rub(a.rmse)} <span class="muted">(${ciText(ci.rmse, rub)})</span>,
WAPE ${pct(a.wape)} — точность прогноза остатка 1 − WAPE = ${pct(a.wape === null ? null : 1 - a.wape)}.
Коридор P10–P90: факт внутри в <b>${pct(m.coverage_p10_p90.inside)}</b> точек (ожидание ≈ 80%), ниже P10 — ${pct(m.coverage_p10_p90.below_p10)}, выше P90 — ${pct(m.coverage_p10_p90.above_p90)}.</p>
<table class="grid"><thead><tr><th>День горизонта</th>${horizonDays.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>
<tr><th>MAE</th>${horizonDays.map(h => `<td class="num">${rub(a.mae_by_horizon[h - 1])}</td>`).join('')}</tr>
<tr><th>RMSE</th>${horizonDays.map(h => `<td class="num">${rub(a.rmse_by_horizon[h - 1])}</td>`).join('')}</tr>
<tr><th>Факт внутри P10–P90</th>${horizonDays.map(h => `<td class="num">${pct(m.coverage_p10_p90.inside_by_horizon[h - 1])}</td>`).join('')}</tr>
</tbody></table>

<h3>Заблаговременность предупреждения</h3>
<p>${leadText('warning')}<br>${leadText('critical')}</p>

<h3>По порогам вероятности</h3>
<table class="grid small"><thead><tr><th>Порог</th><th>Сигналов</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>Accuracy</th><th>Balanced</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th></tr></thead><tbody>${thresholdRows}</tbody></table>
${sourceRows ? `<h3>По файлам</h3><table class="grid small"><thead><tr><th>Файл</th><th>Компаний</th><th>Прогнозов</th><th>Разрывов</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Brier</th><th>MAE</th><th>RMSE</th><th>P10–P90</th></tr></thead><tbody>${sourceRows}</tbody></table><p class="muted">Классификация — для статуса «Повышенный риск» и выше.</p>` : ''}
<h3>По датам среза</h3>
<table class="grid small"><thead><tr><th>Срез</th><th>Прогнозов</th><th>Уже в минусе</th><th>Новых разрывов</th><th>Сигналов «Повышенный риск»</th><th>Из них разрыв</th><th>MAE</th></tr></thead><tbody>${cutoffRows}</tbody></table>`;
}

const CHARTS: Array<[string, string]> = [
  ['6_confusion_matrix.png', 'Матрица ошибок двух статусов'],
  ['7_probability_quality.png', 'ROC-кривая и калибровка вероятности'],
  ['1_precision_recall_by_threshold.png', 'Precision и recall по порогу вероятности'],
  ['2_precision_recall_curve.png', 'Кривая precision–recall'],
  ['8_lead_time.png', 'Заблаговременность предупреждения'],
  ['3_mae_by_horizon.png', 'MAE по горизонту'],
  ['4_rmse_by_horizon.png', 'RMSE по горизонту'],
  ['5_mae_rmse_summary.png', 'MAE и RMSE по моделям'],
  ['9_band_coverage.png', 'Попадание факта в коридор P10–P90'],
];
const chartsHtml = CHARTS.filter(([f]) => fs.existsSync(path.join(dir, f)))
  .map(([f, caption]) => `<figure><img alt="${esc(caption)}" src="data:image/png;base64,${fs.readFileSync(path.join(dir, f)).toString('base64')}"><figcaption>${esc(caption)} · ${esc(f)}</figcaption></figure>`)
  .join('\n');

const tiles = m ? [
  tile('Recall', fixed(m.status.warning.recall), `найдено ${m.status.warning.tp} из ${m.new_gaps} разрывов («Повышенный риск»)`),
  tile('Precision', fixed(m.status.warning.precision), `${m.status.warning.tp} верных из ${m.status.warning.signals} сигналов`),
  tile('F1', fixed(m.status.warning.f1), `«Критический»: ${fixed(m.status.critical.f1)}`),
  tile('Accuracy', fixed(m.status.warning.accuracy), `balanced ${fixed(m.status.warning.balanced_accuracy)}`),
  tile('ROC-AUC', fixed(m.probability.roc_auc), '0,5 — угадывание'),
  tile('Brier score', fixed(m.probability.brier, 4), `эталон ${fixed(m.probability.reference, 4)}`),
  tile('MAE', rub(m.regression.analog.mae), `наивная модель ${rub(m.regression.naive.mae)}`),
  tile('RMSE', rub(m.regression.analog.rmse), `MSE ${sci(m.regression.analog.mse)} ₽²`),
  tile('WAPE', pct(m.regression.analog.wape), 'ошибка от величины остатка'),
  tile('Коридор P10–P90', pct(m.coverage_p10_p90.inside), 'факт внутри, ожидание ≈ 80%'),
  tile('Заблаговременность', m.lead_time.warning.median === null ? '—' : `${fixed(m.lead_time.warning.median, 1)} дня`, 'медиана до первого дня с минусом'),
].join('') : '';

const generated = new Date(fc.generated_at).toLocaleString('ru-RU', { dateStyle: 'long', timeStyle: 'short' });
const html = `<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinRadar — отчёт: ${esc(fc.files.join(', '))}</title>
<style>
:root{--text:#0b0b0b;--text2:#52514e;--line:#e4e3df;--bg:#fcfcfb;--card:#ffffff;--blue:#2a78d6;--crit:#b3261e;--critbg:#fdecea;--warn:#8a5300;--warnbg:#fff3dc;--ok:#1b6e43;--okbg:#e6f4ec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 "Segoe UI",system-ui,-apple-system,Arial,sans-serif}
header{background:#13233f;color:#fff;padding:28px 32px 18px}header h1{margin:0 0 6px;font-size:26px}header p{margin:0;color:#c9d4e6}
nav{position:sticky;top:0;z-index:5;background:#13233f;padding:8px 32px 12px;display:flex;gap:18px;flex-wrap:wrap}nav a{color:#fff;text-decoration:none;font-weight:600;font-size:14px;opacity:.9}nav a:hover{opacity:1;text-decoration:underline}
main{max-width:1240px;margin:0 auto;padding:8px 32px 48px}section{margin-top:34px}h2{font-size:22px;margin:0 0 12px;border-bottom:2px solid var(--line);padding-bottom:6px}h3{font-size:18px;margin:26px 0 10px}h4{margin:0 0 6px;font-size:15px}
.muted{color:var(--text2);font-weight:400;font-size:12.5px}.note{color:var(--text2);font-size:14px}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.tl{color:var(--text2);font-size:13px}.tv{font-size:24px;font-weight:700;margin:2px 0}.ts{color:var(--text2);font-size:12.5px}
.chip{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12.5px;font-weight:600;white-space:nowrap}.chip.crit{background:var(--critbg);color:var(--crit)}.chip.warn{background:var(--warnbg);color:var(--warn)}.chip.ok{background:var(--okbg);color:var(--ok)}
table{border-collapse:collapse;background:var(--card)}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}thead th{background:#f3f2ef;font-size:13px}
table.grid,table.metrics,table.companies{width:100%}table.small td,table.small th{padding:5px 8px;font-size:13.5px}tr.hl td{background:#eef4fc}
.companies th{cursor:pointer;user-select:none}.companies th:after{content:" ↕";color:#a5a39d;font-size:11px}.companies tbody tr:hover{background:#f6f8fc}
.scroll{overflow-x:auto}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0 12px}.controls input{padding:7px 10px;border:1px solid var(--line);border-radius:8px;font:inherit;min-width:220px}.controls button{padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:var(--card);font:inherit;font-size:13.5px;cursor:pointer}.controls button.on{background:#13233f;color:#fff;border-color:#13233f}
.cmgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;margin-bottom:14px}.cm td{width:150px;text-align:center}.cm td b{display:block;font-size:26px}.cm td span{font-size:12px;color:var(--text2)}.cm td.good{background:#e8f1fc}.cm td.bad{background:#fbefe9}
figure{margin:0 0 22px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}figure img{width:100%;height:auto;display:block}figcaption{color:var(--text2);font-size:13px;margin-top:6px}
.steps{padding-left:22px}.steps>li{margin-bottom:10px}.steps ul{margin:2px 0 0;padding-left:18px}.steps .sub{color:var(--text2);margin-left:14px}
details.company{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:8px}details.company summary{padding:10px 14px;cursor:pointer}details.company[open] summary{border-bottom:1px solid var(--line)}.cbody{padding:12px 14px}
.cols{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:22px}.days td,.days th{padding:3px 8px;font-size:13px}.days tr.risk td{background:var(--warnbg)}
.fan{width:100%;max-width:760px;height:auto;display:block;margin-bottom:10px}.fan .band{fill:#2a78d6;fill-opacity:.16}.fan .median{fill:none;stroke:#2a78d6;stroke-width:2}.fan .zero{stroke:#b3261e;stroke-dasharray:4 4}.fan text{font-size:11px;fill:#52514e}.fan .zl{fill:#b3261e}
footer{color:var(--text2);font-size:13px;margin-top:40px}code{background:#f0efec;padding:1px 5px;border-radius:4px}
@media (max-width:860px){.cols{grid-template-columns:1fr}main{padding:8px 16px 40px}header,nav{padding-left:16px;padding-right:16px}}
@media print{nav{display:none}details.company{break-inside:avoid}}
</style></head><body>
<header><h1>FinRadar: прогноз остатка и проверка качества</h1>
<p>Файлы: ${esc(fc.files.join(', '))} · ${companies.length} ${plural(companies.length, 'компания', 'компании', 'компаний')} · операции с ${ruDate(allDates[0])} по ${ruDate(allDates[allDates.length - 1])} · отчёт построен ${esc(generated)}</p></header>
<nav><a href="#summary">Итог</a><a href="#forecast">Прогноз по компаниям</a><a href="#metrics">Метрики качества</a><a href="#confusion">Матрица ошибок</a><a href="#charts">Графики</a><a href="#backtest-log">Журнал метрик</a><a href="#logs">Журналы расчёта по компаниям</a></nav>
<main>
<section id="summary"><h2>Итог</h2>
<div class="tiles">
${tile('Компаний', String(companies.length))}${tile('Критический', String(counts.CRITICAL ?? 0), 'вероятность ≥ 50% (без минуса за год — ≥ 80%)')}${tile('Повышенный риск', String(counts.WARNING ?? 0), 'вероятность ≥ 10% (без минуса за год — ≥ 60%)')}${tile('Стабильно', String(counts.STABLE ?? 0), 'ниже порогов')}
</div>
<h3>Самые рискованные компании</h3>${endedEarly > 0 ? `<p class="note">Без ${endedEarly} ${plural(endedEarly, 'компании', 'компаний', 'компаний')}, чья выписка закончилась больше чем за неделю до конца файла: их прогноз — от последней даты их выписки (см. общую таблицу).</p>` : ''}
<table class="grid"><thead><tr><th>Компания</th><th>Статус</th><th>Вероятность разрыва</th><th>Остаток сейчас</th><th>Худший P10</th><th>Первый рискованный день</th></tr></thead><tbody>
${byRisk.slice(0, 5).map(c => `<tr><td><a href="#c-${esc(c.id)}" class="cid">${esc(c.id)}</a></td><td>${chip(c.status)}</td><td class="num">${fixed(c.probability, 1)}%</td><td class="num">${rub(c.currentBalance)}</td><td class="num">${rub(c.minP10)} <span class="muted">${ruDate(c.minP10Date)}</span></td><td>${ruDate(c.firstRiskDate)}</td></tr>`).join('')}
</tbody></table>
${m ? `<h3>Качество на истории (бэктест)</h3><div class="tiles">${tiles}</div>` : ''}
<h3>Как посчитано</h3>
<ul>
<li><b>Прогноз.</b> Для каждой компании берутся 30-дневные отрезки истории, начинавшиеся в тот же день месяца (±3 дня); каждый отрезок, наложенный на текущий остаток, — сценарий.
P10 / P50 / P90 — перцентили сценариев по дням, вероятность разрыва — доля сценариев, где остаток хоть раз ниже нуля. Те же функции, что на сайте; шаги — в журнале каждой компании.</li>
<li><b>Метрики.</b> Бэктест: прогноз строится на конец месяца только по прошлым операциям и сравнивается с фактом следующих 30 дней. Каждое число выведено пошагово в разделе «Журнал метрик».</li>
<li><b>Файлы рядом с отчётом:</b> <code>forecast_companies.csv</code>, <code>forecast_days.csv</code>, <code>calc_log.md</code>${m ? ', <code>backtest_cases.csv</code> (каждый прогноз против факта), <code>backtest_log.md</code>, <code>backtest_metrics.json</code>' : ''}.</li>
</ul></section>

<section id="forecast"><h2>Прогноз по компаниям</h2>
<div class="controls"><input id="q" type="search" placeholder="Поиск по компании, например C052" aria-label="Поиск по компании">
<button data-f="all" class="on">Все</button><button data-f="CRITICAL">Критический</button><button data-f="WARNING">Повышенный риск</button><button data-f="STABLE">Стабильно</button></div>
<div class="scroll"><table class="companies grid" id="ct"><thead><tr><th>Компания</th><th>Статус</th><th>Вероятность разрыва</th><th>Остаток сейчас</th><th>Худший P10</th><th>Минимум P50</th><th>Первый рискованный день</th><th>Выписка по</th><th>Операций</th></tr></thead>
<tbody>${forecastRows}</tbody></table></div>
<p class="note">Нажмите на компанию: откроется её график, журнал расчёта и прогноз по дням. Первый рискованный день — ниже нуля не меньше 10% сценариев или медиана P50 ниже нуля.</p></section>

<section id="metrics"><h2>Метрики качества (бэктест)</h2>${metricsHtml()}</section>
${chartsHtml ? `<section id="charts"><h2>Графики</h2>${chartsHtml}</section>` : ''}
${m ? `<section id="backtest-log"><h2>Журнал расчёта метрик</h2>${stepsHtml(m.log)}</section>` : ''}
<section id="logs"><h2>Журналы расчёта по компаниям</h2>${companyDetails}</section>
<footer>Отчёт собран командой <code>npm run evaluate</code> · scripts/report.ts</footer>
</main>
<script>
(function(){
  var table=document.getElementById('ct'),body=table.tBodies[0],rows=[].slice.call(body.rows),q=document.getElementById('q'),filter='all';
  function apply(){var s=q.value.trim().toLowerCase();rows.forEach(function(r){var ok=(filter==='all'||r.dataset.status===filter)&&(!s||r.cells[0].textContent.toLowerCase().indexOf(s)>=0);r.style.display=ok?'':'none';});}
  q.addEventListener('input',apply);
  [].forEach.call(document.querySelectorAll('.controls button'),function(b){b.addEventListener('click',function(){filter=b.dataset.f;[].forEach.call(document.querySelectorAll('.controls button'),function(x){x.classList.toggle('on',x===b);});apply();});});
  [].forEach.call(table.tHead.rows[0].cells,function(th,i){var dir=1;th.addEventListener('click',function(){dir=-dir;rows.sort(function(a,b){var x=a.cells[i].dataset.v,y=b.cells[i].dataset.v,nx=parseFloat(x),ny=parseFloat(y);var c=(!isNaN(nx)&&!isNaN(ny)&&String(nx)===x&&String(ny)===y)?nx-ny:String(x).localeCompare(String(y),'ru',{numeric:true});return c*dir;});rows.forEach(function(r){body.appendChild(r);});});});
  function openHash(){var id=decodeURIComponent(location.hash.slice(1));var el=id&&document.getElementById(id);if(el&&el.tagName==='DETAILS'){el.open=true;el.scrollIntoView();}}
  window.addEventListener('hashchange',openHash);openHash();
})();
</script>
</body></html>
`;

const out = path.join(dir, 'report.html');
fs.writeFileSync(out, html);
console.log(`Отчёт: ${path.resolve(out)}`);
