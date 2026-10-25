#!/usr/bin/env python3
"""
Plots the backtest metrics (backtest_metrics.json written by scripts/backtest.ts) into PNG files:
  1-2  precision / recall of cash-gap detection by probability threshold, precision-recall curve
  3-5  MAE / RMSE of the balance forecast by horizon day and overall, against two baselines
  6    confusion matrices of the site statuses («Повышенный риск» from 10%, «Критический» from 50%)
  7    quality of the gap probability: ROC curve and calibration
  8    lead time: days from the forecast to the first negative balance
  9    how often the actual balance falls inside the P10-P90 band, by horizon day

Run after the backtest:
    node scripts/backtest.ts <train.csv> <test.csv>
    python scripts/plot_backtest.py [metrics.json] [out_dir]
For a new file all steps are done by: npm run evaluate -- <file.csv>
"""
import json
import os
import sys
import textwrap
from datetime import date

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

IN_FILE = sys.argv[1] if len(sys.argv) > 1 else 'reports/backtest_metrics.json'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'presentation_assets/metrics'

# Validated categorical palette (light mode), fixed order by entity
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'
MODEL_COLORS = {
    'analog': '#2a78d6',  # current model
    'naive': '#1baf7a',
    'trend': '#eda100',
}
METRIC_COLORS = {'precision': '#2a78d6', 'recall': '#eb6834'}
BLUE = MODEL_COLORS['analog']
ORANGE = METRIC_COLORS['recall']
# Sequential ramp (one hue, light to dark) for the confusion matrix
RAMP_LIGHT = '#eaf1fb'
RAMP_DARK = '#1d5aa6'
# Precision from fewer signals is noise ("4 of 4" reads as 1.00), so it is not drawn
MIN_SIGNALS = 5

with open(IN_FILE, 'r', encoding='utf-8') as f:
    m = json.load(f)


def ru_date(iso):
    return date.fromisoformat(iso).strftime('%d.%m.%Y')


def ru(v, digits=2):
    """Number with a decimal comma: 0.873 -> '0,87'."""
    return '—' if v is None else f'{v:.{digits}f}'.replace('.', ',')


def ru_pct(v, digits=0):
    return '—' if v is None else f'{v * 100:.{digits}f}%'.replace('.', ',')


def plural(n, one, few, many):
    n10, n100 = abs(n) % 10, abs(n) % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return few
    return many


COMMA = FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))

years = {c[:4] for c in m['cutoffs']}
CUTOFFS_TEXT = (
    f"{', '.join(date.fromisoformat(c).strftime('%d.%m') for c in m['cutoffs'])}.{m['cutoffs'][0][:4]}" if len(years) == 1
    else ', '.join(ru_date(c) for c in m['cutoffs'])  # cutoffs over two years: full dates
)
PERIOD = (
    f"Срезы прогноза: {CUTOFFS_TEXT}  ·  "
    f"факт за период {ru_date(m['actual_period']['from'])} – {ru_date(m['actual_period']['to'])}\n"
    f"{m['companies']} {plural(m['companies'], 'компания', 'компании', 'компаний')}, "
    f"{m['forecasts']} {plural(m['forecasts'], 'прогноз', 'прогноза', 'прогнозов')}, горизонт {m['horizon_days']} дней после каждого среза"
)
GAP_NOTE = (
    f"Разрыв = {m['gap_definition']}. Считаются только прогнозы, сделанные до разрыва: {m['gap_eval_forecasts']}, "
    f"новых разрывов {m['new_gaps']}. Ещё {m['already_negative']} прогнозов сделаны, когда остаток уже был в минусе, — это не прогноз, "
    f"они исключены."
)
BASE_RATE = m['new_gaps'] / max(1, m['gap_eval_forecasts'])

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.facecolor': SURFACE,
    'figure.facecolor': SURFACE,
    'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT_2,
    'xtick.color': TEXT_2,
    'ytick.color': TEXT_2,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def new_figure(title, subtitle=PERIOD, note=None, ncols=1):
    fig, ax = plt.subplots(1, ncols, figsize=(11, 6.2))
    fig.subplots_adjust(top=0.80, bottom=0.17 if note else 0.11, left=0.08, right=0.80 if ncols == 1 else 0.97,
                        wspace=0.28)
    fig.text(0.08, 0.95, title, fontsize=15, fontweight='bold', color=TEXT)
    fig.text(0.08, 0.92, subtitle, fontsize=9.5, color=TEXT_2, va='top', linespacing=1.5)
    if note:
        fig.text(0.08, 0.03, textwrap.fill(note, 140), fontsize=8.5, color=TEXT_2)
    return fig, ax


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('saved', path)


def precision_points(rows):
    """(threshold, precision, ci_low, ci_high) for thresholds with at least MIN_SIGNALS signals."""
    return [(r['threshold'], r['precision'], *r['precision_ci95']) for r in rows
            if r['precision'] is not None and r['signals'] >= MIN_SIGNALS]


def base_rate_line(ax, x_label):
    ax.axhline(BASE_RATE, color=TEXT_2, linestyle=':', linewidth=1.2)
    ax.annotate(f'случайное угадывание {ru_pct(BASE_RATE, 1)}', x_label, xytext=(0, 5), textcoords='offset points',
                ha='right' if x_label[0] > 1 else 'left', fontsize=8.5, color=TEXT_2)


GAP_MODELS = [('analog', 'аналоговые периоды')]


def plot_gap_charts():
    # 1. Precision (with 95% band) and recall by threshold: color = model, line style = metric
    fig, ax = new_figure('Precision и Recall обнаружения кассового разрыва по порогу вероятности', note=GAP_NOTE +
                         f' Сигнал — вероятность разрыва за 30 дней ≥ порога. Precision показан при ≥{MIN_SIGNALS} сигналах, '
                         'полоса — 95% интервал.')
    for key, label in GAP_MODELS:
        rows = m['classification'][key]['by_threshold']
        pts = precision_points(rows)
        if pts:
            xs = [p[0] for p in pts]
            ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=MODEL_COLORS[key], alpha=0.12, linewidth=0)
            ax.plot(xs, [p[1] for p in pts], '-', color=MODEL_COLORS[key], linewidth=2, label=f'Precision — {label}')
        ax.plot([r['threshold'] for r in rows], [r['recall'] for r in rows], '--', color=MODEL_COLORS[key], linewidth=2,
                label=f'Recall — {label}')
    base_rate_line(ax, (99, BASE_RATE))
    ax.set_xlabel('Порог вероятности разрыва, %')
    ax.set_ylabel('Значение метрики')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(COMMA)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    save(fig, '1_precision_recall_by_threshold.png')

    # 2. Precision-recall curve (thresholds with enough signals only)
    fig, ax = new_figure('Кривая Precision–Recall: аналоговые периоды', note=GAP_NOTE +
                         ' Подписи точек — порог вероятности. Пунктир — precision случайного угадывания (доля новых разрывов).')
    for key, label in GAP_MODELS:
        rows = m['classification'][key]['by_threshold']
        keep = {p[0] for p in precision_points(rows)}
        pts = [(r['recall'], r['precision'], r['threshold']) for r in rows if r['threshold'] in keep]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=MODEL_COLORS[key], linewidth=2, markersize=4,
                label=label.capitalize())
        if key == 'analog':
            for rec, prec, t in pts:
                if t in (10, 30, 50):
                    ax.annotate(f'порог {t}%', (rec, prec), xytext=(6, 6), textcoords='offset points', fontsize=8.5, color=TEXT)
    base_rate_line(ax, (0.02, BASE_RATE))
    ax.set_xlabel('Recall (доля найденных разрывов)')
    ax.set_ylabel('Precision (доля верных сигналов)')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(COMMA)
    ax.yaxis.set_major_formatter(COMMA)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    save(fig, '2_precision_recall_curve.png')


def blend(share):
    """Color of the sequential ramp for a share 0..1."""
    a = [int(RAMP_LIGHT[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(RAMP_DARK[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * share):02x}' for x, y in zip(a, b))


def plot_confusion():
    # 6. Confusion matrices of the two site statuses: rows = fact, columns = forecast
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    fig.subplots_adjust(top=0.70, bottom=0.25, left=0.11, right=0.97, wspace=0.42)
    fig.text(0.06, 0.95, 'Матрица ошибок: статус прогноза против факта', fontsize=15, fontweight='bold', color=TEXT)
    fig.text(0.06, 0.92, PERIOD, fontsize=9.5, color=TEXT_2, va='top', linespacing=1.5)
    fig.text(0.06, 0.03, textwrap.fill(
        GAP_NOTE + ' Строки — факт, столбцы — прогноз. Цвет — доля внутри строки: сколько разрывов найдено и сколько спокойных '
                   'месяцев прошло без тревоги.', 150), fontsize=8.5, color=TEXT_2)
    for ax, key in zip(axes, ['warning', 'critical']):
        s = m['status'][key]
        cells = [[('TP', s['tp']), ('FN', s['fn'])], [('FP', s['fp']), ('TN', s['tn'])]]
        for i, row in enumerate(cells):
            row_total = sum(v for _, v in row)
            for j, (label, value) in enumerate(row):
                share = value / row_total if row_total else 0
                ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=blend(share), edgecolor=SURFACE, linewidth=4))
                ink = '#ffffff' if share > 0.5 else TEXT
                ax.text(j + 0.5, 1.58 - i, str(value), ha='center', va='center', fontsize=22, fontweight='bold', color=ink)
                ax.text(j + 0.5, 1.3 - i, f'{label} · {ru_pct(share)} строки', ha='center', va='center', fontsize=9, color=ink)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['Сигнал', 'Нет сигнала'], fontsize=10, color=TEXT)
        ax.xaxis.tick_top()
        ax.set_yticks([1.5, 0.5])
        ax.set_yticklabels(['Разрыв\nбыл', 'Разрыва\nне было'], fontsize=10, color=TEXT)
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        note = m.get('threshold_note', {}).get(key, f"вероятность ≥ {s['threshold']}%")
        ax.set_title(f"Статус {s['name']}\n({note})", fontsize=11, fontweight='bold', color=TEXT, pad=30)
        ax.text(1, -0.12, f"Accuracy {ru(s['accuracy'], 3)}  ·  Balanced accuracy {ru(s['balanced_accuracy'], 3)}\n"
                          f"Precision {ru(s['precision'], 3)}  ·  Recall {ru(s['recall'], 3)}\n"
                          f"F1 {ru(s['f1'], 3)}  ·  Specificity {ru(s['specificity'], 3)}",
                ha='center', va='top', fontsize=9.5, color=TEXT, linespacing=1.6, transform=ax.transData)
    save(fig, '6_confusion_matrix.png')


def plot_probability():
    # 7. ROC curve with the two status thresholds, and calibration of the probability
    p = m['probability']
    fig, (ax1, ax2) = new_figure(
        'Качество вероятности разрыва: ROC-кривая и калибровка', ncols=2,
        note=GAP_NOTE + f" Brier score {ru(p['brier'], 4)} при эталоне {ru(p['reference'], 4)} — постоянной вероятности, равной доле "
                        f"разрывов {ru_pct(p['base_rate'], 1)}. Average precision {ru(p['average_precision'], 3)}.")
    fig.subplots_adjust(bottom=0.24)  # two-line group labels under the calibration bars
    roc = m['roc_curve']
    ax1.plot([0, 1], [0, 1], ':', color=TEXT_2, linewidth=1.2)
    ax1.annotate('случайное угадывание', (0.62, 0.55), fontsize=8.5, color=TEXT_2, rotation=0)
    ax1.plot([r['fpr'] for r in roc], [r['tpr'] for r in roc], '-', color=BLUE, linewidth=2)
    for key, label in (('warning', '≥10%'), ('critical', '≥50%')):
        s = m['status'][key]
        fpr = s['fp'] / max(1, s['fp'] + s['tn'])
        ax1.plot([fpr], [s['recall']], 'o', color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax1.annotate(f'порог {label}', (fpr, s['recall']), xytext=(8, -12), textcoords='offset points', fontsize=9, color=TEXT)
    ax1.set_title(f"ROC-кривая, ROC-AUC = {ru(p['roc_auc'], 3)}", fontsize=11, color=TEXT, loc='left')
    ax1.set_xlabel('Доля ложных тревог (1 − specificity)')
    ax1.set_ylabel('Доля найденных разрывов (recall)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.03)
    ax1.xaxis.set_major_formatter(COMMA)
    ax1.yaxis.set_major_formatter(COMMA)

    # Calibration as paired bars: what the forecast promised and how often the gap happened, per probability group
    bins = [b for b in p['calibration'] if b['forecasts'] > 0]
    x = list(range(len(bins)))
    w = 0.38
    promised = [b['mean_predicted'] * 100 for b in bins]
    happened = [b['observed'] * 100 for b in bins]
    bp = ax2.bar([i - w / 2 - 0.01 for i in x], promised, w, color=BLUE, label='обещано (средняя вероятность)')
    bh = ax2.bar([i + w / 2 + 0.01 for i in x], happened, w, color=ORANGE, label='случилось (доля разрывов)')
    for bars in (bp, bh):
        for b in bars:
            ax2.annotate(f'{ru(b.get_height(), 1)}%', (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                         textcoords='offset points', ha='center', fontsize=8, color=TEXT)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b['bin']}\nn = {b['forecasts']}" for b in bins], fontsize=8.5)
    ax2.set_title('Калибровка: обещано против случившегося', fontsize=11, color=TEXT, loc='left')
    ax2.set_xlabel('Группа прогнозов по вероятности разрыва')
    ax2.set_ylabel('%')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='x', visible=False)
    ax2.legend(loc='upper left', frameon=False, fontsize=8.5)
    save(fig, '7_probability_quality.png')


def plot_lead_time():
    # 8. Days from the forecast to the first negative balance: each dot is one gap
    lt = m['lead_time']
    rows = [
        ('Найдены статусом\n«Повышенный риск»', lt['warning']['days'], BLUE, lt['warning']['median']),
        ('Найдены статусом\n«Критический»', lt['critical']['days'], BLUE, lt['critical']['median']),
        ('Пропущены\n(статус «Стабильно»)', lt['warning']['missed_days'], ORANGE, None),
    ]
    fig, ax = new_figure('Заблаговременность: за сколько дней до разрыва пришло предупреждение', note=GAP_NOTE +
                         ' Каждая точка — один разрыв: число дней от даты прогноза до первого дня с отрицательным остатком.')
    fig.subplots_adjust(left=0.2, right=0.95)
    for i, (label, days, color, median) in enumerate(rows):
        y = len(rows) - 1 - i
        stacked = {}
        for d in days:
            k = stacked.get(d, 0)
            stacked[d] = k + 1
            ax.plot([d], [y + 0.14 * k], 'o', color=color, markersize=10, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if median is not None:
            ax.plot([median, median], [y - 0.32, y + 0.32], '-', color=TEXT, linewidth=1.5)
            ax.annotate(f'медиана {ru(median, 1)} дн.', (median, y - 0.32), xytext=(0, -12), textcoords='offset points',
                        ha='center', fontsize=9, color=TEXT)
        if not days:
            ax.annotate('нет', (1, y), va='center', fontsize=9, color=TEXT_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5, color=TEXT)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, m['horizon_days'] + 1)
    ax.set_xlabel('Дней от прогноза до первого дня с отрицательным остатком')
    ax.grid(axis='y', visible=False)
    save(fig, '8_lead_time.png')


if m['new_gaps'] > 0:
    plot_gap_charts()
    plot_confusion()
    plot_probability()
    plot_lead_time()
else:
    print('В данных нет новых разрывов: графики precision / recall, матрица ошибок и ROC не строятся, они были бы пустыми')

# 3-4. MAE and RMSE by horizon day
reg = m['regression']
horizon = list(range(1, m['horizon_days'] + 1))
for idx, (metric, title, fname) in enumerate([
    ('mae_by_horizon', 'MAE прогноза остатка по горизонту (средняя абсолютная ошибка)', '3_mae_by_horizon.png'),
    ('rmse_by_horizon', 'RMSE прогноза остатка по горизонту (среднеквадратичная ошибка)', '4_rmse_by_horizon.png'),
]):
    fig, ax = new_figure(title, note='Ошибка = прогноз остатка на конец дня − фактический остаток, усреднение по всем компаниям и срезам. '
                                     'Для текущей модели используется медиана P50.')
    ends = []
    for key in ['analog', 'naive', 'trend']:
        ys = [v / 1e6 for v in reg[key][metric]]
        ax.plot(horizon, ys, '-', color=MODEL_COLORS[key], linewidth=2, label=reg[key]['name'])
        ax.plot([horizon[-1]], [ys[-1]], 'o', color=MODEL_COLORS[key], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False)
        ends.append([ys[-1], ys[-1]])
    # Spread end labels apart so close lines do not overlap
    top = max(max(v / 1e6 for v in reg[k][metric]) for k in reg)
    gap = top * 0.045
    order = sorted(range(len(ends)), key=lambda i: ends[i][0])
    for prev, cur in zip(order, order[1:]):
        ends[cur][1] = max(ends[cur][1], ends[prev][1] + gap)
    for value, label_y in ends:
        ax.annotate(ru(value), (horizon[-1], label_y), xytext=(10, 0), textcoords='offset points', va='center',
                    fontsize=9, color=TEXT, annotation_clip=False)
    ax.set_xlabel('День горизонта прогноза (1 = следующий день после среза)')
    ax.set_ylabel(f"{'MAE' if idx == 0 else 'RMSE'}, млн ₽")
    ax.set_xlim(1, m['horizon_days'])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(COMMA)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    save(fig, fname)

# 5. Overall MAE / RMSE by model
fig, ax = new_figure('MAE и RMSE прогноза остатка: сравнение подходов (среднее по дням 1–30)',
                     note='Меньше — лучше. Ошибки в рублях; на значения сильнее влияют крупные компании.')
keys = ['analog', 'naive', 'trend']
names = [textwrap.fill(reg[k]['name'], 22) for k in keys]
x = list(range(len(keys)))
w = 0.36
mae = [reg[k]['mae'] / 1e6 for k in keys]
rmse = [reg[k]['rmse'] / 1e6 for k in keys]
b1 = ax.bar([i - w / 2 - 0.01 for i in x], mae, w, color=METRIC_COLORS['precision'], label='MAE')
b2 = ax.bar([i + w / 2 + 0.01 for i in x], rmse, w, color=METRIC_COLORS['recall'], label='RMSE')
for bars in (b1, b2):
    for b in bars:
        ax.annotate(ru(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=9, color=TEXT)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('млн ₽')
ax.yaxis.set_major_formatter(COMMA)
ax.grid(axis='x', visible=False)
ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.25, 1))
save(fig, '5_mae_rmse_summary.png')

# 9. Share of facts inside the P10-P90 band by horizon day
cov = m['coverage_p10_p90']
fig, ax = new_figure('Коридор P10–P90: как часто фактический остаток попадает внутрь',
                     note=f"Для честного коридора факт внутри примерно в 80% случаев. В среднем по горизонту: внутри {ru_pct(cov['inside'], 1)}, "
                          f"ниже P10 — {ru_pct(cov['below_p10'], 1)}, выше P90 — {ru_pct(cov['above_p90'], 1)}.")
ys = [v * 100 for v in cov['inside_by_horizon']]
ax.plot(horizon, ys, '-', color=BLUE, linewidth=2)
ax.plot([horizon[-1]], [ys[-1]], 'o', color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False)
ax.annotate(f'{ru(ys[-1], 1)}%', (horizon[-1], ys[-1]), xytext=(10, 0), textcoords='offset points', va='center', fontsize=9,
            color=TEXT, annotation_clip=False)
ax.axhline(80, color=TEXT_2, linestyle=':', linewidth=1.2)
ax.annotate('ожидание 80%', (1.3, 80), xytext=(0, 5), textcoords='offset points', fontsize=8.5, color=TEXT_2)
ax.set_xlabel('День горизонта прогноза (1 = следующий день после среза)')
ax.set_ylabel('Факт внутри коридора, % прогнозов')
ax.set_xlim(1, m['horizon_days'])
ax.set_ylim(0, 100)
save(fig, '9_band_coverage.png')
