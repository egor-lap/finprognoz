#!/usr/bin/env python3
"""
Plots backtest metrics (precision/recall of cash-gap detection, MAE/RMSE of the balance forecast)
from reports/backtest_metrics.json into PNG files in presentation_assets/metrics/.

Run after the backtest:
    node scripts/backtest.ts <train.csv> <test.csv>
    python scripts/plot_backtest.py [metrics.json] [out_dir]
For a new file both steps are done by: npm run evaluate -- <file.csv>
"""
import json
import os
import sys
import textwrap
from datetime import date

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

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
# Precision from fewer signals is noise ("4 of 4" reads as 1.00), so it is not drawn
MIN_SIGNALS = 5

with open(IN_FILE, 'r', encoding='utf-8') as f:
    m = json.load(f)


def ru(iso):
    return date.fromisoformat(iso).strftime('%d.%m.%Y')


years = {c[:4] for c in m['cutoffs']}
CUTOFFS_TEXT = (
    f"{', '.join(date.fromisoformat(c).strftime('%d.%m') for c in m['cutoffs'])}.{m['cutoffs'][0][:4]}" if len(years) == 1
    else ', '.join(ru(c) for c in m['cutoffs'])  # cutoffs over two years: full dates
)
PERIOD = (
    f"Срезы прогноза: {CUTOFFS_TEXT}  ·  "
    f"факт за период {ru(m['actual_period']['from'])} – {ru(m['actual_period']['to'])}\n"
    f"{m['companies']} компаний, {m['forecasts']} прогнозов, горизонт {m['horizon_days']} дней после каждого среза"
)
GAP_NOTE = (
    f"Разрыв = {m['gap_definition']}. Считаются только прогнозы, сделанные до разрыва: {m['gap_eval_forecasts']} прогнозов, "
    f"новых разрывов {m['new_gaps']}. Ещё {m['already_negative']} прогнозов сделаны, когда остаток уже был в минусе, — это не прогноз, "
    f"они исключены. Precision показан только при ≥{MIN_SIGNALS} сигналах, полоса — 95% интервал."
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


def new_figure(title, subtitle=PERIOD, note=None):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(top=0.80, bottom=0.17 if note else 0.11, left=0.08, right=0.80)
    fig.text(0.08, 0.95, title, fontsize=15, fontweight='bold', color=TEXT)
    fig.text(0.08, 0.92, subtitle, fontsize=9.5, color=TEXT_2, va='top', linespacing=1.5)
    if note:
        fig.text(0.08, 0.03, textwrap.fill(note, 150), fontsize=8.5, color=TEXT_2)
    return fig, ax


def end_label(ax, x, y, text, color):
    # Direct label at the line end, in text ink with a colored marker for identity
    ax.plot([x], [y], 'o', color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False)
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords='offset points', va='center', fontsize=9, color=TEXT, annotation_clip=False)


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
    ax.annotate(f'случайное угадывание {BASE_RATE:.1%}', x_label, xytext=(0, 5), textcoords='offset points',
                ha='right' if x_label[0] > 1 else 'left', fontsize=8.5, color=TEXT_2)


GAP_MODELS = [('analog', 'аналоговые периоды')]


def plot_gap_charts():
    # 1. Precision (with 95% band) and recall by threshold: color = model, line style = metric
    fig, ax = new_figure('Precision и Recall обнаружения кассового разрыва по порогу вероятности', note=GAP_NOTE +
                         ' Сигнал — вероятность разрыва за 30 дней ≥ порога.')
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
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    save(fig, '2_precision_recall_curve.png')


if m['new_gaps'] > 0:
    plot_gap_charts()
else:
    print('В данных нет новых разрывов: графики precision / recall не строятся, они были бы пустыми')

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
        ax.annotate(f'{value:.2f}', (horizon[-1], label_y), xytext=(10, 0), textcoords='offset points', va='center',
                    fontsize=9, color=TEXT, annotation_clip=False)
    ax.set_xlabel('День горизонта прогноза (1 = следующий день после среза)')
    ax.set_ylabel(f"{'MAE' if idx == 0 else 'RMSE'}, млн ₽")
    ax.set_xlim(1, m['horizon_days'])
    ax.set_ylim(bottom=0)
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
        ax.annotate(f'{b.get_height():.2f}', (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=9, color=TEXT)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('млн ₽')
ax.grid(axis='x', visible=False)
ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.25, 1))
save(fig, '5_mae_rmse_summary.png')
