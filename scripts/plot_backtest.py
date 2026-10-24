#!/usr/bin/env python3
"""
Plots backtest metrics (precision/recall of cash-gap detection, MAE/RMSE of the balance forecast)
from reports/backtest_metrics.json into PNG files in presentation_assets/metrics/.

Run after the backtest:
    node scripts/backtest.ts <train.csv> <test.csv>
    python scripts/plot_backtest.py
"""
import json
import os
import textwrap
from datetime import date

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

IN_FILE = 'reports/backtest_metrics.json'
OUT_DIR = 'presentation_assets/metrics'

# Validated categorical palette (light mode), fixed order by entity
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'
MODEL_COLORS = {
    'analog': '#2a78d6',  # current model
    'legacy': '#eb6834',  # previous sin/cos synthetic
    'naive': '#1baf7a',
    'trend': '#eda100',
}
METRIC_COLORS = {'precision': '#2a78d6', 'recall': '#eb6834', 'f1': '#1baf7a'}

with open(IN_FILE, 'r', encoding='utf-8') as f:
    m = json.load(f)


def ru(iso):
    return date.fromisoformat(iso).strftime('%d.%m.%Y')


PERIOD = (
    f"Срезы прогноза: {', '.join(date.fromisoformat(c).strftime('%d.%m') for c in m['cutoffs'])}.{m['cutoffs'][0][:4]}  ·  "
    f"факт за период {ru(m['actual_period']['from'])} – {ru(m['actual_period']['to'])}\n"
    f"{m['companies']} компаний, {m['forecasts']} прогнозов, горизонт {m['horizon_days']} дней после каждого среза"
)
GAP_NOTE = (
    f"Разрыв = {m['gap_definition']}. Фактических разрывов: {m['actual_gaps']} из {m['forecasts']} "
    f"({m['actual_gaps'] / m['forecasts']:.1%})."
)

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


# 1. Precision / recall / F1 by threshold
an = m['classification']['analog']['by_threshold']
lg = m['classification']['legacy']['by_threshold']
th = [r['threshold'] for r in an]
fig, ax = new_figure('Precision и Recall обнаружения кассового разрыва по порогу вероятности', note=GAP_NOTE +
                     ' Прогноз считается сигналом разрыва, если вероятность разрыва за 30 дней ≥ порога.')
series = [
    ('precision', [r['precision'] for r in an], 'Precision', '-'),
    ('recall', [r['recall'] for r in an], 'Recall', '-'),
    ('f1', [r['f1'] for r in an], 'F1', '-'),
]
for key, ys, label, ls in series:
    xs = [t for t, y in zip(th, ys) if y is not None]
    ys = [y for y in ys if y is not None]
    ax.plot(xs, ys, ls, color=METRIC_COLORS[key], linewidth=2, label=f'{label} — текущая модель')
    end_label(ax, xs[-1], ys[-1], f'{label} {ys[-1]:.2f}', METRIC_COLORS[key])
lg_prec = [(r['threshold'], r['precision']) for r in lg if r['precision'] is not None]
ax.plot([x for x, _ in lg_prec], [y for _, y in lg_prec], '--', color=TEXT_2, linewidth=1.5, label='Precision — прежняя модель (sin/cos)')
end_label(ax, lg_prec[-1][0], lg_prec[-1][1], f'Precision прежн. {lg_prec[-1][1]:.2f}', TEXT_2)
ax.set_xlabel('Порог вероятности разрыва, %')
ax.set_ylabel('Значение метрики')
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.86), frameon=False, fontsize=9)
save(fig, '1_precision_recall_by_threshold.png')

# 2. Precision-recall curve
fig, ax = new_figure('Кривая Precision–Recall: текущая и прежняя модель', note=GAP_NOTE +
                     ' Подписи точек — порог вероятности. Пунктир — доля разрывов (precision случайного угадывания).')
for key, rows, label in [('analog', an, 'Текущая модель (аналоговые периоды)'), ('legacy', lg, 'Прежняя модель (sin/cos)')]:
    pts = [(r['recall'], r['precision'], r['threshold']) for r in rows if r['precision'] is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=MODEL_COLORS[key], linewidth=2, markersize=4, label=label)
    if key == 'analog':
        for rec, prec, t in pts:
            if t in (10, 50, 90):
                ax.annotate(f'порог {t}%', (rec, prec), xytext=(-10, 0), textcoords='offset points', ha='right', va='center', fontsize=8.5, color=TEXT)
base = m['actual_gaps'] / m['forecasts']
ax.axhline(base, color=TEXT_2, linestyle=':', linewidth=1.2)
ax.annotate(f'доля разрывов {base:.1%}', (0.02, base), xytext=(0, 5), textcoords='offset points', fontsize=8.5, color=TEXT_2)
ax.set_xlabel('Recall (доля найденных разрывов)')
ax.set_ylabel('Precision (доля верных сигналов)')
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper right', frameon=False, fontsize=9)
save(fig, '2_precision_recall_curve.png')

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
    for key in ['analog', 'legacy', 'naive', 'trend']:
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
keys = ['analog', 'legacy', 'naive', 'trend']
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
