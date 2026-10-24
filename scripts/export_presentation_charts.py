#!/usr/bin/env python3
"""
Generates high-resolution presentation chart assets for hackathon slides and reports.
Outputs PNG/JPEG images into presentation_assets/
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = 'presentation_assets'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 0.8

# Slides are built from the computed EDA summary (data/eda_summary.json).
# Model-quality charts come from the backtest: scripts/backtest.ts + scripts/plot_backtest.py
with open('data/eda_summary.json', 'r', encoding='utf-8') as f:
    eda_data = json.load(f)

n_ops = f"{eda_data['overall']['total_operations']:,}".replace(',', ' ')
n_companies = eda_data['overall']['total_companies']

# 4. EDA Insights: Category breakdown, Delays distribution & Cash Gap Risk
fig, (ax_cat, ax_del, ax_gap) = plt.subplots(1, 3, figsize=(16, 5.2), dpi=220)
fig.patch.set_facecolor('#0f172a')
ax_cat.set_facecolor('#1e293b')
ax_del.set_facecolor('#1e293b')
ax_gap.set_facecolor('#1e293b')

# Categories
top_cats = sorted(eda_data['categories'], key=lambda c: c['total_amount'], reverse=True)[:6]
cat_labels = [c['name'].split('(')[0].strip() for c in top_cats]
cat_amounts = [c['total_amount'] / 1e9 for c in top_cats]
cat_colors = ['#10b981' if c['direction'] == 'Поступление' else '#f43f5e' for c in top_cats]

bars_cat = ax_cat.barh(np.arange(len(top_cats)), cat_amounts, color=cat_colors, edgecolor='#334155')
ax_cat.set_yticks(np.arange(len(top_cats)))
ax_cat.set_yticklabels(cat_labels, color='#e2e8f0', fontsize=8.5)
ax_cat.invert_yaxis()
ax_cat.set_xlabel('Объем операций (млрд ₽)', color='#cbd5e1', fontsize=9)
ax_cat.tick_params(colors='#cbd5e1')
ax_cat.grid(axis='x', linestyle='--', alpha=0.2, color='#94a3b8')
for bar in bars_cat:
    w = bar.get_width()
    ax_cat.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.1f} млрд', va='center', ha='left', color='#f8fafc', fontsize=8, fontweight='bold')
ax_cat.set_title(f'Ключевые потоки ({n_ops} операций)', color='#f8fafc', fontsize=10, fontweight='bold')

# Delays distribution
delays = eda_data['invoice_delays']['distribution']
del_labels = [d['range'].split('(')[0].strip() for d in delays]
del_pcts = [d['pct'] for d in delays]
bars_del = ax_del.bar(del_labels, del_pcts, color='#6366f1', edgecolor='#4338ca', width=0.55)
ax_del.set_ylabel('Доля счетов (%)', color='#cbd5e1', fontsize=9)
ax_del.tick_params(colors='#cbd5e1')
ax_del.set_xticklabels(del_labels, rotation=15, ha='right', fontsize=8)
ax_del.grid(axis='y', linestyle='--', alpha=0.2, color='#94a3b8')
for bar in bars_del:
    h = bar.get_height()
    ax_del.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}%', ha='center', color='#f8fafc', fontsize=8, fontweight='bold')
ax_del.set_title(f'Отсрочка платежей (Ср: {eda_data["invoice_delays"]["mean_delay_days"]} дн.)', color='#f8fafc', fontsize=10, fontweight='bold')

# Companies whose balance went below zero at least once in the statements
with_gap = eda_data['cash_gap_stats']['companies_with_overdraft']
gap_categories = [
    f'Остаток ни разу\nне уходил в минус\n({n_companies - with_gap} комп.)',
    f'Остаток уходил\nв минус\n({with_gap} комп.)'
]
gap_counts = [n_companies - with_gap, with_gap]
gap_colors = ['#10b981', '#f43f5e']
bars_gap = ax_gap.bar(gap_categories, gap_counts, color=gap_colors, edgecolor='#334155', width=0.55)
ax_gap.set_ylabel('Количество компаний', color='#cbd5e1', fontsize=9)
ax_gap.tick_params(colors='#cbd5e1', labelsize=8)
ax_gap.grid(axis='y', linestyle='--', alpha=0.2, color='#94a3b8')
for bar in bars_gap:
    h = bar.get_height()
    pct = round(h / n_companies * 100, 1)
    ax_gap.text(bar.get_x() + bar.get_width()/2, h + 1.8, f'{h} ({pct}%)', ha='center', color='#f8fafc', fontsize=8.5, fontweight='bold')
ax_gap.set_title(f"Кассовые разрывы в истории: {with_gap} из {n_companies} ({eda_data['cash_gap_stats']['pct_companies_with_overdraft']}%)", color='#f43f5e', fontsize=10, fontweight='bold')

fig.suptitle(f'EDA: Структура транзакций, платежная дисциплина и кассовые разрывы ({n_companies} компаний, {n_ops} операций)', color='#f8fafc', fontsize=11.5, fontweight='bold', y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '4_eda_cash_flow_structure.png'), facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

print("All presentation PNG assets generated successfully in presentation_assets/")
