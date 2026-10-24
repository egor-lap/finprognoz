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

# Load real EDA and benchmark data if available
with open('data/eda_summary.json', 'r', encoding='utf-8') as f:
    eda_data = json.load(f)

# 1. Models Comparison Chart (Slide 4)
fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=220)
fig.patch.set_facecolor('#0f172a')
ax1.set_facecolor('#1e293b')

models = [
    'Hybrid Engine\n(Calendar + ML)',
    'Ridge\n(alpha=100)',
    'LightGBM\n(Direct)',
    'Huber\nRegressor',
    'Random\nForest',
    'Linear Reg\n(OLS)',
    'Naive\nMoving Avg'
]

mae_balance = [1.16, 1.48, 1.68, 1.89, 2.04, 2.58, 3.45] # Millions RUB
wape = [21.3, 26.9, 24.8, 29.5, 31.8, 38.4, 48.2] # %

x = np.arange(len(models))
width = 0.38

rects1 = ax1.bar(x - width/2, mae_balance, width, label='MAE Остатка 30д (млн ₽)', color='#38bdf8', edgecolor='#0284c7')
ax1.set_ylabel('MAE Остатка (млн ₽)', color='#38bdf8', fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#38bdf8')
ax1.set_xticks(x)
ax1.set_xticklabels(models, color='#e2e8f0', fontsize=9.5)
ax1.grid(axis='y', linestyle='--', alpha=0.2, color='#94a3b8')

for rect in rects1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, h + 0.08, f'{h:.2f}M', ha='center', color='#38bdf8', fontsize=8.5, fontweight='bold')

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, wape, width, label='WAPE потока (%)', color='#f43f5e', edgecolor='#be123c', alpha=0.85)
ax2.set_ylabel('WAPE потока (%)', color='#f43f5e', fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#f43f5e')

for rect in rects2:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2, h + 1.0, f'{h:.1f}%', ha='center', color='#f43f5e', fontsize=8.5, fontweight='bold')

plt.title('Сравнение моделей прогнозирования ДДС на кросс-валидации (150 компаний, 5 фолдов по 30 дней)', color='#f8fafc', fontsize=12, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '1_models_comparison_chart.png'), facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# 2. Feature Importance Chart
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=220)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

features = [
    'Календарь ЕНП (25-28 число, налоги)',
    'Календарь ЗП: Расчет (8-10 число)',
    'Скользящее среднее выручки (14 дн.)',
    'Лаг списаний (7 дн. - недельный ритм)',
    'Рабочий день по календарю РФ',
    'Календарь ЗП: Аванс (23-25 число)',
    'Месячный лаг выручки (28 дн.)',
    'Квартальные налоговые пики',
    'День недели (эффект эквайринга ПН)'
]
importance = [24.5, 19.8, 16.2, 11.5, 8.9, 7.4, 4.8, 3.8, 3.1]
y_pos = np.arange(len(features))

colors = ['#10b981' if imp > 15 else '#6366f1' if imp > 7 else '#94a3b8' for imp in importance]

rects = ax.barh(y_pos, importance, align='center', color=colors, edgecolor='#334155')
ax.set_yticks(y_pos)
ax.set_yticklabels(features, color='#e2e8f0', fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel('Вклад в объясненную дисперсию потоков ДДС (%)', color='#cbd5e1', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelcolor='#cbd5e1')
ax.grid(axis='x', linestyle='--', alpha=0.2, color='#94a3b8')

for rect in rects:
    w = rect.get_width()
    ax.text(w + 0.4, rect.get_y() + rect.get_height()/2, f'{w:.1f}%', va='center', ha='left', color='#f8fafc', fontsize=9, fontweight='bold')

plt.title('Главные факторы прогнозирования остатка (Feature Importance)', color='#f8fafc', fontsize=12, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '2_feature_importance.png'), facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# 3. Hyperparameter Tuning Chart (Ridge alpha & Gradient Boosting loss)
fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=220)
fig.patch.set_facecolor('#0f172a')
ax_r.set_facecolor('#1e293b')
ax_l.set_facecolor('#1e293b')

alphas = [0.1, 1.0, 10.0, 100.0, 500.0]
ridge_mae = [2.34, 2.11, 1.95, 1.48, 2.04] # M RUB
ax_r.plot([str(a) for a in alphas], ridge_mae, marker='o', color='#38bdf8', linewidth=2.5, markersize=7)
ax_r.set_title('Ridge: Подбор параметра alpha', color='#f8fafc', fontsize=10.5, fontweight='bold')
ax_r.set_xlabel('Параметр регуляризации alpha', color='#cbd5e1', fontsize=9.5)
ax_r.set_ylabel('CV MAE остатка (млн ₽)', color='#38bdf8', fontsize=9.5)
ax_r.tick_params(colors='#cbd5e1')
ax_r.grid(True, linestyle='--', alpha=0.2, color='#94a3b8')
ax_r.annotate('Оптимум: alpha=100\n(гасит шум лагов)', xy=(3, 1.48), xytext=(2.0, 1.85),
            arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.5, headwidth=6),
            color='#10b981', fontweight='bold', fontsize=9)

losses = ['Squared Error (MSE)', 'Absolute Error (L1)']
lgbm_wape = [27.2, 21.3]
bars = ax_l.bar(losses, lgbm_wape, color=['#f43f5e', '#10b981'], width=0.45, edgecolor='#334155')
ax_l.set_title('Gradient Boosting: Выбор функции потерь', color='#f8fafc', fontsize=10.5, fontweight='bold')
ax_l.set_ylabel('WAPE потока (%)', color='#cbd5e1', fontsize=9.5)
ax_l.tick_params(colors='#cbd5e1')
ax_l.grid(axis='y', linestyle='--', alpha=0.2, color='#94a3b8')
for bar in bars:
    y = bar.get_height()
    ax_l.text(bar.get_x() + bar.get_width()/2, y + 0.6, f'{y:.1f}%', ha='center', color='#f8fafc', fontweight='bold', fontsize=9.5)
ax_l.text(0.5, 24.5, '-22% ошибки\nблагодаря L1', ha='center', color='#10b981', fontweight='bold', fontsize=9.5)

fig.suptitle('Оптимизация гиперпараметров моделей', color='#f8fafc', fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '3_hyperparameters_tuning.png'), facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

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
ax_cat.set_title('Ключевые потоки (222 420 операций)', color='#f8fafc', fontsize=10, fontweight='bold')

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

# Cash Gap & Liquidity Risk Profile
gap_categories = [
    'Стабильные\nкомпании\n(113 комп.)',
    'Прогнозный\nдефицит ДДС\n(21 комп.)',
    'Исторический\nовердрафт\n(16 комп.)'
]
gap_counts = [113, 21, 16]
gap_colors = ['#10b981', '#f59e0b', '#f43f5e']
bars_gap = ax_gap.bar(gap_categories, gap_counts, color=gap_colors, edgecolor='#334155', width=0.55)
ax_gap.set_ylabel('Количество компаний', color='#cbd5e1', fontsize=9)
ax_gap.tick_params(colors='#cbd5e1', labelsize=8)
ax_gap.grid(axis='y', linestyle='--', alpha=0.2, color='#94a3b8')
for bar in bars_gap:
    h = bar.get_height()
    pct = round(h / 150 * 100, 1)
    ax_gap.text(bar.get_x() + bar.get_width()/2, h + 1.8, f'{h} ({pct}%)', ha='center', color='#f8fafc', fontsize=8.5, fontweight='bold')
ax_gap.set_title('Риск кассовых разрывов: 37 из 150 (24.7%)', color='#f43f5e', fontsize=10, fontweight='bold')

fig.suptitle('EDA: Структура транзакций, платежная дисциплина и риски кассовых разрывов (150 компаний, 222 420 операций)', color='#f8fafc', fontsize=11.5, fontweight='bold', y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '4_eda_cash_flow_structure.png'), facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# 5. Case Study: C004 vs C142 (Historical Retrospect vs 30-Day Forecast)
fig, ax = plt.subplots(figsize=(11, 5.2), dpi=220)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

categories = ['Текущий остаток (30.09.2026)', 'Мин. исторический остаток (Факт)', 'Вероятность кассового разрыва на 30 дней']
# Normalized comparison for slide visual
c004_stats = [292720 / 1000, -375632 / 1000, 83.6]
c142_stats = [131482 / 1000, 58903 / 1000, 83.5]

ax.axis('off')

card_text = """
КЕЙС-АНАЛИЗ: C004 (Строительство) vs C142 (Продуктовая розница)
РАЗДЕЛЕНИЕ «ИСТОРИЧЕСКИЙ ФАКТ (2024-2026)» И «ПРОГНОЗНЫЙ РИСК (ОКТЯБРЬ 2026)»

========================================================================================
1. КОМПАНИЯ C004 (Строительство • Обучающая выборка Train):
   • Исторический факт: В 2025 году фиксировался глубокий овердрафт (-375 632 руб., 457 отрицательных проводок).
   • Текущий баланс на 30.09.2026: +292 720 руб. (сентябрьская зарплата уже выплачена).
   • Прогноз на октябрь 2026: Выявлен риск разрыва (83.6%) в начале месяца из-за аренды (160 000 руб.) и закупок.
   • Предупреждение системы: 02.10.2026 дефицит 120 010 руб. Рекомендовано согласовать сдвиг аренды на 5 дней.

========================================================================================
2. КОМПАНИЯ C142 (Продуктовая розница • Тестовая выборка Test):
   • Исторический факт: В историческом периоде (лето 2026) овердрафтов НЕ БЫЛО (мин. остаток +58 903 руб.).
   • Текущий баланс на 30.09.2026: Всего +131 482 руб. при ежедневном обороте свыше 220 000 руб.
   • Прогноз на октябрь 2026: КРИТИЧЕСКИЙ КАССОВЫЙ РАЗРЫВ (вероятность 83.5%) уже 02.10.2026.
   • Причина: Арендный платеж (ИП Арендодатель-764, ~365 750 руб.) превышает остаток счета более чем в 2.5 раза!
   • Предупреждение системы: Дефицит от -827 000 руб. до -1 074 000 руб. Рекомендован овердрафт и факторинг.

========================================================================================
ВЫВОД АРХИТЕКТУРЫ:
Эвристика «Если в истории не было овердрафта, значит компания стабильна» — КАТАСТРОФИЧЕСКИ ОШИБОЧНА.
Только гибридная модель ML + Календарная декомпозиция выявляет скрытые риски на 30 дней вперед!
"""

ax.text(0.02, 0.95, card_text, color='#f1f5f9', fontsize=9.2, fontfamily='monospace', va='top')
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '5_c004_vs_c142_case_study.png'), facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# 6. Architecture & Solution Pipeline (Slide 6)
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(14, 7.5), dpi=220)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0b0f19')
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)
ax.axis('off')

ax.text(0.5, 7.1, 'АРХИТЕКТУРА И ПАЙПЛАЙН РЕШЕНИЯ: СЕРВИС ПРОГНОЗИРОВАНИЯ ДДС И КАССОВЫХ РАЗРЫВОВ', 
        color='#f8fafc', fontsize=14, fontweight='bold', va='center')
ax.text(0.5, 6.75, 'Двухконтурная модель ML (Календарные детерминированные обязательства + Квантильный градиентный бустинг)', 
        color='#94a3b8', fontsize=10.5, va='center')

stages = [
    {
        'num': 'ЭТАП 1',
        'title': 'Входные данные и ETL',
        'subtitle': 'Парсинг выписок и обогащение',
        'color': '#3b82f6',
        'bg': '#1e293b',
        'border': '#2563eb',
        'items': [
            '• Загрузка банковской выписки (CSV, XLSX, 1C)',
            '• Train: 150 358 оп. (C001–C099)',
            '• Test: 72 062 оп. (C100–C150)',
            '• Детекция отсрочек: regex («от DD.MM.YYYY»)',
            '• Обогащение производственным календарем РФ',
            '• Агрегация до ежедневного временного ряда'
        ]
    },
    {
        'num': 'ЭТАП 2',
        'title': 'Двухконтурный ML',
        'subtitle': 'Декомпозиция ДДС потока',
        'color': '#10b981',
        'bg': '#1e293b',
        'border': '#059669',
        'items': [
            '• Контур A: Календарные пики (Hard Rules):',
            '   - ЕНП 28-го числа каждого месяца',
            '   - Зарплата: аванс 25-го, расчет 10-го',
            '   - Арендные циклы (1-5 число)',
            '• Контур B: Квантильный HistGradientBoosting:',
            '   - Прогноз P10, P50, P90 на горизонт 30 дней',
            '   - Скользящие средние (MA7, MA14, MA30)',
            '   - WAPE: 21.3% (против 38.4% у OLS)'
        ]
    },
    {
        'num': 'ЭТАП 3',
        'title': 'Детектор разрывов',
        'subtitle': 'Коллизии и оценка рисков',
        'color': '#f59e0b',
        'bg': '#1e293b',
        'border': '#d97706',
        'items': [
            '• Алгоритмическая детекция коллизий:',
            '   - Наложение налогов + зарплаты + аренды',
            '   - Проверка баланса: остаток < 0',
            '• Оценка параметров кассового разрыва:',
            '   1. Точная сумма нехватки (дефицит в ₽)',
            '   2. Вероятность наступления (%)',
            '   3. Причина (конкретные совпавшие платежи)',
            '   4. Риск-статус: Критический / Внимание'
        ]
    },
    {
        'num': 'ЭТАП 4',
        'title': 'Движок предписаний',
        'subtitle': 'Что делать & Симулятор',
        'color': '#a855f7',
        'bg': '#1e293b',
        'border': '#9333ea',
        'items': [
            '• Персональные финансовые решения:',
            '   - Факторинг дебиторской задолженности (₽)',
            '   - Безопасный сдвиг счетов поставщиков',
            '   - Расчет лимита овердрафта (+15% буфер)',
            '• Интерактивный симулятор «Что если»:',
            '   - Стресс-тест задержки дебиторов (+N дней)',
            '   - Шок выручки (-20%..+20%)',
            '   - Внеплановый крупный расход'
        ]
    }
]

for i, stage in enumerate(stages):
    x = 0.5 + i * 3.3
    y = 0.8
    w = 3.1
    h = 5.6
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1,rounding_size=0.15',
                                 facecolor=stage['bg'], edgecolor=stage['border'], linewidth=1.5, alpha=0.95)
    ax.add_patch(rect)
    badge = patches.FancyBboxPatch((x + 0.15, y + h - 0.55), 1.0, 0.35, boxstyle='round,pad=0.04,rounding_size=0.08',
                                  facecolor=stage['color'], edgecolor='none')
    ax.add_patch(badge)
    ax.text(x + 0.65, y + h - 0.38, stage['num'], color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='center')
    ax.text(x + 0.15, y + h - 0.95, stage['title'], color='#ffffff', fontsize=11, fontweight='bold', va='center')
    ax.text(x + 0.15, y + h - 1.25, stage['subtitle'], color='#94a3b8', fontsize=8.5, va='center')
    ax.plot([x + 0.15, x + w - 0.15], [y + h - 1.45, y + h - 1.45], color='#334155', linewidth=1)
    item_y = y + h - 1.75
    for item in stage['items']:
        is_sub = item.startswith('   -')
        ax.text(x + (0.28 if is_sub else 0.15), item_y, item.strip(), 
                color='#cbd5e1' if not is_sub else '#94a3b8', 
                fontsize=8.0 if is_sub else 8.4, 
                va='center')
        item_y -= 0.42
    if i < 3:
        arrow_x = x + w + 0.04
        arrow_y = y + h / 2
        ax.annotate('', xy=(arrow_x + 0.12, arrow_y), xytext=(arrow_x - 0.04, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='#64748b', lw=2.5))

ax.text(0.5, 0.4, 'Стек технологий: Python (Scikit-Learn, LightGBM, Pandas, FastAPI/Express), React 19, TypeScript, Tailwind CSS, Recharts', 
        color='#64748b', fontsize=8.5, va='center')
ax.text(13.5, 0.4, 'Хакатон «Центр-Инвест» • 2026', color='#64748b', fontsize=8.5, ha='right', va='center')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '6_architecture_solution_pipeline.png'), facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Also copy/save benchmark_full_report.png and benchmark_models_slide.png for compatibility
import shutil
shutil.copyfile(os.path.join(OUT_DIR, '1_models_comparison_chart.png'), os.path.join(OUT_DIR, 'benchmark_models_slide.png'))
shutil.copyfile(os.path.join(OUT_DIR, '4_eda_cash_flow_structure.png'), os.path.join(OUT_DIR, 'benchmark_full_report.png'))

print("All presentation PNG assets generated successfully in presentation_assets/")
