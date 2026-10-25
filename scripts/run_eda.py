#!/usr/bin/env python3
import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

TRAIN_PATH = 'data/raw/train.csv'
TEST_PATH = 'data/raw/test.csv'
OUT_DIR = 'data'
os.makedirs(OUT_DIR, exist_ok=True)

COLUMNS = [
    'company_id', 'operation_id', 'operation_date', 'amount',
    'direction', 'counterparty_inn', 'counterparty_name',
    'payment_purpose', 'balance_eod'
]

# The same rules as categorizeOperation in shared/statement.ts, so the slide and the site show the same
# categories: checked in order, the first match wins, otherwise revenue / supplier payment
OWNER_TRANSFER = r'собственн[а-яё]* средств|на личн[а-яё]* (?:сч[её]т|карт)|на сч[её]т физического лица|дивиденд|подотч[её]т|займа учредител'
OUTFLOW_RULES = [
    ('Выводы владельца и переводы на свои счета', OWNER_TRANSFER),
    ('Налоги и взносы (ЕНП 28-го числа)', r'единый налоговый|налогового органа|ндфл|страхов[а-яё]* взнос|уфк по|(?:^|[^а-яё])енп(?:$|[^а-яё])|(?:^|[^а-яё])фнс(?:$|[^а-яё])'),
    ('Зарплата: аванс', r'(?:заработн[а-яё]* плат|зарплат)[^.]*аванс|аванс[^.]*(?:заработн|зарплат|сотрудник)'),
    ('Зарплата: окончательный расчёт', r'заработн[а-яё]* плат|зарплат|по реестру'),
    ('Аренда помещений и оборудования', r'аренд'),
    ('Кредиты, лизинг и проценты', r'кредит|лизинг|займ|процент[а-яё]* за пользование'),
    ('Комиссии банка и РКО', r'комисси|ведение сч[её]та|обслуживание сч[её]та|(?:^|[^а-яё])рко(?:$|[^а-яё])'),
    ('Штрафы и взыскания', r'исполнительн[а-яё]* лист|взыскан|пристав|штраф|(?:^|[^а-яё])пени(?:$|[^а-яё])'),
]
INFLOW_RULES = [
    ('Собственные средства и займы владельца', r'собственн[а-яё]* средств|займа учредител|беспроцентного займа'),
    ('Кредиты банка', r'выдача кредита|кредит[а-яё]* .*транш'),
    ('Торговый эквайринг и СБП', r'эквайринг|qr-код|по картам|мерчант|(?:^|[^а-яёa-z])сбп(?:$|[^а-яёa-z])'),
    ('Возвраты и прочие поступления', r'возврат платежа|возврат гарантийного|страхового возмещения'),
]

INDUSTRY_PATTERNS = [
    ('Общепит', r'(кафе|ресторан|кофе|бар|пицца|бургер|кухн|пекарн|фуд)'),
    ('Продуктовая розница', r'(продукт|гастроном|супермаркет|молок|хлеб|мяс|бакале)'),
    ('Одежда', r'(одежд|обув|текстил|плать|костюм|ткани|бутик)'),
    ('Маркетплейсы', r'(вайлдберриз|wildberries|ozon|озон|маркетплейс|яндекс\.маркет)'),
    ('Строительство', r'(строй|бетон|кирпич|цемент|монтаж|отделк|арматур|кровл)'),
    ('IT-услуги', r'(ит|it|софт|разработк|хостинг|программ|сайт|лицензи|облак)'),
    ('Опт', r'(опт|дистрибью|поставк|трейд|склад|оптов)'),
    ('Автосервис', r'(автосервис|сто|шиномонтаж|автозапчаст|ремонт авто|автохим)'),
    ('Салоны красоты', r'(красот|парикмахер|барбер|маникюр|космет|спа|стилист)'),
    ('Грузоперевозки', r'(грузоперевоз|транспортн|логистик|экспедитор|доставк|тягач|платон)'),
    ('Мебельное производство', r'(мебел|фасад|дсп|фурнитур|столяр|диван|кухни)'),
    ('Сельское хозяйство', r'(агро|зерн|урожай|удобрен|семен|комбайн|ферм|элеватор)'),
    ('Гостиницы', r'(отел|гостиниц|хостел|проживан|номер|турбаз|курорт)'),
    ('Детские центры', r'(детск|развива|сад|лагер|обучени|школ|репетитор)'),
    ('Клиники', r'(клиник|медицин|стом|анализ|врач|пациент|фарм|лечени)')
]

def classify_industry(text_corpus):
    text = text_corpus.lower()
    for name, pat in INDUSTRY_PATTERNS:
        if re.search(pat, text):
            return name
    return 'Услуги и торговля (СМБ)'

def load_data(path, is_test=False):
    header = None if is_test else 0
    df = pd.read_csv(
        path,
        sep=';',
        encoding='utf-8-sig',
        names=COLUMNS if is_test else None,
        header=header,
        dtype={
            'company_id': str,
            'operation_id': str,
            'counterparty_inn': str,
            'counterparty_name': str,
            'payment_purpose': str
        },
        low_memory=False
    )
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
    df['operation_date'] = pd.to_datetime(df['operation_date'], format='%d.%m.%Y')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    df['balance_eod'] = pd.to_numeric(df['balance_eod'], errors='coerce').fillna(0.0)
    df['counterparty_inn'] = df['counterparty_inn'].fillna('БЕЗ_ИНН').str.strip()
    df['counterparty_name'] = df['counterparty_name'].fillna('Неизвестный контрагент').str.strip()
    df['payment_purpose'] = df['payment_purpose'].fillna('').str.strip()
    df['split'] = 'test' if is_test else 'train'
    df = df.dropna(subset=['company_id'])
    df = df[df['company_id'].str.startswith('C')]
    return df

def categorize(df):
    text = df['payment_purpose'].str.lower()
    is_in = df['direction'] == 'Поступление'
    names = pd.Series(np.where(is_in, 'Выручка от покупателей', 'Оплата счетов поставщикам'), index=df.index)
    done = pd.Series(False, index=df.index)
    for direction_mask, rules in ((is_in, INFLOW_RULES), (~is_in, OUTFLOW_RULES)):
        for name, pattern in rules:
            hit = direction_mask & ~done & text.str.contains(pattern, regex=True)
            names[hit] = name
            done |= hit
    return names

def parse_delay_days(row):
    text = row['payment_purpose'] or ''
    m = re.search(r'от\s+(\d{2}\.\d{2}\.\d{4})', text)
    if m:
        try:
            inv_date = pd.to_datetime(m.group(1), format='%d.%m.%Y')
            delay = (row['operation_date'] - inv_date).days
            if 0 <= delay <= 180:
                return delay
        except Exception:
            pass
    return None

print("=== 1. Loading Train and Test datasets ===")
df_train = load_data(TRAIN_PATH, is_test=False)
df_test = load_data(TEST_PATH, is_test=True)
df_all = pd.concat([df_train, df_test], ignore_index=True)
print(f"Total rows: Train={len(df_train):,}, Test={len(df_test):,}, Total={len(df_all):,}")

print("Categorizing operations and extracting delays...")
df_all['category_name'] = categorize(df_all)
df_all['delay_days'] = df_all.apply(parse_delay_days, axis=1)

company_texts = df_all.groupby('company_id').apply(
    lambda g: ' '.join(g['payment_purpose'].dropna().tail(100) + ' ' + g['counterparty_name'].dropna().tail(100))
)
company_industries = {cid: classify_industry(txt) for cid, txt in company_texts.items()}

# -------------------------------------------------------------
# 1. EDA Summary Statistics
# -------------------------------------------------------------
print("=== 2. Computing Exploratory Data Analysis (EDA) ===")
delays_series = df_all['delay_days'].dropna()

eda_summary = {
    'overall': {
        'total_operations': len(df_all),
        'train_operations': len(df_train),
        'test_operations': len(df_test),
        'total_companies': df_all['company_id'].nunique(),
        'train_companies': df_train['company_id'].nunique(),
        'test_companies': df_test['company_id'].nunique(),
        'date_min': df_all['operation_date'].min().strftime('%Y-%m-%d'),
        'date_max': df_all['operation_date'].max().strftime('%Y-%m-%d'),
        'total_inflow': float(df_all[df_all['direction'] == 'Поступление']['amount'].sum()),
        'total_outflow': float(df_all[df_all['direction'] == 'Списание']['amount'].sum()),
        'net_flow': float(df_all[df_all['direction'] == 'Поступление']['amount'].sum() - df_all[df_all['direction'] == 'Списание']['amount'].sum()),
        'avg_operation_amount': float(df_all['amount'].mean()),
        'median_operation_amount': float(df_all['amount'].median()),
        'p90_operation_amount': float(df_all['amount'].quantile(0.90)),
        'p99_operation_amount': float(df_all['amount'].quantile(0.99)),
        'max_operation_amount': float(df_all['amount'].max()),
    },
    'categories': [],
    'industries': [],
    'invoice_delays': {
        'count_with_invoice_date': int(len(delays_series)),
        'pct_with_invoice_date': round(len(delays_series) / len(df_all) * 100, 1),
        'mean_delay_days': round(float(delays_series.mean()), 1),
        'median_delay_days': int(delays_series.median()),
        'p25_delay_days': int(delays_series.quantile(0.25)),
        'p75_delay_days': int(delays_series.quantile(0.75)),
        'p90_delay_days': int(delays_series.quantile(0.90)),
        'distribution': [
            {'range': '0-7 дней (Экспресс)', 'count': int((delays_series <= 7).sum()), 'pct': round((delays_series <= 7).mean() * 100, 1)},
            {'range': '8-14 дней (Стандарт)', 'count': int(((delays_series > 7) & (delays_series <= 14)).sum()), 'pct': round(((delays_series > 7) & (delays_series <= 14)).mean() * 100, 1)},
            {'range': '15-30 дней (Отсрочка)', 'count': int(((delays_series > 14) & (delays_series <= 30)).sum()), 'pct': round(((delays_series > 14) & (delays_series <= 30)).mean() * 100, 1)},
            {'range': '31-60 дней (Задержка)', 'count': int(((delays_series > 30) & (delays_series <= 60)).sum()), 'pct': round(((delays_series > 30) & (delays_series <= 60)).mean() * 100, 1)},
            {'range': '>60 дней (Просрочка/Суд)', 'count': int((delays_series > 60).sum()), 'pct': round((delays_series > 60).mean() * 100, 1)},
        ]
    },
    'cash_gap_stats': {},
    'seasonality': [],
    'day_of_week': []
}

cat_grp = df_all.groupby(['category_name', 'direction']).agg(
    count=('amount', 'count'),
    total_amount=('amount', 'sum'),
    avg_amount=('amount', 'mean'),
    median_amount=('amount', 'median')
).reset_index()

total_in = eda_summary['overall']['total_inflow']
total_out = eda_summary['overall']['total_outflow']

for _, r in cat_grp.iterrows():
    base_sum = total_in if r['direction'] == 'Поступление' else total_out
    eda_summary['categories'].append({
        'name': r['category_name'],
        'direction': r['direction'],
        'count': int(r['count']),
        'total_amount': float(r['total_amount']),
        'avg_amount': round(float(r['avg_amount']), 2),
        'median_amount': round(float(r['median_amount']), 2),
        'share_pct': round(float(r['total_amount']) / base_sum * 100, 2)
    })

ind_series = pd.Series(company_industries)
for ind_name, count in ind_series.value_counts().items():
    comps = ind_series[ind_series == ind_name].index
    ind_df = df_all[df_all['company_id'].isin(comps)]
    eda_summary['industries'].append({
        'industry': ind_name,
        'companies_count': int(count),
        'total_turnover': float(ind_df['amount'].sum()),
        'avg_company_turnover': round(float(ind_df['amount'].sum() / count), 0),
        'operations_count': int(len(ind_df))
    })

dow_names = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
df_all['dow'] = df_all['operation_date'].dt.dayofweek
for dow_idx in range(7):
    sub = df_all[df_all['dow'] == dow_idx]
    in_sum = sub[sub['direction'] == 'Поступление']['amount'].sum()
    out_sum = sub[sub['direction'] == 'Списание']['amount'].sum()
    eda_summary['day_of_week'].append({
        'dow': dow_idx,
        'day_name': dow_names[dow_idx],
        'operations_count': int(len(sub)),
        'inflow': float(in_sum),
        'outflow': float(out_sum)
    })

df_all['month_year'] = df_all['operation_date'].dt.to_period('M').astype(str)
for ym, grp in df_all.groupby('month_year'):
    in_sum = grp[grp['direction'] == 'Поступление']['amount'].sum()
    out_sum = grp[grp['direction'] == 'Списание']['amount'].sum()
    eda_summary['seasonality'].append({
        'month_year': ym,
        'inflow': float(in_sum),
        'outflow': float(out_sum),
        'net_flow': float(in_sum - out_sum),
        'count': int(len(grp))
    })

# Companies whose end-of-day balance was negative at least once in the statements
negative_balances = df_all[df_all['balance_eod'] < 0]
hist_gap_count = int(negative_balances['company_id'].nunique())
eda_summary['cash_gap_stats'] = {
    'total_overdraft_operations': int(len(negative_balances)),
    'companies_with_overdraft': hist_gap_count,
    'pct_companies_with_overdraft': round(hist_gap_count / df_all['company_id'].nunique() * 100, 1),
    'max_negative_balance': float(negative_balances['balance_eod'].min()) if not negative_balances.empty else 0.0,
    'avg_negative_balance': float(negative_balances['balance_eod'].mean()) if not negative_balances.empty else 0.0
}

with open(os.path.join(OUT_DIR, 'eda_summary.json'), 'w', encoding='utf-8', newline='\n') as f:
    json.dump(eda_summary, f, ensure_ascii=False, indent=2)

print("EDA summary saved to data/eda_summary.json")
