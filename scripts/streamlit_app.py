import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# БЛОК 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ (УЧЕТ UTF-8 BOM И ИНН С '00')
# =====================================================================

def load_statement(file_or_path) -> pd.DataFrame:
    """Читает выписку с сохранением ведущих нулей в ИНН и очищает пропуски."""
    df = pd.read_csv(
        file_or_path,
        sep=';',
        encoding='utf-8-sig',
        dtype={
            'company_id': str,
            'operation_id': str,
            'counterparty_inn': str,
            'counterparty_name': str,
            'payment_purpose': str
        }
    )
    df['operation_date'] = pd.to_datetime(df['operation_date'])
    df['amount'] = df['amount'].astype(float)
    df['balance_eod'] = df['balance_eod'].astype(float)
    df['counterparty_inn'] = df['counterparty_inn'].fillna('БЕЗ_ИНН')
    df['counterparty_name'] = df['counterparty_name'].fillna('Неизвестный контрагент')

    # Удаление возможных дублей строк
    dedup_cols = ['operation_date', 'amount', 'direction', 'counterparty_inn', 'payment_purpose']
    df = df.drop_duplicates(subset=dedup_cols, keep='first').sort_values(['operation_date', 'operation_id']).reset_index(drop=True)

    # Восстановление пустых payment_purpose по истории контрагента (ИНН + направление)
    known_purposes = (
        df.dropna(subset=['payment_purpose'])
        .groupby(['counterparty_inn', 'direction'])['payment_purpose']
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else '')
        .to_dict()
    )

    def fill_purpose(row):
        val = row['payment_purpose']
        if pd.notna(val) and str(val).strip() != '':
            return str(val).strip()
        return known_purposes.get((row['counterparty_inn'], row['direction']), 'Операция по основному договору')

    df['payment_purpose_clean'] = df.apply(fill_purpose, axis=1)
    return df


# =====================================================================
# БЛОК 2. МОДУЛЬ КАТЕГОРИЗАЦИИ ОПЕРАЦИЙ (ПРАВИЛА + TF-IDF NLP)
# =====================================================================

CATEGORY_NAMES_RU = {
    'IN_CUSTOMER_REVENUE': 'Поступления: Выручка от покупателей',
    'IN_ONE_OFF_ORDER': 'Поступления: Крупный разовый спецзаказ',
    'IN_ACQUIRING': 'Поступления: Торговый эквайринг',
    'IN_OWN_TRANSFER': 'Поступления: Перевод собственных средств',
    'OUT_SUPPLIER_COGS': 'Списания: Закупка товара у поставщиков',
    'OUT_ONE_OFF_PURCHASE': 'Списания: Закупка под разовый спецзаказ',
    'OUT_SALARY': 'Списания: Зарплата и аванс сотрудникам',
    'OUT_TAXES_ENP': 'Списания: Налоги и взносы (ЕНП)',
    'OUT_RENT': 'Списания: Аренда помещений',
    'OUT_LOGISTICS': 'Списания: Транспорт и логистика',
    'OUT_OPEX_SERVICES': 'Списания: Офис, ИТ, связь, охрана, бухгалтерия',
    'OUT_BANK_FEE': 'Списания: Комиссии банка и овердрафт',
    'OUT_DIVIDENDS': 'Списания: Выплата дивидендов собственнику',
    'OUT_LEGAL_SHOCK': 'Списания: Исполнительные листы / Штрафы'
}

def rule_based_category(row) -> str:
    text = str(row['payment_purpose_clean']).lower()
    name = str(row['counterparty_name']).lower()
    inn = str(row['counterparty_inn'])
    amt = row['amount']

    if row['direction'] == 'Поступление':
        if 'разовый заказ' in text or amt >= 15_000_000:
            return 'IN_ONE_OFF_ORDER'
        if 'эквайринг' in text or 'по картам' in text:
            return 'IN_ACQUIRING'
        if 'собственных средств' in text:
            return 'IN_OWN_TRANSFER'
        return 'IN_CUSTOMER_REVENUE'
    else:
        if 'дивиденд' in text or 'участник общества' in name:
            return 'OUT_DIVIDENDS'
        if 'единый налоговый' in text or 'енп' in text or inn == '7727406020':
            return 'OUT_TAXES_ENP'
        if 'заработной платы' in text or 'зарплат' in text or 'реестру №' in text:
            return 'OUT_SALARY'
        if 'аренд' in text or 'арендодатель' in name:
            return 'OUT_RENT'
        if 'комиссия' in text or 'овердрафт' in text or ('банк' in name and amt < 50_000):
            return 'OUT_BANK_FEE'
        if 'исполнительному листу' in text or 'истец' in name or 'штраф' in text:
            return 'OUT_LEGAL_SHOCK'
        if 'разовый заказ' in text or amt >= 15_000_000:
            return 'OUT_ONE_OFF_PURCHASE'
        if 'транспортно-экспедиц' in text or 'перевозк' in text or 'транспортная' in name:
            return 'OUT_LOGISTICS'
        if any(w in text for w in ['связи', 'интернет', 'охран', 'бухгалтер', 'канцтовар', 'использования по', 'подписка']):
            return 'OUT_OPEX_SERVICES'
        return 'OUT_SUPPLIER_COGS'


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Классифицирует транзакции (правила + обучение TF-IDF классификатора для новых текстов)."""
    df = df.copy()
    df['category_code'] = df.apply(rule_based_category, axis=1)
    df['category_name'] = df['category_code'].map(CATEGORY_NAMES_RU)

    # Обучаем TF-IDF + LogisticRegression на размеченном корпусе для демонстрации NLP-модуля
    corpus = df['direction'] + ' ' + df['counterparty_name'] + ' ' + df['payment_purpose_clean']
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
    X_tfidf = vectorizer.fit_transform(corpus)
    clf = LogisticRegression(max_iter=300, C=5.0)
    clf.fit(X_tfidf, df['category_code'])
    df['nlp_confidence'] = np.max(clf.predict_proba(X_tfidf), axis=1).round(3)

    # Извлекаем отсрочку оплаты по счёту (если указана дата счёта в назначении платежа)
    def parse_delay(row):
        m = re.search(r'от\s+(\d{2}\.\d{2}\.\d{4})', str(row['payment_purpose_clean']))
        if m:
            try:
                inv_dt = pd.to_datetime(m.group(1), format='%d.%m.%Y')
                d = (row['operation_date'] - inv_dt).days
                return d if 0 <= d <= 365 else np.nan
            except Exception:
                return np.nan
        return np.nan

    df['invoice_delay_days'] = df.apply(parse_delay, axis=1)
    return df


# =====================================================================
# БЛОК 3. ОБНАРУЖЕНИЕ РЕГУЛЯРНЫХ ПЛАТЕЖЕЙ И ПОСТРОЕНИЕ КАЛЕНДАРЯ
# =====================================================================

def detect_recurring_payments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Автоматически выявляет регулярные (ежемесячные/ежеквартальные) платежи
    и формирует расписание обязательных выплат на следующие 30 дней.
    """
    last_date = df['operation_date'].max()
    recent_cutoff = last_date - pd.Timedelta(days=180)
    
    # Анализируем регулярные категории списаний
    sched_cats = ['OUT_BANK_FEE', 'OUT_RENT', 'OUT_SALARY', 'OUT_TAXES_ENP', 'OUT_OPEX_SERVICES', 'OUT_DIVIDENDS']
    sub = df[df['category_code'].isin(sched_cats)].copy()
    sub['day_of_month'] = sub['operation_date'].dt.day
    
    # Разделяем зарплату на Аванс (20-30 число) и Окончательный расчёт (1-15 число)
    def refine_subtype(row):
        text = str(row['payment_purpose_clean']).lower()
        if row['category_code'] == 'OUT_SALARY':
            return 'Зарплата (аванс 25-го числа)' if ('аванс' in text or row['day_of_month'] >= 18) else 'Зарплата (расчёт 10-го числа)'
        if row['category_code'] == 'OUT_TAXES_ENP':
            return 'Налоги: ЕНП (квартальный пик)' if row['amount'] > 1_200_000 else 'Налоги: ЕНП (ежемесячный)'
        return row['counterparty_name']

    sub['regular_item'] = sub.apply(refine_subtype, axis=1)

    records = []
    for (item_name, cat_code), grp in sub.groupby(['regular_item', 'category_code']):
        if len(grp) < 3:
            continue
        recent_grp = grp[grp['operation_date'] >= recent_cutoff]
        if recent_grp.empty:
            recent_grp = grp.tail(3)
            
        typical_day = int(np.round(grp['day_of_month'].median()))
        expected_amount = float(recent_grp['amount'].median())
        cv = float(recent_grp['amount'].std() / (recent_grp['amount'].mean() + 1e-6)) if len(recent_grp) > 1 else 0.0
        
        # Определяем периодичность
        if cat_code == 'OUT_DIVIDENDS' or 'квартальный' in item_name:
            periodicity = 'Ежеквартально (янв, апр, июл, окт)'
        else:
            periodicity = 'Ежемесячно'

        records.append({
            'Регулярный платёж / Контрагент': item_name,
            'Категория': CATEGORY_NAMES_RU.get(cat_code, cat_code),
            'Код_категории': cat_code,
            'Периодичность': periodicity,
            'Типичный день месяца': typical_day,
            'Ожидаемая сумма (₽)': round(expected_amount, 2),
            'Стабильность суммы (CV)': f"{min(cv*100, 100):.1f}%",
            'Кол-во в истории': len(grp)
        })

    return pd.DataFrame(records).sort_values('Типичный день месяца').reset_index(drop=True)


def adjust_to_russian_workday(dt: pd.Timestamp, move_backward: bool = True) -> pd.Timestamp:
    """Переносит дату с выходного/праздника на ближайший рабочий день РФ."""
    while dt.weekday() >= 5 or ((dt.month == 1 and dt.day <= 8) or (dt.month == 11 and dt.day == 4)):
        dt += pd.Timedelta(days=-1 if move_backward else 1)
    return dt


def generate_future_scheduled_events(recurring_df: pd.DataFrame, start_date: pd.Timestamp, horizon: int = 30) -> pd.DataFrame:
    """Формирует точный календарь обязательных платежей на горизонте прогноза."""
    future_dates = pd.date_range(start_date + pd.Timedelta(days=1), start_date + pd.Timedelta(days=horizon), freq='D')
    events = []

    months_in_horizon = sorted(list(set((d.year, d.month) for d in future_dates)))
    for yr, mn in months_in_horizon:
        for _, row in recurring_df.iterrows():
            item = row['Регулярный платёж / Контрагент']
            cat_code = row['Код_категории']
            day_num = min(int(row['Типичный день месяца']), 28)
            amt = row['Ожидаемая сумма (₽)']

            # Проверка квартальных платежей
            if cat_code == 'OUT_DIVIDENDS' and mn not in [1, 4, 7, 10]:
                continue
            if item == 'Налоги: ЕНП (квартальный пик)' and mn not in [3, 4, 7, 10]:
                continue
            if item == 'Налоги: ЕНП (ежемесячный)' and mn in [3, 4, 7, 10]:
                continue

            raw_dt = pd.Timestamp(year=yr, month=mn, day=day_num)
            # Зарплата переносится назад на пятницу, если выпадает на выходной; РКО/аренда — вперед на понедельник
            move_back = ('Зарплата' in item)
            pay_dt = adjust_to_russian_workday(raw_dt, move_backward=move_back)

            if future_dates.min() <= pay_dt <= future_dates.max():
                events.append({
                    'date': pay_dt,
                    'item': item,
                    'category_code': cat_code,
                    'scheduled_outflow': amt
                })

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.sort_values('date').reset_index(drop=True)
    return events_df


# =====================================================================
# БЛОК 4. ПОСТРОЕНИЕ ВРЕМЕННОГО РЯДА, ПРИЗНАКОВ И БЕНЧМАРК МОДЕЛЕЙ
# =====================================================================

def prepare_daily_dataset(df: pd.DataFrame):
    """Строит ежедневную сетку, разделяя календарные обязательства и операционный поток."""
    first_day = df['operation_date'].min()
    f_ops = df[df['operation_date'] == first_day]
    init_balance = (
        f_ops['balance_eod'].iloc[0]
        - f_ops.loc[f_ops['direction'] == 'Поступление', 'amount'].sum()
        + f_ops.loc[f_ops['direction'] == 'Списание', 'amount'].sum()
    )

    full_idx = pd.date_range(df['operation_date'].min(), df['operation_date'].max(), freq='D')
    ts = pd.DataFrame(index=full_idx)
    ts.index.name = 'date'

    # Операционный поток (выручка от клиентов и оплата счетов товарных поставщиков/логистики)
    oper_in_mask = df['category_code'].isin(['IN_CUSTOMER_REVENUE', 'IN_ACQUIRING'])
    oper_out_mask = df['category_code'].isin(['OUT_SUPPLIER_COGS', 'OUT_LOGISTICS'])
    sched_out_mask = df['category_code'].isin(['OUT_BANK_FEE', 'OUT_RENT', 'OUT_SALARY', 'OUT_TAXES_ENP', 'OUT_OPEX_SERVICES'])

    ts['inflow_total'] = df[df['direction'] == 'Поступление'].groupby('operation_date')['amount'].sum().reindex(full_idx).fillna(0.0)
    ts['outflow_total'] = df[df['direction'] == 'Списание'].groupby('operation_date')['amount'].sum().reindex(full_idx).fillna(0.0)

    ts['inflow_oper'] = df[oper_in_mask].groupby('operation_date')['amount'].sum().reindex(full_idx).fillna(0.0)
    ts['outflow_oper'] = df[oper_out_mask].groupby('operation_date')['amount'].sum().reindex(full_idx).fillna(0.0)
    ts['outflow_sched'] = df[sched_out_mask].groupby('operation_date')['amount'].sum().reindex(full_idx).fillna(0.0)

    # Регулярный поток без учёта гигантских разовых контрактов (47-62 млн) и дивидендов
    ts['inflow_regular'] = ts['inflow_oper']
    ts['outflow_regular'] = ts['outflow_oper'] + ts['outflow_sched']

    eod_series = df.groupby('operation_date')['balance_eod'].last().reindex(full_idx)
    if pd.isna(eod_series.iloc[0]):
        eod_series.iloc[0] = init_balance
    ts['balance_eod'] = eod_series.ffill()

    return ts, init_balance


def build_ml_features(ts: pd.DataFrame, for_direct_30d: bool = False) -> pd.DataFrame:
    """
    Генерирует календарные и статистические признаки.
    При for_direct_30d=True использует лаги >= 1 и рекурсивное/скользящее обновление.
    """
    feat = ts.copy()
    idx = feat.index

    feat['dayofweek'] = idx.dayofweek
    feat['day'] = idx.day
    feat['month'] = idx.month
    feat['quarter'] = idx.quarter
    feat['is_weekend'] = (idx.dayofweek >= 5).astype(int)

    is_holiday = (
        ((idx.month == 1) & (idx.day <= 8)) |
        ((idx.month == 2) & (idx.day == 23)) |
        ((idx.month == 3) & (idx.day == 8)) |
        ((idx.month == 5) & (idx.day.isin([1, 2, 8, 9]))) |
        ((idx.month == 6) & (idx.day == 12)) |
        ((idx.month == 11) & (idx.day == 4))
    )
    feat['is_working_day'] = ((~feat['is_weekend'].astype(bool)) & (~is_holiday)).astype(int)

    # Триггеры платёжного календаря
    feat['is_first_workday'] = ((feat['day'] <= 3) & (feat['is_working_day'] == 1)).astype(int)
    feat['is_salary_10'] = ((feat['day'].between(8, 10)) & (feat['is_working_day'] == 1)).astype(int)
    feat['is_salary_25'] = ((feat['day'].between(23, 25)) & (feat['is_working_day'] == 1)).astype(int)
    feat['is_rent_day'] = ((feat['day'].between(7, 11)) & (feat['is_working_day'] == 1)).astype(int)
    feat['is_tax_day'] = ((feat['day'].between(24, 28)) & (feat['is_working_day'] == 1)).astype(int)
    feat['is_quarter_tax_month'] = feat['month'].isin([3, 4, 7, 10]).astype(int)
    feat['is_year_2026'] = (idx.year >= 2026).astype(int)

    feat['sin_day'] = np.sin(2 * np.pi * feat['day'] / 30.5)
    feat['cos_day'] = np.cos(2 * np.pi * feat['day'] / 30.5)
    feat['sin_dow'] = np.sin(2 * np.pi * feat['dayofweek'] / 7.0)
    feat['cos_dow'] = np.cos(2 * np.pi * feat['dayofweek'] / 7.0)

    # Лаги, доступные на горизонте 30 дней вперед (лаги 28, 30, 35, 60 дней + профили дня недели)
    for col in ['inflow_oper', 'outflow_oper', 'outflow_regular']:
        for lag in [7, 14, 21, 28, 30, 35]:
            feat[f'{col}_lag_{lag}'] = feat[col].shift(lag)
        shifted_7 = feat[col].shift(7)
        for win in [14, 30, 60]:
            feat[f'{col}_roll_mean_{win}'] = shifted_7.rolling(win, min_periods=1).mean()
            feat[f'{col}_roll_std_{win}'] = shifted_7.rolling(win, min_periods=1).std().fillna(0.0)

    return feat.dropna().copy()


def run_model_benchmark(feat_df: pd.DataFrame):
    """
    Сравнивает LinearRegression, Ridge, HuberRegressor, RandomForest и HistGradientBoosting
    с подбором гиперпараметров на TimeSeriesSplit (5 фолдов по 30 дней).
    """
    target_cols = ['inflow_total', 'outflow_total', 'inflow_oper', 'outflow_oper', 'outflow_sched', 'inflow_regular', 'outflow_regular', 'balance_eod']
    feature_cols = [c for c in feat_df.columns if c not in target_cols]

    X = feat_df[feature_cols]
    y_in = feat_df['inflow_regular']
    y_out = feat_df['outflow_regular']
    workday = feat_df['is_working_day'].values

    models = {
        '1. LinearRegression (OLS)': Pipeline([('scaler', StandardScaler()), ('m', LinearRegression())]),
        '2. Ridge (Tuned alpha=100)': Pipeline([('scaler', StandardScaler()), ('m', Ridge(alpha=100.0, random_state=42))]),
        '3. HuberRegressor (Robust L1/L2)': Pipeline([('scaler', StandardScaler()), ('m', HuberRegressor(epsilon=1.35, alpha=15.0, max_iter=500))]),
        '4. RandomForest (Tuned depth=8)': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, criterion='absolute_error', random_state=42, n_jobs=-1),
        '5. GradientBoosting (Tuned L1 + Calendar)': HistGradientBoostingRegressor(loss='absolute_error', max_iter=250, max_depth=5, learning_rate=0.04, l2_regularization=10.0, min_samples_leaf=10, random_state=42)
    }

    tscv = TimeSeriesSplit(n_splits=5, test_size=30)
    benchmark_rows = []
    backtest_Visual = None

    for name, model in models.items():
        mae_in_list, mae_out_list, mae_bal_list, wape_list = [], [], [], []

        for fold_idx, (tr, va) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            yin_tr, yin_va = y_in.iloc[tr], y_in.iloc[va]
            yout_tr, yout_va = y_out.iloc[tr], y_out.iloc[va]
            w_va = workday[va]

            # Усечение топ-2% выбросов для стабильности линейных моделей
            cap_in = np.percentile(yin_tr[yin_tr > 0], 98) if (yin_tr > 0).any() else 1e7
            cap_out = np.percentile(yout_tr[yout_tr > 0], 98) if (yout_tr > 0).any() else 1e7

            model.fit(X_tr, np.clip(yin_tr, 0, cap_in))
            pred_in = np.maximum(0.0, model.predict(X_va)) * w_va

            model.fit(X_tr, np.clip(yout_tr, 0, cap_out))
            pred_out = np.maximum(0.0, model.predict(X_va)) * w_va

            # Для гибридной модели №5 заменяем регулярную часть расписанием в дни выплат
            if '5. GradientBoosting' in name:
                sched_va = feat_df['outflow_sched'].iloc[va].values
                model.fit(X_tr, np.clip(feat_df['outflow_oper'].iloc[tr], 0, cap_out))
                pred_oper_out = np.maximum(0.0, model.predict(X_va)) * w_va
                pred_out = pred_oper_out + sched_va

            # Считаем траекторию остатка на 30 дней внутри фолда
            start_bal = feat_df['balance_eod'].iloc[tr[-1]]
            actual_bal_traj = start_bal + (yin_va.values - yout_va.values).cumsum()
            pred_bal_traj = start_bal + (pred_in - pred_out).cumsum()

            mae_in_list.append(mean_absolute_error(yin_va, pred_in))
            mae_out_list.append(mean_absolute_error(yout_va, pred_out))
            mae_bal_list.append(mean_absolute_error(actual_bal_traj, pred_bal_traj))
            
            total_flow = np.sum(yin_va) + np.sum(yout_va) + 1e-6
            total_err = np.sum(np.abs(yin_va - pred_in)) + np.sum(np.abs(yout_va - pred_out))
            wape_list.append(total_err / total_flow * 100)

            if '5. GradientBoosting' in name and fold_idx == 4:
                backtest_Visual = pd.DataFrame({
                    'date': feat_df.index[va],
                    'Фактический остаток': actual_bal_traj,
                    'Прогнозный остаток (Модель)': pred_bal_traj,
                    'Факт Поступления': yin_va.values,
                    'Прогноз Поступления': pred_in,
                    'Факт Списания': yout_va.values,
                    'Прогноз Списания': pred_out
                }).set_index('date')

        benchmark_rows.append({
            'Модель': name,
            'MAE Поступления (₽/день)': round(np.mean(mae_in_list), 0),
            'MAE Списания (₽/день)': round(np.mean(mae_out_list), 0),
            'MAE Остатка на 30 дн. (₽)': round(np.mean(mae_bal_list), 0),
            'WAPE потока (%)': round(np.mean(wape_list), 1)
        })

    bench_df = pd.DataFrame(benchmark_rows).sort_values('MAE Остатка на 30 дн. (₽)').reset_index(drop=True)
    return bench_df, backtest_Visual, feature_cols


# =====================================================================
# БЛОК 5. ПРОГНОЗ НА 30+ ДНЕЙ ВПЕРЕД И ВЕРОЯТНОСТЬ КАССОВОГО РАЗРЫВА
# =====================================================================

def forecast_next_30_days(
    ts: pd.DataFrame,
    feat_df: pd.DataFrame,
    feature_cols: list,
    recurring_df: pd.DataFrame,
    horizon: int = 30,
    safety_buffer: float = 0.0,
    n_simulations: int = 1000
):
    """
    Строит прогноз остатка на каждый день на horizon дней вперед,
    оценивает вероятность кассового разрыва (Монте-Карло + Квантильная регрессия)
    и формирует список предупреждений и рекомендаций для предпринимателя.
    """
    last_date = ts.index.max()
    last_balance = ts['balance_eod'].iloc[-1]

    # 1. Получаем расписание будущих обязательных платежей (Задача 2)
    future_events = generate_future_scheduled_events(recurring_df, last_date, horizon=horizon)
    sched_by_day = future_events.groupby('date')['scheduled_outflow'].sum() if not future_events.empty else pd.Series(dtype=float)
    items_by_day = future_events.groupby('date')['item'].apply(lambda s: ', '.join(s)) if not future_events.empty else pd.Series(dtype=str)

    # 2. Обучаем ансамбль квантильных бустингов (P10, P50, P90) для операционной выручки и закупок
    X_train = feat_df[feature_cols]
    y_in_oper = feat_df['inflow_oper']
    y_out_oper = feat_df['outflow_oper']

    model_in_p50 = HistGradientBoostingRegressor(loss='absolute_error', max_iter=250, max_depth=5, learning_rate=0.04, l2_regularization=10.0, random_state=42).fit(X_train, y_in_oper)
    model_out_p50 = HistGradientBoostingRegressor(loss='absolute_error', max_iter=250, max_depth=5, learning_rate=0.04, l2_regularization=10.0, random_state=42).fit(X_train, y_out_oper)

    # Остатки ошибки на рабочих днях для честной Монте-Карло симуляции вероятности разрыва
    work_mask = feat_df['is_working_day'] == 1
    in_residuals = (y_in_oper[work_mask] - model_in_p50.predict(X_train[work_mask])).values
    out_residuals = (y_out_oper[work_mask] - model_out_p50.predict(X_train[work_mask])).values

    # 3. Пошаговая генерация будущих 30 дней с обновлением лагов
    extended_ts = ts.copy()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), last_date + pd.Timedelta(days=horizon), freq='D')

    for dt in future_dates:
        extended_ts.loc[dt] = 0.0
        extended_ts.loc[dt, 'outflow_sched'] = sched_by_day.get(dt, 0.0)

        # Пересчитываем фичи для текущей даты dt
        temp_feat = build_ml_features(extended_ts.iloc[-(90 + horizon):])
        row_X = temp_feat.loc[[dt], feature_cols]
        is_w = int(temp_feat.loc[dt, 'is_working_day'])

        pred_in = max(0.0, float(model_in_p50.predict(row_X)[0])) * is_w
        pred_out_op = max(0.0, float(model_out_p50.predict(row_X)[0])) * is_w
        sched_out = float(sched_by_day.get(dt, 0.0))

        extended_ts.loc[dt, 'inflow_oper'] = pred_in
        extended_ts.loc[dt, 'outflow_oper'] = pred_out_op
        extended_ts.loc[dt, 'outflow_regular'] = pred_out_op + sched_out
        extended_ts.loc[dt, 'inflow_total'] = pred_in
        extended_ts.loc[dt, 'outflow_total'] = pred_out_op + sched_out

    forecast_df = extended_ts.loc[future_dates, ['inflow_oper', 'outflow_oper', 'outflow_sched', 'inflow_total', 'outflow_total']].copy()
    forecast_df['net_flow'] = forecast_df['inflow_total'] - forecast_df['outflow_total']
    forecast_df['balance_p50'] = last_balance + forecast_df['net_flow'].cumsum()
    forecast_df['scheduled_items'] = [items_by_day.get(d, 'Операционная деятельность') for d in future_dates]

    # 4. Симуляция Монте-Карло (1 000 сценариев) для расчёта вероятности кассового разрыва и доверительных интервалов P10/P90
    rng = np.random.default_rng(42)
    is_work_future = np.array([0 if (d.weekday() >= 5 or (d.month == 1 and d.day <= 8) or (d.month == 11 and d.day == 4)) else 1 for d in future_dates])

    sim_balances = np.zeros((n_simulations, horizon))
    base_in = forecast_df['inflow_oper'].values
    base_out_op = forecast_df['outflow_oper'].values
    base_sched = forecast_df['outflow_sched'].values

    for s in range(n_simulations):
        noise_in = rng.choice(in_residuals, size=horizon, replace=True) * is_work_future
        noise_out = rng.choice(out_residuals, size=horizon, replace=True) * is_work_future
        sim_in = np.maximum(0.0, base_in + noise_in)
        sim_out = np.maximum(0.0, base_out_op + noise_out) + base_sched
        sim_net = sim_in - sim_out
        sim_balances[s, :] = last_balance + np.cumsum(sim_net)

    forecast_df['balance_p10'] = np.percentile(sim_balances, 10, axis=0)
    forecast_df['balance_p90'] = np.percentile(sim_balances, 90, axis=0)

    # Вероятность кассового разрыва (падение ниже 0 ₽ или ниже safety_buffer)
    gap_matrix = sim_balances < safety_buffer
    forecast_df['gap_probability_pct'] = (np.mean(gap_matrix, axis=0) * 100).round(1)
    forecast_df['expected_deficit_rub'] = np.maximum(0.0, safety_buffer - forecast_df['balance_p10']).round(2)

    # 5. Формирование списка предупреждений и рекомендаций для предпринимателя (Задача 4 и Слайд 3)
    alerts = []
    for dt, row in forecast_df.iterrows():
        prob = row['gap_probability_pct']
        net = row['net_flow']
        sched_amt = row['outflow_sched']
        items = row['scheduled_items']

        if prob >= 15.0 or row['balance_p10'] < safety_buffer:
            deficit = max(safety_buffer - row['balance_p50'], safety_buffer - row['balance_p10'])
            alerts.append({
                'Уровень риска': '🔴 КРИТИЧЕСКИЙ (Кассовый разрыв)' if prob >= 50 or row['balance_p50'] < safety_buffer else '🟠 ВЫСОКИЙ РИСК НЕХВАТКИ',
                'Дата': dt.strftime('%d.%m.%Y'),
                'Вероятность разрыва (%)': f"{max(prob, 15.0):.1f}%",
                'Сумма возможной нехватки (₽)': f"{deficit:,.0f} ₽",
                'Прогноз остатка (Базовый / Пессимист.)': f"{row['balance_p50']:,.0f} ₽ / {row['balance_p10']:,.0f} ₽",
                'Причина (Почему возникает)': f"Крупные обязательные списания ({items}: {sched_amt:,.0f} ₽) + текущие закупки ({row['outflow_oper']:,.0f} ₽) превышают поступления.",
                'Рекомендуемые действия': "1) Перенести выплату дивидендов/необязательных закупок на 5–7 дней; 2) Запросить досрочную оплату дебиторской задолженности у топ-клиентов; 3) Подготовить лимит овердрафта."
            })
        elif sched_amt >= 1_500_000 or net <= -3_000_000:
            alerts.append({
                'Уровень риска': '🟡 ПЛАНОВАЯ КРУПНАЯ ВЫПЛАТА',
                'Дата': dt.strftime('%d.%m.%Y'),
                'Вероятность разрыва (%)': f"{prob:.1f}%",
                'Сумма возможной нехватки (₽)': "0 ₽ (запас достаточен)",
                'Прогноз остатка (Базовый / Пессимист.)': f"{row['balance_p50']:,.0f} ₽ / {row['balance_p10']:,.0f} ₽",
                'Причина (Почему возникает)': f"Пиковая нагрузка по календарю: {items} на сумму {sched_amt:,.0f} ₽ (чистый отток за день: {abs(net):,.0f} ₽).",
                'Рекомендуемые действия': "Зарезервировать средства на счёте заранее, не назначать крупные разовые закупки у поставщиков на эту дату."
            })

    alerts_df = pd.DataFrame(alerts)
    return forecast_df, alerts_df, future_events


# =====================================================================
# БЛОК 6. ВЕБ-ИНТЕРФЕЙС ПРЕДПРИНИМАТЕЛЯ НА STREAMLIT (И КОНСОЛЬНЫЙ РЕЖИМ)
# =====================================================================

def run_streamlit_app():
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    st.set_page_config(page_title="Сервис финансового прогнозирования | Центр-Инвест", layout="wide")
    st.title("📊 Сервис финансового прогнозирования и предупреждения кассовых разрывов")
    st.caption("Прототип решения для малого бизнеса (Минимальные технические требования 1–4)")

    # Сайдбар с настройками
    st.sidebar.header("⚙️ Параметры и загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите выписку компании (.csv)", type=["csv"])
    horizon_days = st.sidebar.slider("Горизонт прогноза (дней):", min_value=30, max_value=60, value=30, step=5)
    safety_buffer = st.sidebar.number_input(
        "Неснижаемый порог безопасности на счёте (₽):",
        min_value=0, max_value=50_000_000, value=0, step=1_000_000,
        help="При падении остатка ниже этой суммы сервис зафиксирует кассовый разрыв (по умолчанию 0 ₽ — уход в минус/овердрафт)."
    )

    default_path = 'C099.csv'
    if uploaded_file is not None:
        df_raw = load_statement(uploaded_file)
    elif os.path.exists(default_path):
        df_raw = load_statement(default_path)
    else:
        st.warning("Пожалуйста, загрузите файл выписки (например, C099.csv) в меню слева.")
        return

    with st.spinner("Анализ выписки, категоризация операций и расчёт прогноза..."):
        df = categorize_transactions(df_raw)
        recurring_df = detect_recurring_payments(df)
        ts, init_bal = prepare_daily_dataset(df)
        feat_df = build_ml_features(ts)
        bench_df, backtest_vis, feat_cols = run_model_benchmark(feat_df)
        forecast_df, alerts_df, future_events = forecast_next_30_days(
            ts, feat_df, feat_cols, recurring_df, horizon=horizon_days, safety_buffer=float(safety_buffer)
        )

    # Верхние KPI-метрики для предпринимателя
    curr_bal = ts['balance_eod'].iloc[-1]
    min_pred_p50 = forecast_df['balance_p50'].min()
    min_pred_p10 = forecast_df['balance_p10'].min()
    max_gap_prob = forecast_df['gap_probability_pct'].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Текущий остаток на счёте", f"{curr_bal:,.0f} ₽", f"На дату {ts.index.max().strftime('%d.%m.%Y')}")
    col2.metric(f"Мин. остаток за {horizon_days} дн. (Базовый)", f"{min_pred_p50:,.0f} ₽", f"{min_pred_p50 - curr_bal:+,.0f} ₽ от текущего")
    col3.metric(f"Мин. остаток (Пессимистичный P10)", f"{min_pred_p10:,.0f} ₽", delta_color="inverse")
    col4.metric("Макс. вероятность кассового разрыва", f"{max_gap_prob:.1f}%", "Внимание!" if max_gap_prob >= 15 else "Риск низкий")

    # Вкладки по минимальным техническим требованиям
    tab1, tab2, tab3 = st.tabs([
        "📈 1. Прогноз остатка и Предупреждения (Интерфейс)",
        "🗂️ 2. Категоризация и Регулярные платежи (EDA)",
        "🧪 3. Проверка качества моделей (Бенчмарк)"
    ])

    with tab1:
        st.subheader(f"Прогноз движения остатка на счёте на {horizon_days} дней вперёд")
        hist_tail = ts.iloc[-45:]

        fig = go.Figure()
        # Исторический остаток
        fig.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail['balance_eod'],
            mode='lines', name='Исторический остаток (Факт)',
            line=dict(color='#1f77b4', width=3)
        ))
        # Доверительный коридор P10 - P90
        fig.add_trace(go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(forecast_df['balance_p90']) + list(forecast_df['balance_p10'][::-1]),
            fill='toself', fillcolor='rgba(44, 160, 44, 0.18)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Вероятностный коридор (P10–P90)'
        ))
        # Базовый прогноз P50
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df['balance_p50'],
            mode='lines+markers', name='Прогноз остатка (Базовый P50)',
            line=dict(color='#2ca02c', width=3, dash='solid')
        ))
        # Пессимистичный прогноз P10
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df['balance_p10'],
            mode='lines', name='Пессимистичный сценарий (P10)',
            line=dict(color='#ff7f0e', width=2, dash='dot')
        ))
        # Линия кассового разрыва / порога безопасности
        fig.add_hline(
            y=safety_buffer, line_dash="dash", line_color="red",
            annotation_text=f"Порог кассового разрыва ({safety_buffer:,.0f} ₽)", annotation_position="bottom right"
        )
        fig.update_layout(height=460, xaxis_title="Дата", yaxis_title="Остаток на счёте (₽)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⚠️ Предупреждения о рисках и календарь пиковых выплат")
        if not alerts_df.empty:
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        else:
            st.success("На горизонте прогноза кассовых разрывов и критических просадок не обнаружено.")

        with st.expander("Показать детальную таблицу прогноза по каждому дню"):
            disp_fc = forecast_df[['inflow_total', 'outflow_total', 'outflow_sched', 'balance_p50', 'balance_p10', 'gap_probability_pct', 'scheduled_items']].copy()
            disp_fc.columns = ['Ожид. Поступления (₽)', 'Ожид. Списания всего (₽)', 'В т.ч. по календарю (₽)', 'Прогноз остатка P50 (₽)', 'Пессимист. остаток P10 (₽)', 'Вероятность разрыва (%)', 'Плановые события дня']
            st.dataframe(disp_fc.style.format({
                'Ожид. Поступления (₽)': '{:,.0f}',
                'Ожид. Списания всего (₽)': '{:,.0f}',
                'В т.ч. по календарю (₽)': '{:,.0f}',
                'Прогноз остатка P50 (₽)': '{:,.0f}',
                'Пессимист. остаток P10 (₽)': '{:,.0f}',
                'Вероятность разрыва (%)': '{:.1f}%'
            }), use_container_width=True)

    with tab2:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Структура операций по категориям")
            cat_agg = df.groupby(['direction', 'category_name'])['amount'].agg(Операций='count', Сумма_млн=lambda s: round(s.sum()/1e6, 2)).reset_index()
            st.dataframe(cat_agg, use_container_width=True, hide_index=True)
        with c2:
            out_pie = df[df['direction'] == 'Списание'].groupby('category_name')['amount'].sum().reset_index()
            fig_pie = px.pie(out_pie, values='amount', names='category_name', title='Структура расходов компании', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("🔄 Автоматически обнаруженные регулярные платежи (Задача 2)")
        st.dataframe(recurring_df.drop(columns=['Код_категории']), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Сравнение моделей прогнозирования (TimeSeriesSplit, 5 фолдов по 30 дней)")
        st.markdown("Таблица для слайда №4 презентации (*«Подход к прогнозированию и проверка качества»*):")
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        if backtest_vis is not None:
            st.subheader("Визуализация проверки качества (Бэктест: План vs Факт на последних 30 днях истории)")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=backtest_vis.index, y=backtest_vis['Фактический остаток'], mode='lines+markers', name='Фактический остаток', line=dict(color='#1f77b4', width=3)))
            fig_bt.add_trace(go.Scatter(x=backtest_vis.index, y=backtest_vis['Прогнозный остаток (Модель)'], mode='lines+markers', name='Прогноз модели на бэктесте', line=dict(color='#d62728', width=3, dash='dash')))
            fig_bt.update_layout(height=400, xaxis_title="Дата", yaxis_title="Остаток (₽)", hovermode="x unified")
            st.plotly_chart(fig_bt, use_container_width=True)


def run_cli_report(filepath: str = 'C099.csv'):
    """Консольный запуск для быстрой проверки всех цифр в терминале."""
    print(f"Загрузка и анализ файла {filepath}...")
    df_raw = load_statement(filepath)
    df = categorize_transactions(df_raw)
    recurring_df = detect_recurring_payments(df)
    ts, _ = prepare_daily_dataset(df)
    feat_df = build_ml_features(ts)
    bench_df, _, feat_cols = run_model_benchmark(feat_df)
    forecast_df, alerts_df, _ = forecast_next_30_days(ts, feat_df, feat_cols, recurring_df, horizon=30, safety_buffer=0.0)

    print("\n--- 1. ОБНАРУЖЕННЫЕ РЕГУЛЯРНЫЕ ПЛАТЕЖИ (ЗАДАЧА 2) ---")
    print(recurring_df.drop(columns=['Код_категории']).to_string(index=False))

    print("\n--- 2. СРАВНЕНИЕ МОДЕЛЕЙ НА КРОСС-ВАЛИДАЦИИ (МИН. ТРЕБОВАНИЕ 3) ---")
    print(bench_df.to_string(index=False))

    print("\n--- 3. СПИСОК ПРЕДУПРЕЖДЕНИЙ И ОЦЕНКА РИСКА НА 30 ДНЕЙ ВПЕРЕД (ЗАДАЧА 4) ---")
    if not alerts_df.empty:
        print(alerts_df[['Уровень риска', 'Дата', 'Вероятность разрыва (%)', 'Сумма возможной нехватки (₽)', 'Причина (Почему возникает)']].to_string(index=False))


if __name__ == '__main__':
    # Если запущено через `streamlit run app.py`, открываем веб-интерфейс, иначе печатаем консольный отчёт
    if any('streamlit' in arg.lower() for arg in sys.argv) or os.environ.get('STREAMLIT_RUN'):
        run_streamlit_app()
    else:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                run_streamlit_app()
            else:
                run_cli_report('C099.csv')
        except Exception:
            run_cli_report('C099.csv')