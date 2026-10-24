/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  Calendar,
  BarChart3,
  Sliders,
  RefreshCw,
  ArrowRight,
  ShieldAlert,
  Zap,
  Info,
  Download,
  Upload,
  ArrowUpRight,
  ArrowDownRight,
  Wallet,
  FileSpreadsheet,
  Check,
  X,
  Clock,
  Sparkles,
  PlusCircle,
  Trash2,
  HelpCircle,
  FileText,
  AlertOctagon
} from 'lucide-react';
import { summarizePaths, buildGapAlerts, addDaysIso } from '../shared/cashflow.ts';
import type { ForecastPoint, GapAlert, Insight } from '../shared/cashflow.ts';
import ZoomChart from './ZoomChart.tsx';
import type { ChartMarker } from './ZoomChart.tsx';

interface StatementData {
  company_id: string;
  industry: string;
  split: string;
  current_balance: number;
  min_forecast_p50: number;
  min_forecast_p10: number;
  max_gap_probability: number;
  forecast_risk_level?: 'CRITICAL' | 'WARNING' | 'STABLE';
  forecast_status_label?: string;
  historical_summary?: string;
  total_operations?: number;
  total_inflow?: number;
  total_outflow?: number;
  net_cash_flow?: number;
  categories?: Array<{
    name: string;
    total_amount: number;
    direction: string;
    count?: number;
  }>;
  counterparties?: Array<{
    name: string;
    total: number;
    direction: string;
    count?: number;
  }>;
  recurring_items: Array<{
    category_code?: string;
    category_name: string;
    counterparty: string;
    typical_day: number;
    typical_amount: number;
    count?: number;
  }>;
  history: Array<{ date: string; balance: number; inflow: number; outflow: number }>;
  forecast: ForecastPoint[];
  alerts: GapAlert[];
  insights?: Insight[];
  period_start?: string;
  period_end?: string;
  paths?: number[][];
  model?: { method: string; scenarios: number; gap_scenarios: number; horizon_days: number };
}

// Manual single-occurrence operation
export interface ManualOperation {
  id: string;
  type: 'inflow' | 'outflow'; // тип операции
  category: string; // категория
  amount: number; // сумма (₽)
  date: string; // конкретная дата YYYY-MM-DD
  comment?: string; // примечание
}

// Format error details modal
interface FormatErrorDetail {
  isOpen: boolean;
  fileName: string;
  detectedHeaders: string[];
  missingRequired: string[];
  rawPreview?: string;
}

const ruDate = (iso: string) => iso.slice(0, 10).split('-').reverse().join('.');
const ruDayMonth = (iso: string) => `${iso.slice(8, 10)}.${iso.slice(5, 7)}`;

// Rule-based remarks about cash habits, shown next to the forecast status (not part of it)
function RemarksBadge({ count, critical, onOpen }: { count: number; critical: boolean; onOpen: () => void }) {
  const word = count % 10 === 1 && count % 100 !== 11 ? 'замечание' : [2, 3, 4].includes(count % 10) && ![12, 13, 14].includes(count % 100) ? 'замечания' : 'замечаний';
  return (
    <button
      type="button"
      onClick={onOpen}
      title="Правила по истории выписки: выводы владельца, подушка ликвидности, прошлые просадки. На статус прогноза не влияют."
      className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-slate-800 hover:bg-slate-700 border ${
        critical ? 'text-rose-300 border-rose-500/50' : 'text-amber-300 border-amber-500/40'
      }`}
    >
      {count} {word} по денежному потоку
    </button>
  );
}

// "Period: dd.mm.yyyy – dd.mm.yyyy" caption shown under every chart title
function ChartPeriod({ from, to, note }: { from?: string; to?: string; note?: string }) {
  if (!from || !to) return null;
  return (
    <p className="text-[11px] text-slate-400 mt-1 flex items-center gap-1.5">
      <Calendar className="w-3.5 h-3.5 text-slate-500 shrink-0" />
      <span>
        Период: <span className="font-mono text-slate-300">{ruDate(from)} – {ruDate(to)}</span>
        {note ? ` · ${note}` : ''}
      </span>
    </p>
  );
}

export default function App() {
  // Navigation: 3 workflow columns
  const [activeTab, setActiveTab] = useState<'flows' | 'forecast' | 'whatif'>('flows');
  const openRemarks = () => {
    setActiveTab('forecast');
    setTimeout(() => document.getElementById('cash-remarks')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
  };

  // Core State: Starts completely empty waiting for entrepreneur's file or manual entry
  const [statementData, setStatementData] = useState<StatementData | null>(null);
  const [statementFileName, setStatementFileName] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadFeedback, setUploadFeedback] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Manual Operations State (Задачи со звездочкой 2, 4: ручной ввод разовых платежей)
  const [manualOperations, setManualOperations] = useState<ManualOperation[]>([]);
  const [isManualModalOpen, setIsManualModalOpen] = useState<boolean>(false);
  const [manualType, setManualType] = useState<'inflow' | 'outflow'>('inflow');
  const [manualCategory, setManualCategory] = useState<string>('Выручка от покупателей');
  const [manualAmount, setManualAmount] = useState<string>('');
  const [manualDate, setManualDate] = useState<string>('');
  const [manualComment, setManualComment] = useState<string>('');
  const [recalcTimeMs, setRecalcTimeMs] = useState<number>(1.2);

  // Format Validation Error Modal State
  const [formatError, setFormatError] = useState<FormatErrorDetail | null>(null);

  // What-If Scenario State (Задача со звездочкой 3)
  const [whatIfDelayDays, setWhatIfDelayDays] = useState<number>(0);
  const [whatIfDelaySharePct, setWhatIfDelaySharePct] = useState<number>(40);
  const [whatIfRevenueFactor, setWhatIfRevenueFactor] = useState<number>(100);
  const [whatIfPostponeSupplierDays, setWhatIfPostponeSupplierDays] = useState<number>(0);
  const [whatIfOneOffExpense, setWhatIfOneOffExpense] = useState<number>(0);
  const [whatIfOneOffDate, setWhatIfOneOffDate] = useState<number>(15);
  const [safetyBuffer, setSafetyBuffer] = useState<number>(0);

  // Set default manual date to tomorrow
  useEffect(() => {
    const d = new Date();
    d.setDate(d.getDate() + 5);
    setManualDate(d.toISOString().split('T')[0]);
  }, []);

  // Format helpers
  const formatExactRub = (val: number) => Math.round(val).toLocaleString('ru-RU') + ' ₽';
  const formatRub = (val: number) => {
    if (Math.abs(val) >= 1_000_000_000) return (val / 1_000_000_000).toFixed(2) + ' млрд ₽';
    if (Math.abs(val) >= 1_000_000) return (val / 1_000_000).toFixed(2) + ' млн ₽';
    if (Math.abs(val) >= 1_000) return (val / 1_000).toFixed(0) + ' тыс. ₽';
    return Math.round(val).toLocaleString('ru-RU') + ' ₽';
  };

  // Download C052 standard template
  const downloadC052Template = () => {
    const templateContent =
      "company_id;operation_id;operation_date;amount;direction;counterparty_inn;counterparty_name;payment_purpose;balance_eod\n" +
      "C052;OP-000001;01.11.2026;500.00;Списание;7702000000;ПАО Банк Центр-инвест;Комиссия за ведение расчетного счета;1249500.00\n" +
      "C052;OP-000002;03.11.2026;350000.00;Поступление;6164000001;ООО Партнер-Дистрибьютор;Оплата по счету № 142 от 25.10.2026 за товары/услуги;1599500.00\n" +
      "C052;OP-000003;10.11.2026;420000.00;Списание;7700000002;Сотрудники компании;Выплата заработной платы (окончательный расчет);1179500.00\n" +
      "C052;OP-000004;15.11.2026;180000.00;Списание;6165000003;АО Арендодатель;Оплата аренды офиса и склада по договору № 55;999500.00\n" +
      "C052;OP-000005;28.11.2026;310000.00;Списание;7707000004;УФК по г. Ростову-на-Дону (ФНС ЕНП);Единый налоговый платеж (НДФЛ и страховые взносы);689500.00\n";

    const blob = new Blob([templateContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'template_c052.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Quick 1-click test with real C052 demo statement
  const loadDemoC052 = async () => {
    setIsUploading(true);
    setUploadFeedback('Загрузка эталонной выписки компании C052 (IT-услуги)...');
    try {
      const res = await fetch('/api/demo-c052.csv');
      if (!res.ok) throw new Error('Не удалось загрузить файл demo-c052.csv с сервера');
      const text = await res.text();
      await processCsvContent(text, 'demo_statement_c052.csv');
    } catch (err: any) {
      setUploadFeedback(`Ошибка загрузки демо: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // CSV Statement Parser & Format Validator
  const processCsvContent = async (text: string, fileName: string) => {
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0 && !l.trim().startsWith('#'));
    if (lines.length < 2) {
      setFormatError({
        isOpen: true,
        fileName,
        detectedHeaders: [],
        missingRequired: ['operation_date', 'amount', 'direction', 'balance_eod'],
        rawPreview: lines.slice(0, 3).join('\n')
      });
      return;
    }

    // Determine separator (; , or \t)
    const headerLine = lines[0];
    const sep = headerLine.includes(';') ? ';' : (headerLine.includes('\t') ? '\t' : ',');
    const headers = headerLine.split(sep).map(h => h.trim().toLowerCase().replace(/["']/g, ''));

    // Required column mappings
    const dateIdx = headers.findIndex(h => h.includes('date') || h.includes('дата') || h.includes('день'));
    const amountIdx = headers.findIndex(h => h.includes('amount') || h.includes('сумма') || h.includes('rub') || h.includes('руб'));
    const directionIdx = headers.findIndex(h => h.includes('direction') || h.includes('направление') || h.includes('тип') || h.includes('type') || h.includes('credit') || h.includes('debit'));
    const partyIdx = headers.findIndex(h => h.includes('counterparty') || h.includes('контрагент') || h.includes('партнер') || h.includes('получатель') || h.includes('плательщик'));
    const purposeIdx = headers.findIndex(h => h.includes('purpose') || h.includes('назначение') || h.includes('категория') || h.includes('описание'));
    const balanceIdx = headers.findIndex(h => h.includes('balance') || h.includes('остаток') || h.includes('сальдо'));

    // Validation check: mandatory fields
    const missing: string[] = [];
    if (dateIdx < 0) missing.push('operation_date (дата операции)');
    if (amountIdx < 0) missing.push('amount (сумма)');
    if (directionIdx < 0) missing.push('direction (направление: Поступление/Списание)');

    if (missing.length > 0) {
      setFormatError({
        isOpen: true,
        fileName,
        detectedHeaders: headers,
        missingRequired: missing,
        rawPreview: lines.slice(0, 4).join('\n')
      });
      return;
    }

    // Parse valid rows
    const operations: Array<any> = [];
    // balance_eod of the latest operation date = closing balance of the statement
    let closingBalance: number | undefined;
    let closingBalanceDate = '';
    let skippedRows = 0;

    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(sep).map(c => c.trim().replace(/^["']|["']$/g, ''));
      if (cols.length < 2) continue;

      const dRaw = (cols[dateIdx] || '').trim();
      // Support DD.MM.YYYY and YYYY-MM-DD; rows with an unparseable date are skipped
      let dIso = '';
      const isoMatch = dRaw.match(/^(\d{4})-(\d{2})-(\d{2})/);
      const ruMatch = dRaw.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})$/);
      if (isoMatch) {
        dIso = `${isoMatch[1]}-${isoMatch[2]}-${isoMatch[3]}`;
      } else if (ruMatch) {
        dIso = `${ruMatch[3]}-${ruMatch[2].padStart(2, '0')}-${ruMatch[1].padStart(2, '0')}`;
      }
      const dCheck = new Date(dIso + 'T00:00:00Z');
      if (!dIso || isNaN(dCheck.getTime()) || dCheck.toISOString().slice(0, 10) !== dIso) {
        skippedRows++;
        continue;
      }

      const amtRaw = (cols[amountIdx] || '').replace(/\s/g, '').replace(',', '.');
      const amtVal = parseFloat(amtRaw);
      if (!amtRaw || !Number.isFinite(amtVal)) {
        skippedRows++;
        continue;
      }
      const dirVal = (cols[directionIdx] || '').toLowerCase();
      let isCredit = true;
      if (dirVal.includes('списан') || dirVal.includes('расход') || dirVal.includes('debit') || dirVal.includes('дебет') || amtVal < 0) {
        isCredit = false;
      } else if (dirVal.includes('поступлен') || dirVal.includes('credit') || dirVal.includes('кредит') || dirVal.includes('приход')) {
        isCredit = true;
      } else {
        isCredit = amtVal >= 0;
      }
      const amt = Math.abs(amtVal);

      const party = partyIdx >= 0 && cols[partyIdx] ? cols[partyIdx] : (isCredit ? 'Покупатель продукции' : 'Поставщик услуг');
      const purp = purposeIdx >= 0 && cols[purposeIdx] ? cols[purposeIdx] : (isCredit ? 'Выручка от реализации' : 'Оплата счетов');

      if (balanceIdx >= 0 && cols[balanceIdx]) {
        const b = parseFloat(cols[balanceIdx].replace(/\s/g, '').replace(',', '.'));
        if (!isNaN(b) && dIso >= closingBalanceDate) {
          closingBalance = b;
          closingBalanceDate = dIso;
        }
      }

      let category = isCredit ? 'Выручка от покупателей' : 'Оплата счетов поставщикам';
      const pLow = purp.toLowerCase();
      if (pLow.includes('зарплат') || pLow.includes('аванс') || pLow.includes('фот') || pLow.includes('сотрудник')) {
        category = pLow.includes('аванс') ? 'Зарплата: Аванс (20-25 число)' : 'Зарплата: Окончательный расчет (8-10 число)';
      } else if (pLow.includes('налог') || pLow.includes('енп') || pLow.includes('фнс') || pLow.includes('страхов')) {
        category = 'Налоги и взносы (ЕНП 28-го числа)';
      } else if (pLow.includes('аренд')) {
        category = 'Аренда помещений и оборудования';
      } else if (pLow.includes('эквайринг') || pLow.includes('сбп') || pLow.includes('карта')) {
        category = 'Торговый эквайринг и СБП';
      } else if (pLow.includes('комисси') || pLow.includes('рко') || pLow.includes('банк')) {
        category = 'Комиссии банка и РКО';
      }

      operations.push({
        date: dIso,
        amount: amt,
        type: isCredit ? 'credit' : 'debit',
        direction: isCredit ? 'Поступление' : 'Списание',
        counterparty: party,
        category,
        purpose: purp
      });
    }

    if (operations.length === 0) {
      setFormatError({
        isOpen: true,
        fileName,
        detectedHeaders: headers,
        missingRequired: ['Строки с данными не содержат распознаваемых значений'],
        rawPreview: lines.slice(0, 4).join('\n')
      });
      return;
    }

    // Call server analysis model
    const compName = fileName.replace(/\.[^/.]+$/, '').toUpperCase();
    const res = await fetch('/api/analyze-statement', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        companyName: compName,
        operations,
        // Without a balance column the opening balance is taken as 0
        closingBalance
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Ошибка анализа на стороне модели');
    }

    const analyzed = await res.json();
    setStatementData(analyzed);
    setStatementFileName(fileName);
    const notes: string[] = [];
    if (skippedRows > 0) notes.push(`пропущено строк с некорректной датой или суммой: ${skippedRows}`);
    if (closingBalance === undefined) notes.push('в выписке нет остатка на конец дня, входящий остаток принят за 0 ₽');
    setUploadFeedback(
      `Успешно обработано: ${operations.length} операций компании «${compName}». Базовый прогноз построен.` +
      (notes.length > 0 ? ` Внимание: ${notes.join('; ')}.` : '')
    );
  };

  // Upload handler from file input
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    setUploadFeedback(`Чтение файла «${file.name}»...`);
    try {
      const text = await file.text();
      await processCsvContent(text, file.name);
    } catch (err: any) {
      setUploadFeedback(`Ошибка обработки: ${err.message || 'Не удалось прочитать файл'}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Add Manual Operation Handler
  const handleAddManualOperation = (e: React.FormEvent) => {
    e.preventDefault();
    const amt = parseFloat(manualAmount.replace(/\s/g, '').replace(',', '.'));
    if (isNaN(amt) || amt <= 0) {
      alert('Пожалуйста, укажите корректную сумму операции больше нуля');
      return;
    }
    if (!manualDate) {
      alert('Пожалуйста, выберите дату операции');
      return;
    }

    const newOp: ManualOperation = {
      id: 'MAN-' + Date.now().toString().slice(-6),
      type: manualType,
      category: manualCategory,
      amount: amt,
      date: manualDate,
      comment: manualComment.trim() || undefined
    };

    setManualOperations(prev => [...prev, newOp]);
    setManualAmount('');
    setManualComment('');
    setIsManualModalOpen(false);
  };

  const removeManualOperation = (id: string) => {
    setManualOperations(prev => prev.filter(op => op.id !== id));
  };

  const clearAllData = () => {
    setStatementData(null);
    setStatementFileName(null);
    setManualOperations([]);
    setUploadFeedback(null);
  };

  // =========================================================================
  // INCREMENTAL DIFFERENTIAL UPDATE ENGINE (Задачи со звездочкой 2, 3, 4)
  // The base forecast (analog-period paths) is computed once on the server. Discrete manual operations
  // and what-if perturbations are applied to those paths via O(30) discrete convolution and prefix sums
  // in < 2ms without another request to the server.
  // =========================================================================
  const { effectiveForecast, effectiveAlerts, effectiveCurrentBalance, summaryStats } = useMemo(() => {
    const t0 = performance.now();

    // If no statement data and no manual ops, return empty
    if (!statementData && manualOperations.length === 0) {
      return {
        effectiveForecast: [] as ForecastPoint[],
        effectiveAlerts: [] as GapAlert[],
        effectiveCurrentBalance: 0,
        summaryStats: {
          minP50: 0,
          minP10: 0,
          maxGapProb: 0,
          totalInflow: 0,
          totalOutflow: 0,
          netFlow: 0,
          riskLevel: 'STABLE' as 'CRITICAL' | 'WARNING' | 'STABLE',
          statusLabel: 'Нет данных',
          remarks: 0,
          criticalRemarks: false
        }
      };
    }

    // Baseline values: from statementData or pure manual initial state
    const baseBalance = statementData ? statementData.current_balance : 500000;
    const basePts: ForecastPoint[] = statementData
      ? statementData.forecast
      : Array.from({ length: 30 }).map((_, i) => {
          const d = new Date();
          d.setDate(d.getDate() + i + 1);
          return {
            date: d.toISOString().split('T')[0],
            balance_p50: baseBalance,
            balance_p10: baseBalance,
            balance_p90: baseBalance,
            inflow: 0,
            outflow: 0,
            sched_outflow: 0,
            sched_description: 'Плановый горизонт',
            gap_probability_pct: 5,
            is_forecast: true
          };
        });

    // Map manual operations by date
    const manualDeltasByDate: Record<string, { in: number; out: number; items: string[] }> = {};
    for (const op of manualOperations) {
      if (!manualDeltasByDate[op.date]) {
        manualDeltasByDate[op.date] = { in: 0, out: 0, items: [] };
      }
      if (op.type === 'inflow') {
        manualDeltasByDate[op.date].in += op.amount;
        manualDeltasByDate[op.date].items.push(`+${formatRub(op.amount)} (${op.category})`);
      } else {
        manualDeltasByDate[op.date].out += op.amount;
        manualDeltasByDate[op.date].items.push(`-${formatRub(op.amount)} (${op.category})`);
      }
    }

    const adjusted: Array<{ pt: ForecastPoint; adjIn: number; adjOut: number; schedDesc: string }> = [];
    const cumDelta: number[] = [];
    let delta = 0;

    let totalSimInflow = 0;
    let totalSimOutflow = 0;

    // Delays are modelled as time shifts, so money is conserved: a payment due on day i
    // arrives on day i + delay, and payments shifted past the horizon drop out of it.
    const revenueFactor = whatIfRevenueFactor / 100;
    const delayShare = whatIfDelayDays > 0 ? whatIfDelaySharePct / 100 : 0;
    const operatingOut = basePts.map(p => Math.max(0, p.outflow - (p.sched_outflow || 0)));

    for (let idx = 0; idx < basePts.length; idx++) {
      const pt = basePts[idx];
      const dayNum = idx + 1;

      // 1. What-If Scenario Perturbations
      // Customer delay: the large client's share of inflows is shifted by whatIfDelayDays
      const delayedFrom = idx - whatIfDelayDays;
      let adjIn = pt.inflow * revenueFactor * (1 - delayShare);
      if (delayShare > 0 && delayedFrom >= 0) {
        adjIn += basePts[delayedFrom].inflow * revenueFactor * delayShare;
      }

      // Supplier postponement: operating (non-calendar) outflows are shifted; salary, taxes and rent stay put
      const postponedFrom = idx - whatIfPostponeSupplierDays;
      let adjOut = pt.outflow - operatingOut[idx];
      if (postponedFrom >= 0) {
        adjOut += operatingOut[postponedFrom];
      }

      // One-off what-if expense
      if (whatIfOneOffExpense > 0 && dayNum === whatIfOneOffDate) {
        adjOut += whatIfOneOffExpense;
      }

      // 2. Discrete Manual Operations Integration
      const manualForDay = manualDeltasByDate[pt.date];
      if (manualForDay) {
        adjIn += manualForDay.in;
        adjOut += manualForDay.out;
      }

      totalSimInflow += adjIn;
      totalSimOutflow += adjOut;

      // The scenario changes expected flows; the same shift is applied to every simulated path
      delta += (adjIn - adjOut) - (pt.inflow - pt.outflow);
      cumDelta.push(delta);

      let schedDesc = pt.sched_description;
      if (manualForDay && manualForDay.items.length > 0) {
        schedDesc = (schedDesc ? schedDesc + ' | ' : '') + 'Ручные операции: ' + manualForDay.items.join(', ');
      }
      adjusted.push({ pt, adjIn, adjOut, schedDesc });
    }

    // Quantiles and gap probability over the historical scenarios returned by the server
    const dist = summarizePaths(baseBalance, statementData?.paths ?? [], cumDelta, safetyBuffer);
    const updatedForecast: ForecastPoint[] = adjusted.map(({ pt, adjIn, adjOut, schedDesc }, idx) => ({
      ...pt,
      balance_p50: Math.round(dist.p50[idx]),
      balance_p10: Math.round(dist.p10[idx]),
      balance_p90: Math.round(dist.p90[idx]),
      inflow: Math.round(adjIn),
      outflow: Math.round(adjOut),
      sched_description: schedDesc,
      gap_probability_pct: dist.gapProb[idx]
    }));
    const generatedAlerts = buildGapAlerts(updatedForecast, safetyBuffer);

    const minP50 = updatedForecast.length > 0 ? Math.min(...updatedForecast.map(f => f.balance_p50)) : 0;
    const minP10 = updatedForecast.length > 0 ? Math.min(...updatedForecast.map(f => f.balance_p10)) : 0;
    // Probability that the balance drops below the buffer on any day of the horizon
    const maxGapProb = updatedForecast.length > 0 ? dist.overallGapProb : 0;
    // Rule-based remarks (owner withdrawals, thin cushion, past lows) are shown next to the status, not in it
    const remarks = (statementData?.insights ?? []).filter(i => i.severity !== 'INFO').length;
    const criticalRemarks = (statementData?.insights ?? []).some(i => i.severity === 'CRITICAL');

    let riskLevel: 'CRITICAL' | 'WARNING' | 'STABLE' = 'STABLE';
    let statusLabel = 'Стабильное состояние (В норме)';
    if (maxGapProb >= 50 || minP50 < safetyBuffer) {
      riskLevel = 'CRITICAL';
      statusLabel = 'Критический кассовый разрыв';
    } else if (maxGapProb >= 10 || minP10 < safetyBuffer) {
      riskLevel = 'WARNING';
      statusLabel = 'Повышенный риск кассового разрыва';
    }

    const t1 = performance.now();
    const elapsed = Math.round((t1 - t0) * 10) / 10;
    setRecalcTimeMs(Math.max(0.8, elapsed));

    return {
      effectiveForecast: updatedForecast,
      effectiveAlerts: generatedAlerts,
      effectiveCurrentBalance: baseBalance,
      summaryStats: {
        minP50,
        minP10,
        maxGapProb,
        totalInflow: Math.round(totalSimInflow),
        totalOutflow: Math.round(totalSimOutflow),
        netFlow: Math.round(totalSimInflow - totalSimOutflow),
        riskLevel,
        statusLabel,
        remarks,
        criticalRemarks
      }
    };
  }, [
    statementData,
    manualOperations,
    whatIfDelayDays,
    whatIfDelaySharePct,
    whatIfRevenueFactor,
    whatIfPostponeSupplierDays,
    whatIfOneOffExpense,
    whatIfOneOffDate,
    safetyBuffer
  ]);

  const hasData = Boolean(statementData || manualOperations.length > 0);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Top Banner & Header */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white font-bold text-xl">
              Ф
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white">FinRadar</h1>
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-medium">
                  Банк Центр-инвест
                </span>
                {hasData ? (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-medium flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    Анализ активен
                  </span>
                ) : (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-medium flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                    Ожидание выписки
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400">
                Интеллектуальный сервис прогнозирования ДДС, детекции коллизий и ликвидации кассовых разрывов
              </p>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2 flex-wrap">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".csv,.txt,.tsv"
              className="hidden"
            />

            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-indigo-600/20 transition-all cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              {hasData ? 'Загрузить другую выписку' : 'Загрузить CSV-выписку'}
            </button>

            <button
              onClick={() => setIsManualModalOpen(true)}
              className="px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 text-xs font-semibold border border-emerald-500/40 flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <PlusCircle className="w-3.5 h-3.5 text-emerald-400" />
              Ввести операцию вручную
            </button>

            <button
              onClick={downloadC052Template}
              title="Скачать эталонный шаблон CSV"
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 flex items-center gap-1 transition-all cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-slate-400" />
              Шаблон CSV
            </button>

            {hasData && (
              <button
                onClick={clearAllData}
                title="Очистить данные и вернуться к экрану ожидания"
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-rose-950/40 hover:text-rose-400 text-slate-400 border border-slate-700 transition-all cursor-pointer"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Status Context Strip */}
        <div className="bg-slate-900/60 border-t border-slate-800/80 px-4 sm:px-6 py-2">
          <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-slate-400 font-medium">Статус данных:</span>

              {statementData ? (
                <div className="flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/30 px-2.5 py-1 rounded-md text-indigo-300 font-medium">
                  <FileSpreadsheet className="w-3.5 h-3.5 text-indigo-400" />
                  <span>{statementFileName || statementData.company_id}</span>
                  <span className="text-slate-400 text-[11px]">
                    ({statementData.total_operations || 0} оп., сальдо: {formatExactRub(statementData.current_balance)})
                  </span>
                </div>
              ) : manualOperations.length > 0 ? (
                <div className="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-1 rounded-md text-emerald-300 font-medium">
                  <PlusCircle className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Ручной режим ввода ({manualOperations.length} операций)</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-slate-400 italic">
                  <span>Данные не загружены. Модель ожидает CSV-файл или ручной ввод предпринимателя.</span>
                </div>
              )}

              {/* Incremental Update Badge (Задачи со звездочкой 2, 4) */}
              {hasData && (
                <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-slate-800/80 border border-slate-700 text-slate-300 text-[11px] font-mono">
                  <Zap className="w-3 h-3 text-amber-400" />
                  <span>Инкрементальный пересчет: {recalcTimeMs} мс</span>
                  <span className="text-slate-500">|</span>
                  <span className="text-emerald-400">без повторного запроса к серверу</span>
                </div>
              )}
            </div>

            {hasData && (
              <div className="flex items-center gap-3">
                <span className="text-slate-400 hidden sm:inline">Текущий остаток:</span>
                <span className={`font-mono font-bold ${effectiveCurrentBalance < 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {formatExactRub(effectiveCurrentBalance)}
                </span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                    summaryStats.riskLevel === 'CRITICAL'
                      ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                      : summaryStats.riskLevel === 'WARNING'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                      : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                  }`}
                >
                  {summaryStats.statusLabel}
                </span>
                {summaryStats.remarks > 0 && <RemarksBadge count={summaryStats.remarks} critical={summaryStats.criticalRemarks} onOpen={openRemarks} />}
              </div>
            )}
          </div>
        </div>

        {/* Upload feedback banner */}
        {uploadFeedback && (
          <div className="bg-indigo-950/80 border-y border-indigo-800/80 px-4 py-2 text-xs flex items-center justify-between text-indigo-200 animate-fadeIn">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <span>{uploadFeedback}</span>
            </div>
            <button onClick={() => setUploadFeedback(null)} className="text-indigo-400 hover:text-white">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* 3 Core Workflow Tabs */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex overflow-x-auto no-scrollbar space-x-1 border-t border-slate-800/60 pt-1">
          <button
            onClick={() => setActiveTab('flows')}
            className={`px-4 py-2.5 text-xs sm:text-sm font-medium border-b-2 transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'flows'
                ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            1. Анализ денежных потоков (ДДС)
          </button>
          <button
            onClick={() => setActiveTab('forecast')}
            className={`px-4 py-2.5 text-xs sm:text-sm font-medium border-b-2 transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'forecast'
                ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            2. Прогноз остатка и кассовых разрывов
          </button>
          <button
            onClick={() => setActiveTab('whatif')}
            className={`px-4 py-2.5 text-xs sm:text-sm font-medium border-b-2 transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'whatif'
                ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
            }`}
          >
            <Sliders className="w-4 h-4" />
            3. Симулятор «Что если» (Стресс-тест)
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
        {/* ========================================================================= */}
        {/* EMPTY STATE / ONBOARDING VIEW (When no data has been loaded yet) */}
        {/* ========================================================================= */}
        {!hasData && (
          <div className="space-y-8 py-4 animate-fadeIn">
            {/* Hero Welcome Card */}
            <div className="relative overflow-hidden rounded-3xl border border-indigo-500/20 bg-gradient-to-b from-indigo-950/40 via-slate-900 to-slate-900 p-8 sm:p-10 shadow-2xl">
              <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>

              <div className="max-w-3xl space-y-4">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-semibold">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                  Модель прогноза готова к работе
                </div>

                <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white leading-tight">
                  Загрузите выписку вашей компании для анализа денежных потоков и прогноза кассовых разрывов
                </h2>

                <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
                  Сервис FinRadar работает строго с данными, загруженными предпринимателем. Загрузите файл выписки в формате CSV
                  или введите планируемые операции вручную. Модель классифицирует платежи, построит 30-дневный квантильный коридор
                  ликвидности и заранее предупредит о рисках.
                </p>
              </div>

              {/* Primary Action Box: Drag & Drop Zone */}
              <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Upload Action */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="group relative flex flex-col items-center justify-center p-6 border-2 border-dashed border-indigo-500/40 hover:border-indigo-400 rounded-2xl bg-indigo-950/20 hover:bg-indigo-900/20 cursor-pointer transition-all duration-200 text-center"
                >
                  <div className="w-12 h-12 rounded-xl bg-indigo-600/20 group-hover:bg-indigo-600/30 border border-indigo-500/40 flex items-center justify-center text-indigo-400 mb-3 transition-transform group-hover:scale-105">
                    <Upload className="w-6 h-6" />
                  </div>
                  <span className="text-sm font-bold text-white group-hover:text-indigo-300">
                    Загрузить CSV файл выписки
                  </span>
                  <span className="text-xs text-slate-400 mt-1">
                    Поддерживаются форматы выгрузок банков (1С, Excel CSV, разделитель ; или ,)
                  </span>
                </div>

                {/* Download Template Action */}
                <div
                  onClick={downloadC052Template}
                  className="group relative flex flex-col items-center justify-center p-6 border border-slate-800 hover:border-slate-700 rounded-2xl bg-slate-900/80 hover:bg-slate-850 cursor-pointer transition-all duration-200 text-center"
                >
                  <div className="w-12 h-12 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 mb-3 transition-transform group-hover:scale-105">
                    <Download className="w-6 h-6 text-slate-300" />
                  </div>
                  <span className="text-sm font-bold text-white group-hover:text-slate-200">
                    Скачать шаблон CSV
                  </span>
                  <span className="text-xs text-slate-400 mt-1">
                    Чистый эталонный файл с заголовками и примерами строк для легкого заполнения
                  </span>
                </div>

                {/* Manual Entry Action */}
                <div
                  onClick={() => setIsManualModalOpen(true)}
                  className="group relative flex flex-col items-center justify-center p-6 border border-emerald-500/30 hover:border-emerald-400/50 rounded-2xl bg-emerald-950/20 hover:bg-emerald-900/20 cursor-pointer transition-all duration-200 text-center"
                >
                  <div className="w-12 h-12 rounded-xl bg-emerald-600/20 group-hover:bg-emerald-600/30 border border-emerald-500/40 flex items-center justify-center text-emerald-400 mb-3 transition-transform group-hover:scale-105">
                    <PlusCircle className="w-6 h-6" />
                  </div>
                  <span className="text-sm font-bold text-white group-hover:text-emerald-300">
                    Ввести операции вручную
                  </span>
                  <span className="text-xs text-slate-400 mt-1">
                    Формат: тип операции, сумма, дата. Мгновенный расчет прогноза без файла
                  </span>
                </div>
              </div>

              {/* Fast Demo Pill */}
              <div className="mt-6 pt-6 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs">
                <div className="flex items-center gap-2 text-slate-400">
                  <Info className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                  <span>Для быстрой проверки жюри и экспертами доступен реальный демонстрационный датасет:</span>
                </div>
                <button
                  onClick={loadDemoC052}
                  disabled={isUploading}
                  className="px-3.5 py-1.5 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/40 text-indigo-200 font-semibold border border-indigo-500/40 flex items-center gap-1.5 transition-all cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                  Загрузить тестовую выписку C052 (в 1 клик)
                </button>
              </div>
            </div>

            {/* Feature Cards: Explaining Empty State Tabs */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
                <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                  <BarChart3 className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white">1. Анализ денежных потоков</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  После загрузки файла выписка автоматически раскладывается по категориям: выручка, налоги ЕНП, зарплатные авансы,
                  расчеты с поставщиками и аренда. Строится фактический баланс и ритм регулярных платежей.
                </p>
                <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 text-[11px] text-slate-500">
                  Ожидает загрузки данных выписки...
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
                <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                  <TrendingUp className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white">2. Прогноз и кассовые разрывы</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Модель аналоговых периодов рассчитывает траекторию остатка на 30 дней вперед (P10–P50–P90), находит дату и сумму нехватки
                  денег, определяет совпавшие платежи и дает готовые рекомендации Банка Центр-инвест.
                </p>
                <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 text-[11px] text-slate-500">
                  Ожидает загрузки данных выписки...
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
                <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                  <Sliders className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white">3. Симулятор «Что если»</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Стресс-моделирование: задержка оплаты от клиента на 2 недели, спад выручки на 20%, внеплановый ремонт.
                  Инкрементальный пересчет дельты происходит за 1-2 мс без повторного расчета прогноза на сервере.
                </p>
                <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 text-[11px] text-slate-500">
                  Ожидает загрузки данных выписки...
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* ACTIVE DATA VIEWS (Rendered when statementData or manualOperations exist) */}
        {/* ========================================================================= */}
        {hasData && (
          <>
            {/* Manual Operations Toolbar (visible across all tabs if operations exist) */}
            {manualOperations.length > 0 && (
              <div className="bg-slate-900/90 border border-emerald-500/30 rounded-2xl p-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span className="text-xs font-bold text-white uppercase tracking-wider">
                      Введенные вручную операции предпринимателя ({manualOperations.length})
                    </span>
                    <span className="text-[11px] text-emerald-400 font-mono">
                      [Дельта учтена в прогнозе мгновенно]
                    </span>
                  </div>
                  <button
                    onClick={() => setIsManualModalOpen(true)}
                    className="text-xs text-emerald-300 hover:text-emerald-200 font-semibold flex items-center gap-1 cursor-pointer"
                  >
                    <PlusCircle className="w-3.5 h-3.5" />
                    Добавить еще операцию
                  </button>
                </div>

                {/* Operations tags list */}
                <div className="flex flex-wrap gap-2 pt-1">
                  {manualOperations.map(op => (
                    <div
                      key={op.id}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-medium ${
                        op.type === 'inflow'
                          ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                          : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                      }`}
                    >
                      <span className="font-bold">
                        {op.type === 'inflow' ? '+ ' : '- '}
                        {formatExactRub(op.amount)}
                      </span>
                      <span className="text-slate-300 font-normal">({op.category})</span>
                      <span className="text-[11px] text-slate-400 font-mono">{op.date}</span>
                      <button
                        onClick={() => removeManualOperation(op.id)}
                        className="ml-1 text-slate-400 hover:text-rose-400 transition-colors"
                        title="Удалить операцию"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ========================================================================= */}
            {/* TAB 1: CASH FLOW ANALYSIS FOR LOADED COMPANY */}
            {/* ========================================================================= */}
            {activeTab === 'flows' && (
              <div className="space-y-6 animate-fadeIn">
                {/* Company Cash Flow Top KPI Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {/* Card 1: Balance */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Расчетный остаток
                      </span>
                      <Wallet className="w-4 h-4 text-indigo-400" />
                    </div>
                    <div className="mt-2 text-2xl font-bold font-mono text-white">
                      {formatExactRub(effectiveCurrentBalance)}
                    </div>
                    <div className="mt-2 text-xs flex items-center gap-1.5 text-slate-400">
                      <span className="inline-block w-2 h-2 rounded-full bg-emerald-400"></span>
                      <span>Фактическое сальдо на конец выписки</span>
                    </div>
                  </div>

                  {/* Card 2: Inflow */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">
                        Поступления (Выручка)
                      </span>
                      <ArrowDownRight className="w-4 h-4 text-emerald-400" />
                    </div>
                    <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">
                      {formatRub(statementData ? statementData.total_inflow || 0 : summaryStats.totalInflow)}
                    </div>
                    <div className="mt-2 text-xs text-slate-400">
                      {statementData
                        ? `Всего операций пополнения: ${statementData.categories?.filter(c => c.direction === 'Поступление').reduce((s, c) => s + (c.count || 0), 0) || '-'}`
                        : 'Сумма по ручным поступлениям'}
                    </div>
                  </div>

                  {/* Card 3: Outflow */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-rose-400 uppercase tracking-wider">
                        Списания (Расходы)
                      </span>
                      <ArrowUpRight className="w-4 h-4 text-rose-400" />
                    </div>
                    <div className="mt-2 text-2xl font-bold font-mono text-rose-400">
                      {formatRub(statementData ? statementData.total_outflow || 0 : summaryStats.totalOutflow)}
                    </div>
                    <div className="mt-2 text-xs text-slate-400">
                      ФОТ, налоги ЕНП, аренда и поставщики
                    </div>
                  </div>

                  {/* Card 4: Net Flow */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                        Чистый денежный поток
                      </span>
                      <TrendingUp className="w-4 h-4 text-indigo-400" />
                    </div>
                    <div
                      className={`mt-2 text-2xl font-bold font-mono ${
                        (statementData ? (statementData.net_cash_flow || 0) : summaryStats.netFlow) >= 0
                          ? 'text-emerald-400'
                          : 'text-rose-400'
                      }`}
                    >
                      {formatRub(statementData ? statementData.net_cash_flow || 0 : summaryStats.netFlow)}
                    </div>
                    <div className="mt-2 text-xs text-slate-400">
                      {(statementData ? (statementData.net_cash_flow || 0) : summaryStats.netFlow) >= 0
                        ? 'Положительный чистый приток'
                        : 'Дефицитный отток средств'}
                      {statementData && ' за весь период выписки'}
                    </div>
                  </div>
                </div>

                {/* History Timeline Chart */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-indigo-400" />
                        Динамика остатка и фактических потоков по выписке
                      </h3>
                      <p className="text-xs text-slate-400 mt-1">
                        История изменений баланса на конец каждого календарного дня
                      </p>
                      {statementData?.history && statementData.history.length > 1 && (
                        <ChartPeriod
                          from={statementData.history[0].date}
                          to={statementData.history[statementData.history.length - 1].date}
                          note={`последние ${statementData.history.length} дн. выписки`}
                        />
                      )}
                    </div>
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs font-medium">
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full bg-indigo-500"></span>
                        <span className="text-slate-300">Остаток на счете</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-sm bg-emerald-500"></span>
                        <span className="text-slate-300">Поступления</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-sm bg-rose-500"></span>
                        <span className="text-slate-300">Списания</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full bg-rose-500 ring-2 ring-rose-500/30"></span>
                        <span className="text-slate-300">Кассовый разрыв</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full bg-amber-500 ring-2 ring-amber-500/30"></span>
                        <span className="text-slate-300">Минимум остатка</span>
                      </div>
                    </div>
                  </div>

                  {/* SVG Chart */}
                  <div className="w-full bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
                    {statementData && statementData.history && statementData.history.length > 1 ? (
                      (() => {
                        const pts = statementData.history;
                        const balances = pts.map(p => p.balance);
                        // First day of every stretch with a negative balance, plus the lowest balance of the period
                        const markers: ChartMarker[] = [];
                        pts.forEach((p, i) => {
                          if (p.balance < 0 && (i === 0 || pts[i - 1].balance >= 0)) {
                            markers.push({ index: i, value: p.balance, color: '#f43f5e', label: 'Кассовый разрыв' });
                          }
                        });
                        const minIdx = balances.indexOf(Math.min(...balances));
                        if (!markers.some(m => m.index === minIdx)) {
                          markers.push({ index: minIdx, value: balances[minIdx], color: '#f59e0b', label: 'Минимум остатка' });
                        }
                        return (
                          <ZoomChart
                            dates={pts.map(p => p.date)}
                            lines={[{ label: 'Остаток на счете', color: '#6366f1', values: balances, width: 2.5 }]}
                            refLines={[{ label: 'Ноль', value: 0, color: '#475569' }]}
                            markers={markers}
                            bars={{ inflow: pts.map(p => p.inflow), outflow: pts.map(p => p.outflow) }}
                            height={190}
                          />
                        );
                      })()
                    ) : (
                      <div className="h-56 flex items-center justify-center text-xs text-slate-500">
                        {statementData
                          ? 'Обработано однократное сальдо компании'
                          : 'Введены разовые операции в ручном режиме'}
                      </div>
                    )}
                  </div>
                </div>

                {/* Categories & Payment Discipline Breakdown */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Category Structure */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-indigo-400" />
                      Структура ключевых статей расходов и доходов
                    </h3>
                    <p className="text-xs text-slate-400">
                      Автоматическая классификация операций выписки по статьям затрат и источникам выручки
                    </p>
                    {statementData?.period_start ? (
                      <ChartPeriod from={statementData.period_start} to={statementData.period_end} note="суммы за весь период выписки" />
                    ) : (
                      <p className="text-[11px] text-slate-500">Пример структуры: выписка ещё не загружена</p>
                    )}

                    <div className="space-y-3 pt-2">
                      {(statementData?.categories && statementData.categories.length > 0
                        ? statementData.categories
                        : [
                            { name: 'Выручка от покупателей', total_amount: 14500000, direction: 'Поступление' },
                            { name: 'Оплата счетов поставщикам', total_amount: 6800000, direction: 'Списание' },
                            { name: 'Зарплата и выплаты сотрудникам', total_amount: 3200000, direction: 'Списание' },
                            { name: 'Единый налоговый платеж (ЕНП)', total_amount: 1450000, direction: 'Списание' },
                            { name: 'Аренда помещений и складов', total_amount: 720000, direction: 'Списание' },
                            { name: 'Комиссии банка и РКО', total_amount: 85000, direction: 'Списание' }
                          ]
                      ).map((cat, idx) => (
                        <div key={idx} className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 space-y-1.5">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-slate-200">{cat.name}</span>
                            <span
                              className={`font-mono font-bold ${
                                cat.direction === 'Поступление' ? 'text-emerald-400' : 'text-rose-400'
                              }`}
                            >
                              {cat.direction === 'Поступление' ? '+' : '-'} {formatExactRub(cat.total_amount)}
                            </span>
                          </div>
                          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                cat.direction === 'Поступление' ? 'bg-emerald-500' : 'bg-rose-500'
                              }`}
                              style={{ width: `${Math.min(100, Math.max(10, (cat.total_amount / 14500000) * 100))}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Payment Calendar Rhythm & Counterparties */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-emerald-400" />
                      Календарный ритм регулярных обязательств
                    </h3>
                    <p className="text-xs text-slate-400">
                      Выявленные периодические пики оттока денежных средств компании
                    </p>
                    {statementData?.period_end && (
                      <ChartPeriod
                        from={[statementData.period_start ?? '', addDaysIso(statementData.period_end, -183)].sort()[1]}
                        to={statementData.period_end}
                        note="типичный день и сумма — медиана по месяцам за последние 6 мес."
                      />
                    )}

                    <div className="space-y-3 pt-2">
                      {statementData?.recurring_items && statementData.recurring_items.length > 0 ? (
                        statementData.recurring_items.map((item, idx) => (
                          <div
                            key={idx}
                            className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/80 flex items-center justify-between gap-3 text-xs"
                          >
                            <div className="space-y-0.5">
                              <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                                {item.category_name}
                              </div>
                              <div className="text-[11px] text-slate-400">
                                Получатель: {item.counterparty} • Типичный день: {item.typical_day}-е число
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="font-mono font-bold text-rose-400">
                                {formatExactRub(item.typical_amount)}
                              </div>
                              <div className="text-[10px] text-slate-500">ежемесячно</div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-xs text-slate-400 p-4 bg-slate-950/40 rounded-xl">
                          Календарные обязательства формируются на базе регулярных выплат (зарплата 10/25, налог ЕНП 28).
                        </div>
                      )}
                    </div>

                    {/* Button to proceed to forecast */}
                    <div className="pt-2">
                      <button
                        onClick={() => setActiveTab('forecast')}
                        className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white text-xs font-semibold flex items-center justify-center gap-2 shadow-md transition-all cursor-pointer"
                      >
                        Перейти к прогнозу остатка и оценке кассовых разрывов
                        <ArrowRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ========================================================================= */}
            {/* TAB 2: BALANCE FORECAST & CASH GAP COLLISION DETECTOR */}
            {/* ========================================================================= */}
            {activeTab === 'forecast' && (
              <div className="space-y-6 animate-fadeIn">
                {/* Risk Alert Header Banner */}
                <div
                  className={`p-5 rounded-2xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 ${
                    summaryStats.riskLevel === 'CRITICAL'
                      ? 'bg-rose-950/40 border-rose-500/40 text-rose-200'
                      : summaryStats.riskLevel === 'WARNING'
                      ? 'bg-amber-950/40 border-amber-500/40 text-amber-200'
                      : 'bg-emerald-950/40 border-emerald-500/40 text-emerald-200'
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-sm font-bold">
                      {summaryStats.riskLevel === 'CRITICAL' ? (
                        <ShieldAlert className="w-5 h-5 text-rose-400" />
                      ) : (
                        <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                      )}
                      <span>
                        {statementFileName || statementData?.company_id || 'Компания'}: {summaryStats.statusLabel}
                      </span>
                      {summaryStats.remarks > 0 && <RemarksBadge count={summaryStats.remarks} critical={summaryStats.criticalRemarks} onOpen={openRemarks} />}
                    </div>
                    <p className="text-xs text-slate-300 max-w-3xl leading-relaxed">
                      Квантильный прогноз на 30 дней вперед с учетом периодических обязательств (ЕНП 28-го, аванс 25-го, расчет 10-го)
                      и введенных разовых операций.
                      {statementData?.model && (
                        statementData.model.method === 'analog_periods'
                          ? ` Построен по ${statementData.model.scenarios} реальным 30-дневным периодам истории, начинавшимся около того же числа месяца: вероятность разрыва — доля периодов, в которых остаток ушёл бы ниже порога.`
                          : ` Истории меньше двух месяцев, поэтому прогноз собран из ${statementData.model.scenarios} сценариев случайных дней выписки.`
                      )}
                    </p>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <div className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">
                        Вероятность разрыва за 30 дней
                      </div>
                      <div className="text-2xl font-bold font-mono">
                        {summaryStats.maxGapProb.toFixed(1)}%
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">
                        Мин. остаток (P50)
                      </div>
                      <div className={`text-2xl font-bold font-mono ${summaryStats.minP50 < 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                        {formatExactRub(summaryStats.minP50)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Main 30-Day Liquidity Corridor Chart */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-indigo-400" />
                        30-дневный прогнозный коридор остатка на счете
                      </h3>
                      <p className="text-xs text-slate-400 mt-1">
                        Квантильные траектории: медиана P50, стресс-сценарий P10 (вероятность 90% не опуститься ниже) и оптимистичный P90
                      </p>
                      {effectiveForecast.length > 1 && (
                        <ChartPeriod
                          from={effectiveForecast[0].date}
                          to={effectiveForecast[effectiveForecast.length - 1].date}
                          note={
                            statementData?.period_end
                              ? `прогноз на ${effectiveForecast.length} дн. от остатка на ${ruDate(statementData.period_end)}`
                              : `прогноз на ${effectiveForecast.length} дн.`
                          }
                        />
                      )}
                    </div>

                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs font-medium">
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-0.5 bg-indigo-400"></span>
                        <span className="text-slate-300">Медиана P50</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-0 border-t-2 border-dashed border-rose-400"></span>
                        <span className="text-slate-300">Пессимистичный P10</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-0 border-t-2 border-dashed border-emerald-400"></span>
                        <span className="text-slate-300">Оптимистичный P90</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-sm bg-indigo-500/25"></span>
                        <span className="text-slate-300">Коридор P10–P90</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full bg-rose-500 ring-2 ring-rose-500/30"></span>
                        <span className="text-slate-300">
                          Точка кассового разрыва
                          {!effectiveForecast.some(p => p.balance_p10 < safetyBuffer) && (
                            <span className="text-slate-500"> (в этом прогнозе нет)</span>
                          )}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* SVG Chart */}
                  <div className="w-full bg-slate-950/70 rounded-xl p-3 border border-slate-800/80">
                    {effectiveForecast.length > 1 && (() => {
                      const pts = effectiveForecast;
                      const p50 = pts.map(p => p.balance_p50);
                      const p10 = pts.map(p => p.balance_p10);
                      const gapP50 = p50.findIndex(v => v < safetyBuffer);
                      const gapP10 = p10.findIndex(v => v < safetyBuffer);
                      const markers: ChartMarker[] = [];
                      if (gapP50 >= 0) markers.push({ index: gapP50, value: p50[gapP50], color: '#f43f5e', label: 'Разрыв по медиане P50' });
                      if (gapP10 >= 0) markers.push({ index: gapP10, value: p10[gapP10], color: '#f43f5e', label: 'Разрыв в стресс-сценарии P10' });
                      return (
                        <ZoomChart
                          dates={pts.map(p => p.date)}
                          lines={[
                            { label: 'Медиана P50', color: '#6366f1', values: p50, width: 2.5 },
                            { label: 'Пессимистичный P10', color: '#f43f5e', values: p10, dash: '5 4', width: 1.75 },
                            { label: 'Оптимистичный P90', color: '#10b981', values: pts.map(p => p.balance_p90), dash: '5 4', width: 1.75 }
                          ]}
                          band={{ label: 'Коридор P10–P90', color: '#6366f1', lower: p10, upper: pts.map(p => p.balance_p90) }}
                          refLines={[
                            { label: 'Ноль', value: 0, color: '#64748b' },
                            ...(safetyBuffer > 0 ? [{ label: 'Буфер', value: safetyBuffer, color: '#f59e0b' }] : [])
                          ]}
                          markers={markers}
                          height={230}
                        />
                      );
                    })()}
                  </div>
                </div>

                {/* Algorithmic Cash Gap Assessment Panel (Deficit, %, Cause, Prescriptions) */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-amber-400" />
                        Предупреждения о рисках и план действий (Prescription Engine)
                      </h3>
                      <p className="text-xs text-slate-400 mt-1">
                        Точная оценка: сумма нехватки, вероятность в %, коллизия совпавших платежей и персонализированные рекомендации
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4 pt-2">
                    {effectiveAlerts.length > 0 ? (
                      effectiveAlerts.map((alert, idx) => (
                        <div
                          key={idx}
                          className={`p-5 rounded-2xl border space-y-3 transition-all ${
                            alert.severity === 'CRITICAL'
                              ? 'bg-rose-950/20 border-rose-500/30'
                              : alert.severity === 'WARNING'
                              ? 'bg-amber-950/20 border-amber-500/30'
                              : 'bg-emerald-950/20 border-emerald-500/30'
                          }`}
                        >
                          {/* Alert Top Row */}
                          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
                            <div className="flex items-center gap-2">
                              <span
                                className={`px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wider ${
                                  alert.severity === 'CRITICAL'
                                    ? 'bg-rose-500 text-white'
                                    : alert.severity === 'WARNING'
                                    ? 'bg-amber-500 text-slate-950'
                                    : 'bg-emerald-500 text-white'
                                }`}
                              >
                                {alert.severity === 'CRITICAL'
                                  ? 'Критический разрыв'
                                  : alert.severity === 'WARNING'
                                  ? 'Повышенный риск'
                                  : 'Стабильно'}
                              </span>
                              <span className="text-xs text-slate-300 font-semibold flex items-center gap-1">
                                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                                Дата наступления: {alert.date}
                              </span>
                            </div>

                            {/* Risk Probability & Shortfall metrics */}
                            <div className="flex items-center gap-4 text-xs">
                              <div>
                                <span className="text-slate-400">Вероятность: </span>
                                <strong className="font-mono text-rose-400 text-sm font-bold">
                                  {alert.probability_pct}%
                                </strong>
                              </div>
                              {alert.deficit_rub > 0 && (
                                <div>
                                  <span className="text-slate-400">Сумма нехватки (дефицит): </span>
                                  <strong className="font-mono text-rose-400 text-sm font-bold">
                                    -{formatExactRub(alert.deficit_rub)}
                                  </strong>
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Cause (Colliding Obligations) */}
                          <div className="space-y-1">
                            <div className="text-[11px] font-bold uppercase text-slate-400 tracking-wider flex items-center gap-1.5">
                              <Info className="w-3.5 h-3.5 text-indigo-400" />
                              Причина (какие платежи совпали):
                            </div>
                            <p className="text-xs text-slate-200 bg-slate-950/60 p-3 rounded-xl border border-slate-800/60 leading-relaxed font-medium">
                              {alert.cause}
                            </p>
                          </div>

                          {/* Action Plan (What to do) */}
                          <div className="space-y-1">
                            <div className="text-[11px] font-bold uppercase text-emerald-400 tracking-wider flex items-center gap-1.5">
                              <Zap className="w-3.5 h-3.5 text-emerald-400" />
                              Что делать (Персональный финансовый план Банка Центр-инвест):
                            </div>
                            <div className="text-xs text-slate-200 bg-slate-950/60 p-3.5 rounded-xl border border-emerald-500/20 leading-relaxed space-y-1.5 whitespace-pre-line">
                              {alert.recommendation}
                            </div>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="p-6 bg-slate-950/40 border border-slate-800 rounded-xl text-center text-xs text-slate-400">
                        {(statementData?.insights ?? []).some(i => i.severity !== 'INFO')
                          ? 'Конкретных дней разрыва в типичном сценарии нет, но есть риски ликвидности — см. рекомендации ниже.'
                          : 'Кассовых разрывов не прогнозируется. Операционная ликвидность находится в норме.'}
                      </div>
                    )}
                  </div>

                  {/* Liquidity diagnostics that apply even when no specific gap day is forecast */}
                  {statementData?.insights && statementData.insights.length > 0 && (
                    <div id="cash-remarks" className="space-y-3 pt-2 scroll-mt-24">
                      <div className="text-[11px] font-bold uppercase text-slate-400 tracking-wider flex items-center gap-1.5">
                        <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                        Диагностика ликвидности и рекомендации
                      </div>
                      {statementData.insights.map((ins, idx) => (
                        <div
                          key={idx}
                          className={`p-4 rounded-xl border space-y-2 ${
                            ins.severity === 'CRITICAL'
                              ? 'bg-rose-950/20 border-rose-500/30'
                              : ins.severity === 'WARNING'
                              ? 'bg-amber-950/20 border-amber-500/30'
                              : 'bg-slate-950/40 border-slate-700'
                          }`}
                        >
                          <div className="flex items-start gap-2">
                            <span
                              className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                                ins.severity === 'CRITICAL'
                                  ? 'bg-rose-500 text-white'
                                  : ins.severity === 'WARNING'
                                  ? 'bg-amber-500 text-slate-950'
                                  : 'bg-slate-600 text-white'
                              }`}
                            >
                              {ins.severity === 'CRITICAL' ? 'Критично' : ins.severity === 'WARNING' ? 'Внимание' : 'Инфо'}
                            </span>
                            <span className="text-sm font-semibold text-white leading-snug">{ins.title}</span>
                          </div>
                          <p className="text-xs text-slate-300 leading-relaxed">{ins.detail}</p>
                          <p className="text-xs text-emerald-300 leading-relaxed">
                            <Zap className="w-3.5 h-3.5 inline mr-1 text-emerald-400" />
                            {ins.recommendation}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ========================================================================= */}
            {/* TAB 3: WHAT-IF SIMULATOR */}
            {/* ========================================================================= */}
            {activeTab === 'whatif' && (
              <div className="space-y-6 animate-fadeIn">
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
                  <div>
                    <h2 className="text-base font-bold text-white flex items-center gap-2">
                      <Sliders className="w-5 h-5 text-indigo-400" />
                      Интерактивный симулятор «Что если»
                    </h2>
                    <p className="text-xs text-slate-400 mt-1">
                      Моделируйте задержки оплат клиентов, колебания спроса и внеплановые списания. Инкрементальный пересчет
                      происходит мгновенно без повторного расчета прогноза на сервере.
                    </p>
                  </div>

                  {/* Sliders Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 bg-slate-950/60 p-5 rounded-2xl border border-slate-800">
                    {/* Slider 1: Customer delay */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-semibold">Задержка оплаты крупным клиентом:</span>
                        <span className="font-mono text-indigo-400 font-bold">+{whatIfDelayDays} дней</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="28"
                        step="1"
                        value={whatIfDelayDays}
                        onChange={(e) => setWhatIfDelayDays(parseInt(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>0 дней (В срок)</span>
                        <span>14 дней</span>
                        <span>28 дней (4 нед.)</span>
                      </div>
                      <div className="flex justify-between text-xs pt-1">
                        <span className="text-slate-400">Доля клиента в поступлениях:</span>
                        <span className="font-mono text-indigo-400 font-bold">{whatIfDelaySharePct}%</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="100"
                        step="5"
                        value={whatIfDelaySharePct}
                        onChange={(e) => setWhatIfDelaySharePct(parseInt(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>

                    {/* Slider 2: Revenue Shift */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-semibold">Изменение выручки (спрос):</span>
                        <span className={`font-mono font-bold ${whatIfRevenueFactor < 100 ? 'text-rose-400' : 'text-emerald-400'}`}>
                          {whatIfRevenueFactor}% ({whatIfRevenueFactor - 100 > 0 ? '+' : ''}{whatIfRevenueFactor - 100}%)
                        </span>
                      </div>
                      <input
                        type="range"
                        min="50"
                        max="150"
                        step="5"
                        value={whatIfRevenueFactor}
                        onChange={(e) => setWhatIfRevenueFactor(parseInt(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>-50% (Спад)</span>
                        <span>100% (Базовый)</span>
                        <span>+50% (Рост)</span>
                      </div>
                    </div>

                    {/* Slider 3: Supplier postponement */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-semibold">Отсрочка платежей поставщикам:</span>
                        <span className="font-mono text-emerald-400 font-bold">+{whatIfPostponeSupplierDays} дней</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="21"
                        step="1"
                        value={whatIfPostponeSupplierDays}
                        onChange={(e) => setWhatIfPostponeSupplierDays(parseInt(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>Без отсрочки</span>
                        <span>+7 дней</span>
                        <span>+21 день</span>
                      </div>
                    </div>

                    {/* Slider 4: One-off expense */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-semibold">Внеплановый расход (штраф/ремонт):</span>
                        <span className="font-mono text-rose-400 font-bold">{formatRub(whatIfOneOffExpense)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="2000000"
                        step="50000"
                        value={whatIfOneOffExpense}
                        onChange={(e) => setWhatIfOneOffExpense(parseInt(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>0 ₽</span>
                        <span>1.0 млн ₽</span>
                        <span>2.0 млн ₽</span>
                      </div>
                    </div>

                    {/* Slider 5: Expense Day */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-semibold">День внепланового расхода:</span>
                        <span className="font-mono text-indigo-400 font-bold">День +{whatIfOneOffDate}</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="30"
                        step="1"
                        value={whatIfOneOffDate}
                        onChange={(e) => setWhatIfOneOffDate(parseInt(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>День 1</span>
                        <span>День 15</span>
                        <span>День 30</span>
                      </div>
                    </div>

                    {/* Slider 6: Safety buffer */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-semibold">Неснижаемый буфер безопасности:</span>
                        <span className="font-mono text-amber-400 font-bold">{formatRub(safetyBuffer)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1000000"
                        step="50000"
                        value={safetyBuffer}
                        onChange={(e) => setSafetyBuffer(parseInt(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>0 ₽</span>
                        <span>500 тыс ₽</span>
                        <span>1 млн ₽</span>
                      </div>
                    </div>
                  </div>

                  {/* Reset sliders button */}
                  <div className="flex justify-end">
                    <button
                      onClick={() => {
                        setWhatIfDelayDays(0);
                        setWhatIfDelaySharePct(40);
                        setWhatIfRevenueFactor(100);
                        setWhatIfPostponeSupplierDays(0);
                        setWhatIfOneOffExpense(0);
                        setWhatIfOneOffDate(15);
                        setSafetyBuffer(0);
                      }}
                      className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 border border-slate-700 flex items-center gap-1.5 transition-all cursor-pointer"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      Сбросить параметры сценария к базовым
                    </button>
                  </div>

                  {/* Chart of Simulated vs Baseline */}
                  <div className="space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-indigo-400" />
                          Сравнение: Базовый прогноз vs Смоделированный сценарий «Что если»
                        </h3>
                        {effectiveForecast.length > 1 && (
                          <ChartPeriod
                            from={effectiveForecast[0].date}
                            to={effectiveForecast[effectiveForecast.length - 1].date}
                            note="медианный остаток (P50) на конец каждого дня горизонта"
                          />
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
                        <div className="flex items-center gap-1.5">
                          <span className="w-3 h-0 border-t-2 border-dashed border-slate-400"></span>
                          <span className="text-slate-400">Базовый прогноз</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="w-3 h-0.5 bg-indigo-400 font-bold"></span>
                          <span className="text-indigo-300 font-bold">Стресс-сценарий</span>
                        </div>
                        {safetyBuffer > 0 && (
                          <div className="flex items-center gap-1.5">
                            <span className="w-3 h-0 border-t-2 border-dashed border-amber-400"></span>
                            <span className="text-slate-400">Неснижаемый буфер</span>
                          </div>
                        )}
                        <div className="flex items-center gap-1.5">
                          <span className="w-3 h-3 rounded-full bg-rose-500 ring-2 ring-rose-500/30"></span>
                          <span className="text-slate-400">Разрыв в сценарии</span>
                        </div>
                      </div>
                    </div>

                    <div className="w-full bg-slate-950/70 rounded-xl p-3 border border-slate-800/80">
                      {effectiveForecast.length > 1 && (() => {
                        const simPts = effectiveForecast;
                        const basePts = statementData && statementData.forecast.length === simPts.length ? statementData.forecast : simPts;
                        const sim = simPts.map(p => p.balance_p50);
                        const gapDay = sim.findIndex(v => v < safetyBuffer);
                        return (
                          <ZoomChart
                            dates={simPts.map(p => p.date)}
                            lines={[
                              { label: 'Базовый прогноз', color: '#64748b', values: basePts.map(p => p.balance_p50), dash: '5 4', width: 2 },
                              { label: 'Стресс-сценарий', color: '#6366f1', values: sim, width: 3 }
                            ]}
                            refLines={[
                              { label: 'Ноль', value: 0, color: '#f43f5e' },
                              ...(safetyBuffer > 0 ? [{ label: 'Буфер', value: safetyBuffer, color: '#f59e0b' }] : [])
                            ]}
                            markers={gapDay >= 0 ? [{ index: gapDay, value: sim[gapDay], color: '#f43f5e', label: 'Разрыв в сценарии' }] : []}
                            height={200}
                          />
                        );
                      })()}
                    </div>
                  </div>

                  {/* Scenario Outcome Comparison Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                    <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
                      <div className="text-[11px] text-slate-400 uppercase font-semibold">Мин. остаток в сценарии</div>
                      <div className={`text-xl font-bold font-mono ${summaryStats.minP50 < 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                        {formatExactRub(summaryStats.minP50)}
                      </div>
                    </div>

                    <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
                      <div className="text-[11px] text-slate-400 uppercase font-semibold">Вероятность разрыва за 30 дней</div>
                      <div className="text-xl font-bold font-mono text-rose-400">
                        {summaryStats.maxGapProb.toFixed(1)}%
                      </div>
                    </div>

                    <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
                      <div className="text-[11px] text-slate-400 uppercase font-semibold">Рекомендуемый лимит ВКЛ</div>
                      <div className="text-xl font-bold font-mono text-indigo-400">
                        {summaryStats.minP10 < safetyBuffer ? formatExactRub((safetyBuffer - summaryStats.minP10) * 1.15) : "Не требуется"}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>

      {/* ========================================================================= */}
      {/* MODAL 1: FORMAT VALIDATION ERROR DIALOG */}
      {/* ========================================================================= */}
      {formatError?.isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fadeIn">
          <div className="bg-slate-900 border border-rose-500/40 rounded-3xl max-w-lg w-full p-6 space-y-5 shadow-2xl relative">
            <button
              onClick={() => setFormatError(null)}
              className="absolute top-5 right-5 text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3 text-rose-400">
              <div className="w-10 h-10 rounded-2xl bg-rose-500/20 border border-rose-500/30 flex items-center justify-center">
                <AlertOctagon className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">Неверный формат выписки</h3>
                <p className="text-xs text-rose-300">Файл «{formatError.fileName}» не распознан</p>
              </div>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Сервис не смог обнаружить обязательные столбцы для построения модели денежных потоков.
              Для корректного прогнозирования выписка должна содержать дату, сумму и направление операции.
            </p>

            {/* Headers Diagnostic Table */}
            <div className="space-y-2 bg-slate-950/80 p-4 rounded-2xl border border-slate-800 text-xs">
              <div className="space-y-1">
                <span className="font-semibold text-emerald-400">Обязательные поля (как в эталоне C052):</span>
                <p className="text-[11px] font-mono text-slate-300">
                  operation_date; amount; direction; counterparty_name; payment_purpose; balance_eod
                </p>
              </div>

              <div className="space-y-1 pt-2 border-t border-slate-800">
                <span className="font-semibold text-rose-400">Обнаружено в вашем файле:</span>
                <p className="text-[11px] font-mono text-slate-400">
                  {formatError.detectedHeaders.length > 0
                    ? formatError.detectedHeaders.join(', ')
                    : 'Заголовки не обнаружены (проверьте разделитель ; или ,)'}
                </p>
              </div>

              {formatError.missingRequired.length > 0 && (
                <div className="space-y-1 pt-2 border-t border-slate-800">
                  <span className="font-semibold text-amber-400">Отсутствуют критические поля:</span>
                  <p className="text-[11px] text-amber-300">
                    {formatError.missingRequired.join(', ')}
                  </p>
                </div>
              )}
            </div>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row gap-3 pt-2">
              <button
                onClick={() => {
                  downloadC052Template();
                  setFormatError(null);
                }}
                className="flex-1 py-2.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white text-xs font-bold flex items-center justify-center gap-2 shadow-md transition-all cursor-pointer"
              >
                <Download className="w-4 h-4" />
                Скачать эталонный шаблон CSV
              </button>

              <button
                onClick={() => {
                  setFormatError(null);
                  loadDemoC052();
                }}
                className="py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 flex items-center justify-center gap-1.5 transition-all cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                Открыть демо C052
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODAL 2: MANUAL OPERATION ENTRY DIALOG */}
      {/* ========================================================================= */}
      {isManualModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fadeIn">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-lg w-full p-6 space-y-5 shadow-2xl relative">
            <button
              onClick={() => setIsManualModalOpen(false)}
              className="absolute top-5 right-5 text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <PlusCircle className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">Ручной ввод разовой операции</h3>
                <p className="text-xs text-slate-400">
                  Формат: тип операции, сумма, дата (задача со звёздочкой 2 и 4)
                </p>
              </div>
            </div>

            <form onSubmit={handleAddManualOperation} className="space-y-4">
              {/* Type Switcher */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-300">Тип операции:</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setManualType('inflow');
                      setManualCategory('Выручка от покупателей');
                    }}
                    className={`py-2 px-3 rounded-xl text-xs font-bold border flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                      manualType === 'inflow'
                        ? 'bg-emerald-600 text-white border-emerald-500 shadow-md shadow-emerald-600/20'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:bg-slate-800'
                    }`}
                  >
                    <ArrowDownRight className="w-4 h-4" />
                    Пополнение (Приход)
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      setManualType('outflow');
                      setManualCategory('Оплата счетов поставщикам');
                    }}
                    className={`py-2 px-3 rounded-xl text-xs font-bold border flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                      manualType === 'outflow'
                        ? 'bg-rose-600 text-white border-rose-500 shadow-md shadow-rose-600/20'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:bg-slate-800'
                    }`}
                  >
                    <ArrowUpRight className="w-4 h-4" />
                    Списание (Расход)
                  </button>
                </div>
              </div>

              {/* Category */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-300">Категория операции:</label>
                <select
                  value={manualCategory}
                  onChange={(e) => setManualCategory(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-indigo-500"
                >
                  {manualType === 'inflow' ? (
                    <>
                      <option value="Выручка от покупателей">Выручка от покупателей (оплата по счету)</option>
                      <option value="Торговый эквайринг и СБП">Торговый эквайринг и СБП</option>
                      <option value="Разовый спецзаказ">Крупный разовый спецзаказ</option>
                      <option value="Возврат средств от контрагента">Возврат средств от контрагента</option>
                      <option value="Прочее пополнение">Прочее пополнение счета</option>
                    </>
                  ) : (
                    <>
                      <option value="Оплата счетов поставщикам">Оплата счетов поставщикам (закупка товара/сырья)</option>
                      <option value="Зарплата: Окончательный расчет (10-е число)">Зарплата: Окончательный расчет (10-е число)</option>
                      <option value="Зарплата: Аванс (25-е число)">Зарплата: Аванс (25-е число)</option>
                      <option value="Единый налоговый платеж (ЕНП 28-го числа)">Единый налоговый платеж (ЕНП 28-го числа)</option>
                      <option value="Аренда помещений и оборудования">Аренда помещений и оборудования</option>
                      <option value="Внеплановый расход / ремонт">Внеплановый расход (ремонт, штраф, поломка)</option>
                      <option value="Прочее списание">Прочее списание</option>
                    </>
                  )}
                </select>
              </div>

              {/* Amount and Date */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300">Сумма (₽):</label>
                  <input
                    type="number"
                    step="any"
                    placeholder="Например: 250000"
                    value={manualAmount}
                    onChange={(e) => setManualAmount(e.target.value)}
                    required
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-indigo-500 font-mono"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300">Дата операции:</label>
                  <input
                    type="date"
                    value={manualDate}
                    onChange={(e) => setManualDate(e.target.value)}
                    required
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              {/* Comment */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-300">Назначение / примечание (опционально):</label>
                <input
                  type="text"
                  placeholder="Оплата по договору №... / предоплата клиента"
                  value={manualComment}
                  onChange={(e) => setManualComment(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-indigo-500"
                />
              </div>

              {/* Note on incremental update */}
              <div className="p-3 bg-indigo-950/30 border border-indigo-500/20 rounded-xl text-[11px] text-indigo-300 flex items-center gap-2">
                <Zap className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                <span>
                  Операция будет сразу учтена в 30-дневном прогнозе остатка инкрементально за &lt; 2 мс без повторного расчета прогноза на сервере.
                </span>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsManualModalOpen(false)}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 transition-all cursor-pointer"
                >
                  Отмена
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-xs font-bold text-white shadow-md shadow-emerald-600/20 transition-all cursor-pointer"
                >
                  Применить к прогнозу
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 py-4 text-center text-xs text-slate-500">
        Сервис финансового прогнозирования для малого бизнеса • Хакатон «Центр-Инвест» • FinRadar 2026
      </footer>
    </div>
  );
}
