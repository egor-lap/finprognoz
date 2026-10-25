// Bank statement CSV: parsing, cleaning and rule-based categorization.
// Shared by the browser (upload), the server and the scripts (backtest, forecast), so the site and the
// quality check read a statement in exactly the same way.

import type { CalcStep } from './cashflow.ts';

// Column order of the case dataset; a file without a header row is read in this order
export const STANDARD_COLUMNS = ['company_id', 'operation_id', 'operation_date', 'amount', 'direction', 'counterparty_inn', 'counterparty_name', 'payment_purpose', 'balance_eod'];

export interface ParsedOperation {
  company: string; // '' when the file has no company column
  id: string;
  date: string; // YYYY-MM-DD
  amount: number; // > 0
  isCredit: boolean;
  category: string;
  counterparty: string;
  inn: string;
  purpose: string;
  balance: number | null; // end-of-day balance from the file
}

export interface ParsedStatement {
  operations: ParsedOperation[];
  companies: string[]; // in order of appearance
  rows: number;
  skipped: number; // rows without a valid date or amount
  headerFound: boolean;
  separator: string;
  headers: string[];
  missing: string[]; // required columns that were not found
}

// ---------------------------------------------------------------- categorization
// Cyrillic letters are not \w for JS regular expressions, so whole Russian words are matched with
// (^|[^а-яё])word($|[^а-яё]) instead of \b.

// Money that goes to the owner or to the company's own accounts elsewhere
export const OWNER_TRANSFER = /собственн[а-яё]* средств|на личн[а-яё]* (сч[её]т|карт)|на сч[её]т физического лица|дивиденд|подотч[её]т|займа учредител/i;

export const CATEGORY = {
  revenue: 'Выручка от покупателей',
  acquiring: 'Торговый эквайринг и СБП',
  ownIn: 'Собственные средства и займы владельца',
  creditIn: 'Кредиты банка',
  refundIn: 'Возвраты и прочие поступления',
  supplier: 'Оплата счетов поставщикам',
  owner: 'Выводы владельца и переводы на свои счета',
  tax: 'Налоги и взносы (ЕНП 28-го числа)',
  salaryAdvance: 'Зарплата: аванс',
  salaryFinal: 'Зарплата: окончательный расчёт',
  rent: 'Аренда помещений и оборудования',
  credit: 'Кредиты, лизинг и проценты',
  bankFee: 'Комиссии банка и РКО',
  legal: 'Штрафы и взыскания',
} as const;

// Checked in order: the first match wins. «Без налога (НДС)» in a supplier payment is not a tax,
// and «АО «…Банк»» in a transfer to the owner's account is not a bank fee.
const OUTFLOW_RULES: Array<[string, RegExp]> = [
  [CATEGORY.owner, OWNER_TRANSFER],
  [CATEGORY.tax, /единый налоговый|налогового органа|ндфл|страхов[а-яё]* взнос|уфк по|(^|[^а-яё])енп($|[^а-яё])|(^|[^а-яё])фнс($|[^а-яё])/i],
  [CATEGORY.salaryAdvance, /(заработн[а-яё]* плат|зарплат)[^.]*аванс|аванс[^.]*(заработн|зарплат|сотрудник)/i],
  [CATEGORY.salaryFinal, /заработн[а-яё]* плат|зарплат|по реестру/i],
  [CATEGORY.rent, /аренд/i],
  [CATEGORY.credit, /кредит|лизинг|займ|процент[а-яё]* за пользование/i],
  [CATEGORY.bankFee, /комисси|ведение сч[её]та|обслуживание сч[её]та|(^|[^а-яё])рко($|[^а-яё])/i],
  [CATEGORY.legal, /исполнительн[а-яё]* лист|взыскан|пристав|штраф|(^|[^а-яё])пени($|[^а-яё])/i],
];

const INFLOW_RULES: Array<[string, RegExp]> = [
  [CATEGORY.ownIn, /собственн[а-яё]* средств|займа учредител|беспроцентного займа/i],
  [CATEGORY.creditIn, /выдача кредита|кредит[а-яё]* .*транш/i],
  [CATEGORY.acquiring, /эквайринг|qr-код|по картам|мерчант|(^|[^а-яёa-z])сбп($|[^а-яёa-z])/i],
  [CATEGORY.refundIn, /возврат платежа|возврат гарантийного|страхового возмещения/i],
];

export function categorizeOperation(purpose: string, isCredit: boolean): string {
  const text = purpose.toLowerCase();
  for (const [category, re] of isCredit ? INFLOW_RULES : OUTFLOW_RULES) {
    if (re.test(text)) return category;
  }
  return isCredit ? CATEGORY.revenue : CATEGORY.supplier;
}

// ---------------------------------------------------------------- parsing
function splitCsvLine(line: string, sep: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === sep) { out.push(cur.trim()); cur = ''; }
    else cur += ch;
  }
  out.push(cur.trim());
  return out;
}

// DD.MM.YYYY or YYYY-MM-DD; null for anything else, including impossible dates like 31.02
export function parseDate(raw: string): string | null {
  const s = raw.trim();
  let iso: string;
  let m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (m) iso = `${m[1]}-${m[2]}-${m[3]}`;
  else if ((m = s.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})$/))) iso = `${m[3]}-${m[2].padStart(2, '0')}-${m[1].padStart(2, '0')}`;
  else return null;
  const t = new Date(iso + 'T00:00:00Z');
  return isNaN(t.getTime()) || t.toISOString().slice(0, 10) !== iso ? null : iso;
}

// Spaces as thousand separators and a decimal comma are accepted; Excel-damaged values («авг.18») are not
export function parseAmount(raw: string): number | null {
  const s = raw.replace(/[\s ]/g, '').replace(',', '.');
  if (!/^[-+]?\d+(\.\d+)?$/.test(s)) return null;
  return Number(s);
}

function findColumns(headers: string[]) {
  const find = (test: (h: string) => boolean) => headers.findIndex(test);
  const notInn = (h: string) => !/inn|инн/.test(h);
  return {
    company: find(h => /company|компан/.test(h)),
    id: find(h => /operation_id|номер операции|^id$/.test(h)),
    date: find(h => /date|дата/.test(h)),
    amount: find(h => /amount|сумма/.test(h)),
    direction: find(h => /direction|направлен|тип|type|credit|debit/.test(h)),
    inn: find(h => /inn|инн/.test(h)),
    // The name column, never the INN one: «counterparty_inn» comes before «counterparty_name» in the case files
    counterparty: find(h => notInn(h) && /name|наимен|контрагент|получател|плательщик|партнер|counterparty/.test(h)),
    purpose: find(h => /purpose|назначен|описан|комментар/.test(h)),
    balance: find(h => /balance|остаток|сальдо/.test(h)),
  };
}

export function parseStatementCsv(text: string): ParsedStatement {
  // Lines made only of separators (the case files had tens of thousands of them) are empty, not broken rows
  const lines = text.replace(/^﻿/, '').split(/\r?\n/).filter(l => l.replace(/[;,\t"\s]/g, '').length > 0 && !l.trim().startsWith('#'));
  const empty: ParsedStatement = { operations: [], companies: [], rows: 0, skipped: 0, headerFound: false, separator: ';', headers: [], missing: [] };
  if (lines.length === 0) return { ...empty, missing: ['operation_date', 'amount', 'direction'] };

  const first = lines[0];
  const separator = first.includes(';') ? ';' : first.includes('\t') ? '\t' : ',';
  const firstCells = splitCsvLine(first, separator);
  // No header row: the first line already looks like an operation in the standard column order
  const headerless = firstCells.length >= STANDARD_COLUMNS.length && parseDate(firstCells[2]) !== null && parseAmount(firstCells[3]) !== null;
  const headers = headerless ? STANDARD_COLUMNS : firstCells.map(h => h.toLowerCase().replace(/["']/g, ''));
  const col = findColumns(headers);

  const missing: string[] = [];
  if (col.date < 0) missing.push('operation_date (дата операции)');
  if (col.amount < 0) missing.push('amount (сумма)');
  if (col.direction < 0) missing.push('direction (направление: Поступление/Списание)');
  if (missing.length > 0) return { ...empty, separator, headers, missing };

  const operations: ParsedOperation[] = [];
  const companies: string[] = [];
  const seen = new Set<string>();
  let skipped = 0;
  const dataLines = headerless ? lines : lines.slice(1);
  for (const line of dataLines) {
    const c = splitCsvLine(line, separator);
    const get = (i: number) => (i >= 0 && i < c.length ? c[i] : '');
    const date = parseDate(get(col.date));
    const signed = parseAmount(get(col.amount));
    if (!date || signed === null || signed === 0) { skipped++; continue; }
    const dir = get(col.direction).toLowerCase();
    let isCredit: boolean;
    if (signed < 0 || /списан|расход|debit|дебет/.test(dir)) isCredit = false;
    else if (/поступлен|credit|кредит|приход/.test(dir)) isCredit = true;
    else isCredit = signed > 0;
    const purpose = get(col.purpose);
    const company = get(col.company);
    const balance = col.balance >= 0 ? parseAmount(get(col.balance)) : null;
    operations.push({
      company,
      id: get(col.id),
      date,
      amount: Math.abs(signed),
      isCredit,
      category: categorizeOperation(purpose, isCredit),
      counterparty: get(col.counterparty) || (isCredit ? 'Покупатель' : 'Контрагент'),
      inn: get(col.inn),
      purpose,
      balance,
    });
    if (!seen.has(company)) { seen.add(company); companies.push(company); }
  }
  return { operations, companies, rows: dataLines.length, skipped, headerFound: !headerless, separator, headers, missing };
}

// ---------------------------------------------------------------- cleaning
export interface CleanResult {
  operations: ParsedOperation[]; // one company, sorted by date (file order kept within a day)
  duplicatesRemoved: number;
  balanceBreaks: string[]; // dates where the end-of-day balance does not follow from the operations
  closingBalance: number | null; // end-of-day balance of the last date
}

// Checks the operations of one company against the end-of-day balance column:
// an exact duplicate row is dropped only when the balance confirms it is extra;
// days that still do not add up are reported (missing or damaged operations).
export function cleanCompanyOperations(ops: ParsedOperation[]): CleanResult {
  const sorted = ops.map((op, i) => ({ op, i })).sort((a, b) => (a.op.date < b.op.date ? -1 : a.op.date > b.op.date ? 1 : a.i - b.i)).map(x => x.op);
  const byDate = new Map<string, ParsedOperation[]>();
  for (const op of sorted) {
    const list = byDate.get(op.date);
    if (list) list.push(op);
    else byDate.set(op.date, [op]);
  }
  const kept: ParsedOperation[] = [];
  const breaks: string[] = [];
  let duplicatesRemoved = 0;
  let prevEod: number | null = null;
  const signed = (op: ParsedOperation) => (op.isCredit ? op.amount : -op.amount);
  for (const [date, dayOps] of byDate) {
    let day = [...dayOps];
    const eod = [...day].reverse().find(op => op.balance !== null)?.balance ?? null;
    if (prevEod !== null && eod !== null) {
      let diff = eod - (prevEod + day.reduce((s, op) => s + signed(op), 0));
      if (Math.abs(diff) > 1) {
        const key = (op: ParsedOperation) => `${op.amount}|${op.isCredit}|${op.counterparty}|${op.purpose}`;
        const firstSeen = new Set<string>();
        day = day.filter(op => {
          const k = key(op);
          if (!firstSeen.has(k)) { firstSeen.add(k); return true; }
          // Drop the repeated row only when it explains the gap in the balance exactly
          if (Math.abs(diff + signed(op)) <= 1) { diff += signed(op); duplicatesRemoved++; return false; }
          return true;
        });
        if (Math.abs(diff) > 1) breaks.push(date);
      }
    }
    kept.push(...day);
    if (eod !== null) prevEod = eod;
    else if (prevEod !== null) prevEod += day.reduce((s, op) => s + signed(op), 0);
  }
  const last = kept.length > 0 ? kept[kept.length - 1].date : null;
  const closing = last ? [...(byDate.get(last) ?? [])].reverse().find(op => op.balance !== null)?.balance ?? null : null;
  return { operations: kept, duplicatesRemoved, balanceBreaks: breaks, closingBalance: closing };
}

// ---------------------------------------------------------------- calculation log
// Russian plural form for a count: plural(5, 'операция', 'операции', 'операций') -> 'операций'
export function plural(n: number, one: string, few: string, many: string): string {
  const m10 = Math.abs(n) % 10;
  const m100 = Math.abs(n) % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
const ops = (n: number) => `${n} ${plural(n, 'операция', 'операции', 'операций')}`;

const SEPARATOR_NAME: Record<string, string> = { ';': '«;»', ',': '«,»', '\t': 'табуляция' };
const ruDate = (iso: string) => iso.split('-').reverse().join('.');

export function explainParsing(fileName: string, parsed: ParsedStatement, company: string, clean: CleanResult): CalcStep[] {
  const hasBalance = clean.operations.some(op => op.balance !== null);
  const breaks = clean.balanceBreaks;
  return [
    {
      title: 'Чтение файла',
      lines: [
        `Файл «${fileName}»: ${parsed.rows.toLocaleString('ru-RU')} ${plural(parsed.rows, 'строка', 'строки', 'строк')} с операциями, разделитель ${SEPARATOR_NAME[parsed.separator] ?? parsed.separator}, ` +
          (parsed.headerFound ? 'строка заголовка найдена.' : 'без строки заголовка — стандартный порядок колонок кейса.'),
        `Пропущено строк с некорректной датой или суммой: ${parsed.skipped}.`,
        parsed.companies.length > 1
          ? `Компаний в файле: ${parsed.companies.length}; прогноз строится для ${company} (${ops(clean.operations.length)}).`
          : `Операций для прогноза: ${clean.operations.length}.`,
      ],
    },
    {
      title: 'Сверка операций с остатком на конец дня',
      lines: !hasBalance
        ? ['Колонки остатка нет: остаток на каждый день считается из операций.']
        : [
            clean.duplicatesRemoved > 0
              ? `Удалено повторов строк, которые остаток не подтверждает: ${clean.duplicatesRemoved}.`
              : 'Лишних повторов строк нет.',
            breaks.length > 0
              ? `Дней, где остаток не сходится с операциями (пропуски или испорченные суммы в выписке): ${breaks.length} — ` +
                `${breaks.slice(0, 5).map(ruDate).join(', ')}${breaks.length > 5 ? ' …' : ''}. Остаток по дням берётся из колонки выписки.`
              : 'Остаток на конец каждого дня сходится с операциями.',
          ],
    },
  ];
}
