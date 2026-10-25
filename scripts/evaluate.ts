// Forecast and quality check on any statement file(s) in the case format, in one command:
//   npm run evaluate -- path/to/file.csv [more.csv ...] [--open]
// Steps (each prints what it did):
//   1. forecast of every company exactly as on the site, with the calculation log   scripts/forecast.ts
//   2. backtest: forecasts from past month ends against the fact, all metrics         scripts/backtest.ts
//   3. charts of the metrics                                                          scripts/plot_backtest.py
//   4. one HTML report with everything                                                scripts/report.ts
// Output goes to reports/evaluate/<file name>/, so the presentation charts in presentation_assets/metrics stay
// untouched. --open opens the report in the browser at the end.

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// npm runs scripts from the project root; INIT_CWD is where the command was typed, so relative paths work from there
const baseDir = process.env.INIT_CWD ?? process.cwd();
const args = process.argv.slice(2);
const openReport = args.includes('--open');
const files = args.filter(a => a !== '--open').map(f => path.resolve(baseDir, f));
if (files.length === 0) {
  console.error('Укажите файл с выписками: npm run evaluate -- путь/к/файлу.csv');
  process.exit(1);
}
const missing = files.filter(f => !fs.existsSync(f));
if (missing.length > 0) {
  console.error(`Файл не найден: ${missing.join(', ')}`);
  process.exit(1);
}

const name = files.map(f => path.basename(f, path.extname(f))).join('+');
const outDir = path.join('reports', 'evaluate', name);
const metricsFile = path.join(outDir, 'backtest_metrics.json');
const errorFile = path.join(outDir, 'backtest_error.txt');
fs.mkdirSync(outDir, { recursive: true });
const step = (n: number, title: string) => console.log(`\n=== Шаг ${n}/4. ${title} ===`);

step(1, 'Прогноз по каждой компании (как на сайте) и журнал расчёта');
const forecast = spawnSync(process.execPath, ['scripts/forecast.ts', '--out-dir', outDir, ...files], { stdio: 'inherit' });
if (forecast.status !== 0) process.exit(forecast.status ?? 1);

step(2, 'Бэктест: прогнозы с прошлых дат против факта, метрики');
// Results of an earlier run must not be mistaken for this one
for (const f of [metricsFile, errorFile]) fs.rmSync(f, { force: true });
const backtest = spawnSync(process.execPath, ['scripts/backtest.ts', '--out', metricsFile, ...files], { stdio: ['inherit', 'inherit', 'pipe'], encoding: 'utf8' });
const backtestOk = backtest.status === 0;
if (!backtestOk) {
  // Too short a history is not fatal: the forecast and the report are still built
  const reason = (backtest.stderr || '').trim();
  console.error(reason);
  fs.writeFileSync(errorFile, reason);
  console.log('Бэктест не выполнен — отчёт будет без метрик.');
}

step(3, 'Графики метрик');
let plotted = false;
if (backtestOk) {
  // Charts are drawn by Python + matplotlib: 'python' on Windows, 'python3' on macOS / Linux
  for (const python of ['python', 'python3']) {
    const plot = spawnSync(python, ['scripts/plot_backtest.py', metricsFile, outDir], { stdio: 'inherit' });
    if (plot.error) continue; // this interpreter name is not installed, try the next one
    plotted = plot.status === 0;
    if (!plotted) console.error('Графики не построены: проверьте, что установлены Python 3 и matplotlib (pip install matplotlib)');
    break;
  }
  if (!plotted) console.log(`Отчёт будет без графиков; построить их отдельно: python scripts/plot_backtest.py ${metricsFile} ${outDir}`);
} else {
  console.log('Пропущено: нет метрик.');
}

step(4, 'HTML-отчёт');
const report = spawnSync(process.execPath, ['scripts/report.ts', outDir], { stdio: 'inherit' });
if (report.status !== 0) process.exit(report.status ?? 1);

const reportFile = path.resolve(outDir, 'report.html');
console.log(`\nГотово. Папка с результатами: ${path.resolve(outDir)}`);
console.log(`Отчёт для просмотра: ${reportFile}`);
if (openReport) {
  const opener = process.platform === 'win32' ? ['cmd', ['/c', 'start', '', reportFile]] : process.platform === 'darwin' ? ['open', [reportFile]] : ['xdg-open', [reportFile]];
  spawnSync(opener[0] as string, opener[1] as string[], { stdio: 'ignore' });
}
