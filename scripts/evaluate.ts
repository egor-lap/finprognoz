// Quality check of the forecast on any statement file(s) in the case format, in one command:
//   npm run evaluate -- path/to/file.csv [more.csv ...]
// Runs the backtest (cutoffs are taken from the data) and draws the charts into reports/evaluate/<file name>/,
// so the presentation charts in presentation_assets/metrics stay untouched.

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// npm runs scripts from the project root; INIT_CWD is where the command was typed, so relative paths work from there
const baseDir = process.env.INIT_CWD ?? process.cwd();
const files = process.argv.slice(2).map(f => path.resolve(baseDir, f));
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

const backtest = spawnSync(process.execPath, ['scripts/backtest.ts', '--out', metricsFile, ...files], { stdio: 'inherit' });
if (backtest.status !== 0) process.exit(backtest.status ?? 1);

// Charts are drawn by Python + matplotlib: 'python' on Windows, 'python3' on macOS / Linux
for (const python of ['python', 'python3']) {
  const plot = spawnSync(python, ['scripts/plot_backtest.py', metricsFile, outDir], { stdio: 'inherit' });
  if (plot.error) continue; // this interpreter name is not installed, try the next one
  if (plot.status !== 0) {
    console.error('Графики не построены: проверьте, что установлены Python 3 и matplotlib (pip install matplotlib)');
    process.exit(plot.status ?? 1);
  }
  console.log(`\nГотово: метрики и графики в ${path.resolve(outDir)}`);
  process.exit(0);
}
console.error(`Python не найден. Метрики сохранены в ${metricsFile}; установите Python 3 и matplotlib и запустите:`);
console.error(`python scripts/plot_backtest.py ${metricsFile} ${outDir}`);
process.exit(1);
