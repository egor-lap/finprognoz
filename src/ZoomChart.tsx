import { useEffect, useId, useRef, useState } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';

// Interactive time chart for the dashboard: a balance panel (lines, a quantile band, reference lines, labeled
// points) and an optional inflow / outflow panel below it on the same date axis.
// Mouse: wheel zooms around the cursor, dragging selects a range to zoom into, double click resets.

export interface LineSeries {
  label: string;
  color: string;
  values: number[];
  dash?: string;
  width?: number;
}
export interface BandSeries {
  label: string;
  color: string;
  lower: number[];
  upper: number[];
}
export interface RefLine {
  label: string;
  value: number;
  color: string;
}
export interface ChartMarker {
  index: number;
  value: number;
  color: string;
  label: string;
}
export interface FlowBars {
  inflow: number[];
  outflow: number[];
}

interface Props {
  dates: string[];
  lines: LineSeries[];
  band?: BandSeries;
  refLines?: RefLine[];
  markers?: ChartMarker[];
  bars?: FlowBars;
  height?: number;
}

const PAD_L = 76;
const PAD_R = 14;
const PAD_T = 12;
const AXIS_H = 22;
const BARS_H = 96;
const PANEL_GAP = 14;
const GRID = '#1e293b';
const TEXT = '#94a3b8';
const INFLOW = '#10b981';
const OUTFLOW = '#f43f5e';

const rub = (v: number) => Math.round(v).toLocaleString('ru-RU') + ' ₽';
const ruDate = (iso: string) => iso.slice(0, 10).split('-').reverse().join('.');
const dayMonth = (iso: string) => `${iso.slice(8, 10)}.${iso.slice(5, 7)}`;

function compact(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)} млн`;
  if (a >= 1e3) return `${Math.round(v / 1e3)} тыс`;
  return `${Math.round(v)}`;
}

function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min || Math.abs(max) || 1;
  const raw = span / (count - 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) ?? 10 * mag;
  const ticks: number[] = [];
  for (let t = Math.floor(min / step) * step; t <= max + step * 1e-9; t += step) ticks.push(Math.round(t / step) * step);
  return ticks;
}

export default function ZoomChart({ dates, lines, band, refLines = [], markers = [], bars, height = 220 }: Props) {
  const n = dates.length;
  const clipId = `zc-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(800);
  const [view, setView] = useState<[number, number]>([0, Math.max(1, n - 1)]);
  const [hover, setHover] = useState<number | null>(null);
  const [drag, setDrag] = useState<{ from: number; to: number } | null>(null);

  // New statement (different number of days): show it whole
  useEffect(() => setView([0, Math.max(1, n - 1)]), [n]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => setWidth(Math.max(320, entries[0].contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const plotW = width - PAD_L - PAD_R;
  const [v0, v1] = view;
  const span = Math.max(1, v1 - v0);
  const x = (i: number) => PAD_L + ((i - v0) / span) * plotW;
  const indexAt = (px: number) => Math.min(v1, Math.max(v0, Math.round(v0 + ((px - PAD_L) / plotW) * span)));

  // Wheel zoom needs a non-passive listener so the page does not scroll while zooming
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (n < 3) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      setView(([a, b]) => {
        const s = Math.max(1, b - a);
        const center = a + ((e.clientX - rect.left - PAD_L) / (rect.width - PAD_L - PAD_R)) * s;
        const next = Math.min(n - 1, Math.max(2, Math.round(s * (e.deltaY < 0 ? 0.8 : 1.25))));
        let start = Math.round(center - ((center - a) / s) * next);
        start = Math.min(n - 1 - next, Math.max(0, start));
        return [start, start + next];
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [n]);

  if (n < 2) return null;

  const visible = Array.from({ length: v1 - v0 + 1 }, (_, k) => v0 + k);
  const domainVals = [
    ...lines.flatMap(l => visible.map(i => l.values[i])),
    ...(band ? visible.flatMap(i => [band.lower[i], band.upper[i]]) : []),
    ...refLines.map(r => r.value)
  ].filter(v => Number.isFinite(v));
  const lo = Math.min(...domainVals);
  const hi = Math.max(...domainVals);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.06;
  // The scale follows the visible data; round ticks are drawn inside it
  const yMin = lo - pad;
  const yMax = hi + pad;
  const ticks = niceTicks(yMin, yMax, 6).filter(t => t >= yMin && t <= yMax);
  const mainBottom = PAD_T + height;
  const y = (v: number) => mainBottom - ((v - yMin) / (yMax - yMin || 1)) * height;

  const barsTop = mainBottom + PANEL_GAP;
  const barsMax = bars ? Math.max(1, ...visible.flatMap(i => [bars.inflow[i] ?? 0, bars.outflow[i] ?? 0])) : 1;
  const barsMid = barsTop + BARS_H / 2;
  const by = (v: number) => (v / barsMax) * (BARS_H / 2 - 4);
  const barW = Math.max(1.5, Math.min(14, (plotW / (span + 1)) * 0.6));

  const totalH = (bars ? barsTop + BARS_H : mainBottom) + AXIS_H;
  const path = (vals: number[]) =>
    visible.map((i, k) => `${k === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(vals[i]).toFixed(1)}`).join(' ');

  const xTickCount = Math.min(visible.length, Math.max(2, Math.floor(plotW / 70)));
  const xTicks = [...new Set(Array.from({ length: xTickCount }, (_, k) => v0 + Math.round((k * span) / (xTickCount - 1))))];

  const visibleMarkers = markers.filter(m => m.index >= v0 && m.index <= v1);
  const zoomed = v0 > 0 || v1 < n - 1;

  const localX = (e: ReactMouseEvent) => e.clientX - (svgRef.current?.getBoundingClientRect().left ?? 0);
  const onMove = (e: ReactMouseEvent) => {
    const px = localX(e);
    if (drag) setDrag({ ...drag, to: px });
    setHover(px >= PAD_L && px <= width - PAD_R ? indexAt(px) : null);
  };
  const onUp = () => {
    if (drag && Math.abs(drag.to - drag.from) > 8) {
      let a = indexAt(Math.min(drag.from, drag.to));
      let b = indexAt(Math.max(drag.from, drag.to));
      if (b - a < 2) {
        a = Math.max(0, Math.min(a, n - 3));
        b = a + 2;
      }
      setView([a, b]);
    }
    setDrag(null);
  };

  const tipLeft = hover !== null ? (x(hover) + 220 > width ? x(hover) - 222 : x(hover) + 12) : 0;

  return (
    <div ref={wrapRef} className="relative w-full select-none">
      {zoomed && (
        <button
          type="button"
          onClick={() => setView([0, n - 1])}
          className="absolute right-2 top-1 z-10 px-2 py-0.5 rounded-md text-[10px] font-semibold bg-slate-800 border border-slate-700 text-slate-200 hover:bg-slate-700"
        >
          Сбросить масштаб
        </button>
      )}
      <svg
        ref={svgRef}
        width={width}
        height={totalH}
        className="block cursor-crosshair"
        onMouseMove={onMove}
        onMouseDown={e => setDrag({ from: localX(e), to: localX(e) })}
        onMouseUp={onUp}
        onMouseLeave={() => {
          setHover(null);
          setDrag(null);
        }}
        onDoubleClick={() => setView([0, n - 1])}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={PAD_L} y={0} width={plotW} height={totalH} />
          </clipPath>
        </defs>

        {/* Balance panel: grid and value labels */}
        {ticks.map(t => (
          <g key={t}>
            <line x1={PAD_L} x2={width - PAD_R} y1={y(t)} y2={y(t)} stroke={GRID} strokeWidth={1} />
            <text x={PAD_L - 8} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill={TEXT} fontFamily="ui-monospace, monospace">
              {compact(t)}
            </text>
          </g>
        ))}

        <g clipPath={`url(#${clipId})`}>
          {band && (
            <path
              d={`${path(band.upper)} ${[...visible].reverse().map(i => `L${x(i).toFixed(1)},${y(band.lower[i]).toFixed(1)}`).join(' ')} Z`}
              fill={band.color}
              fillOpacity={0.14}
              stroke="none"
            />
          )}
          {refLines.map(r => (
            <line key={r.label} x1={PAD_L} x2={width - PAD_R} y1={y(r.value)} y2={y(r.value)} stroke={r.color} strokeWidth={1.5} strokeDasharray="4 4" />
          ))}
          {lines.map(l => (
            <path
              key={l.label}
              d={path(l.values)}
              fill="none"
              stroke={l.color}
              strokeWidth={l.width ?? 2}
              strokeDasharray={l.dash}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
        </g>

        {/* Flow panel: inflows up, outflows down from a shared baseline */}
        {bars && (
          <g>
            <text x={PAD_L - 8} y={barsTop + 8} textAnchor="end" fontSize={10} fill={INFLOW} fontFamily="ui-monospace, monospace">
              +{compact(barsMax)}
            </text>
            <text x={PAD_L - 8} y={barsMid + 3.5} textAnchor="end" fontSize={10} fill={TEXT} fontFamily="ui-monospace, monospace">
              0
            </text>
            <text x={PAD_L - 8} y={barsTop + BARS_H - 2} textAnchor="end" fontSize={10} fill={OUTFLOW} fontFamily="ui-monospace, monospace">
              −{compact(barsMax)}
            </text>
            <line x1={PAD_L} x2={width - PAD_R} y1={barsMid} y2={barsMid} stroke="#334155" strokeWidth={1} />
            <g clipPath={`url(#${clipId})`}>
              {visible.map(i => (
                <g key={i}>
                  {(bars.inflow[i] ?? 0) > 0 && (
                    <rect x={x(i) - barW / 2} y={barsMid - by(bars.inflow[i])} width={barW} height={by(bars.inflow[i])} rx={Math.min(2, barW / 2)} fill={INFLOW} />
                  )}
                  {(bars.outflow[i] ?? 0) > 0 && (
                    <rect x={x(i) - barW / 2} y={barsMid} width={barW} height={by(bars.outflow[i])} rx={Math.min(2, barW / 2)} fill={OUTFLOW} />
                  )}
                </g>
              ))}
            </g>
          </g>
        )}

        {/* Labeled points (cash gap, minimum), drawn above the lines */}
        {visibleMarkers.map(m => {
          const mx = x(m.index);
          const my = y(m.value);
          const above = my - PAD_T > 28;
          const anchor = mx < PAD_L + 90 ? 'start' : mx > width - PAD_R - 90 ? 'end' : 'middle';
          return (
            <g key={`${m.label}-${m.index}`} pointerEvents="none">
              <circle cx={mx} cy={my} r={9} fill={m.color} fillOpacity={0.2} />
              <circle cx={mx} cy={my} r={5} fill={m.color} stroke="#020617" strokeWidth={2} />
              {visibleMarkers.length <= 4 && (
                <text x={mx} y={above ? my - 13 : my + 20} textAnchor={anchor} fontSize={10.5} fontWeight={600} fill="#e2e8f0">
                  {m.label}: {compact(m.value)} · {dayMonth(dates[m.index])}
                </text>
              )}
            </g>
          );
        })}

        {/* Date axis */}
        {xTicks.map(i => (
          <text key={i} x={x(i)} y={totalH - 6} textAnchor="middle" fontSize={10} fill={TEXT} fontFamily="ui-monospace, monospace">
            {dayMonth(dates[i])}
          </text>
        ))}

        {/* Hover crosshair */}
        {hover !== null && !drag && (
          <g pointerEvents="none">
            <line x1={x(hover)} x2={x(hover)} y1={PAD_T} y2={totalH - AXIS_H} stroke="#64748b" strokeWidth={1} />
            {lines.map(l => (
              <circle key={l.label} cx={x(hover)} cy={y(l.values[hover])} r={4} fill={l.color} stroke="#020617" strokeWidth={2} />
            ))}
          </g>
        )}

        {/* Drag selection */}
        {drag && Math.abs(drag.to - drag.from) > 2 && (
          <rect
            x={Math.max(PAD_L, Math.min(drag.from, drag.to))}
            y={PAD_T}
            width={Math.abs(drag.to - drag.from)}
            height={totalH - AXIS_H - PAD_T}
            fill="#6366f1"
            fillOpacity={0.15}
            stroke="#6366f1"
            strokeOpacity={0.6}
          />
        )}
      </svg>

      {hover !== null && !drag && (
        <div
          className="pointer-events-none absolute z-20 w-[210px] rounded-lg border border-slate-700 bg-slate-900/95 p-2.5 text-[11px] shadow-xl"
          style={{ left: tipLeft, top: PAD_T }}
        >
          <div className="mb-1.5 font-semibold text-white">{ruDate(dates[hover])}</div>
          {lines.map(l => (
            <div key={l.label} className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-slate-400">
                <span className="inline-block h-0.5 w-3" style={{ background: l.color }} />
                {l.label}
              </span>
              <span className="font-mono text-slate-100">{rub(l.values[hover])}</span>
            </div>
          ))}
          {band && (
            <div className="flex items-center justify-between gap-2">
              <span className="text-slate-400">{band.label}</span>
              <span className="font-mono text-slate-100">
                {compact(band.lower[hover])} … {compact(band.upper[hover])}
              </span>
            </div>
          )}
          {bars && (
            <>
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-400">Поступления</span>
                <span className="font-mono text-emerald-400">+{rub(bars.inflow[hover] ?? 0)}</span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-400">Списания</span>
                <span className="font-mono text-rose-400">−{rub(bars.outflow[hover] ?? 0)}</span>
              </div>
            </>
          )}
          {markers
            .filter(m => m.index === hover)
            .map(m => (
              <div key={m.label} className="mt-1 font-semibold" style={{ color: m.color }}>
                ● {m.label}
              </div>
            ))}
        </div>
      )}

      <p className="mt-1 text-[10px] text-slate-500">
        Колесо мыши — приблизить, выделите участок мышью — увеличить его, двойной клик — исходный масштаб
      </p>
    </div>
  );
}
