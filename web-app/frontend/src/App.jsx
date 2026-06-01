import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000/api/train';

// ─── Canvas Render Engine ─────────────────────────────────────────────────────

const MARGIN = { top: 30, right: 24, bottom: 36, left: 46 };

function getPlotArea(w, h) {
  return {
    x0: MARGIN.left,
    y0: MARGIN.top,
    x1: w - MARGIN.right,
    y1: h - MARGIN.bottom,
    w: w - MARGIN.left - MARGIN.right,
    h: h - MARGIN.top - MARGIN.bottom,
  };
}

function dataToPixel(dx, dy, bounds, plot) {
  const [xMin, xMax, yMin, yMax] = bounds;
  const px = plot.x0 + ((dx - xMin) / (xMax - xMin)) * plot.w;
  const py = plot.y1 - ((dy - yMin) / (yMax - yMin)) * plot.h;
  return [px, py];
}

function pixelToData(px, py, bounds, plot) {
  const [xMin, xMax, yMin, yMax] = bounds;
  const dx = xMin + ((px - plot.x0) / plot.w) * (xMax - xMin);
  const dy = yMin + ((plot.y1 - py) / plot.h) * (yMax - yMin);
  return [dx, dy];
}

function drawAxes(ctx, bounds, plot) {
  const [xMin, xMax, yMin, yMax] = bounds;
  const { x0, y0, x1, y1, w, h } = plot;

  // Clip to plot area
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.clip();

  // Background
  const bgGrad = ctx.createRadialGradient(x0 + w/2, y0 + h/2, 0, x0 + w/2, y0 + h/2, Math.max(w, h) * 0.7);
  bgGrad.addColorStop(0, 'rgba(15, 30, 60, 0.95)');
  bgGrad.addColorStop(1, 'rgba(6, 13, 31, 0.98)');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(x0, y0, w, h);

  // Grid lines
  ctx.setLineDash([]);
  const ticks = 8;
  for (let i = 0; i <= ticks; i++) {
    const t = i / ticks;
    // Vertical
    const gx = x0 + t * w;
    ctx.strokeStyle = 'rgba(96, 165, 250, 0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(gx, y0); ctx.lineTo(gx, y1); ctx.stroke();
    // Horizontal
    const gy = y0 + t * h;
    ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x1, gy); ctx.stroke();
  }

  // Zero lines
  const zeroX = x0 + (0 - xMin) / (xMax - xMin) * w;
  const zeroY = y0 + (yMax - 0) / (yMax - yMin) * h;
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
  ctx.lineWidth = 1;
  if (zeroX >= x0 && zeroX <= x1) {
    ctx.beginPath(); ctx.moveTo(zeroX, y0); ctx.lineTo(zeroX, y1); ctx.stroke();
  }
  if (zeroY >= y0 && zeroY <= y1) {
    ctx.beginPath(); ctx.moveTo(x0, zeroY); ctx.lineTo(x1, zeroY); ctx.stroke();
  }

  ctx.restore();
}

function drawHeatmap(ctx, gridData, plot) {
  const { x0, y0, w, h } = plot;
  const { x: xs, y: ys, z } = gridData;

  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.clip();

  let zMin = Infinity, zMax = -Infinity;
  z.forEach(row => row.forEach(v => { zMin = Math.min(zMin, v); zMax = Math.max(zMax, v); }));

  const cellW = w / xs.length;
  const cellH = h / ys.length;

  for (let row = ys.length - 1; row >= 0; row--) {
    for (let col = 0; col < xs.length; col++) {
      const val = z[row][col];
      const t = (val - zMin) / (zMax - zMin);
      const midpoint = -zMin / (zMax - zMin);

      let r, g, b, a;
      if (t < midpoint) {
        const tt = Math.pow(1 - t / midpoint, 0.7);
        r = 239; g = 68; b = 68; a = tt * 0.35;
      } else {
        const tt = Math.pow((t - midpoint) / (1 - midpoint), 0.7);
        r = 16; g = 185; b = 129; a = tt * 0.35;
      }

      const px = x0 + col * cellW;
      const py = y0 + (ys.length - 1 - row) * cellH;
      ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
      ctx.fillRect(px, py, cellW + 1, cellH + 1);
    }
  }

  ctx.restore();
}

function drawContours(ctx, gridData, plot) {
  const { x0, y0, w, h } = plot;
  const { x: xs, y: ys, z } = gridData;

  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.clip();

  const cellW = w / xs.length;
  const cellH = h / ys.length;

  function drawLevel(threshold, strokeStyle, lineWidth, dash) {
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    for (let row = 0; row < ys.length - 1; row++) {
      for (let col = 0; col < xs.length - 1; col++) {
        const v  = z[row][col];
        const vr = z[row][col + 1];
        const vu = z[row + 1] ? z[row + 1][col] : v;
        const p = (a, b) => Math.abs(a - threshold) / (Math.abs(a - threshold) + Math.abs(b - threshold));
        const base_x = x0 + col * cellW;
        const base_y = y0 + (ys.length - 1 - row) * cellH;
        if ((v < threshold) !== (vr < threshold)) {
          const fx = p(v, vr); const px = base_x + fx * cellW;
          ctx.moveTo(px, base_y); ctx.lineTo(px, base_y + cellH);
        }
        if ((v < threshold) !== (vu < threshold)) {
          const fy = p(v, vu); const py = base_y - fy * cellH;
          ctx.moveTo(base_x, py); ctx.lineTo(base_x + cellW, py);
        }
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Margin lines (±1) — dashed subtle
  drawLevel(-1, 'rgba(255,255,255,0.2)', 1.2, [5, 5]);
  drawLevel( 1, 'rgba(255,255,255,0.2)', 1.2, [5, 5]);
  // Decision boundary (0) — solid bright
  drawLevel(0, 'rgba(255,255,255,0.92)', 2.5);

  ctx.restore();
}

function drawTickLabels(ctx, bounds, plot) {
  const [xMin, xMax, yMin, yMax] = bounds;
  const { x0, y0, y1, w, h } = plot;

  ctx.font = '11px Inter, sans-serif';
  ctx.fillStyle = 'rgba(120,144,176,0.7)';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) {
    const t = i / nTicks;
    const val = xMin + t * (xMax - xMin);
    const px = x0 + t * w;
    ctx.fillText(val.toFixed(1), px, y1 + 6);
  }

  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= nTicks; i++) {
    const t = i / nTicks;
    const val = yMin + t * (yMax - yMin);
    const py = y1 - t * h;
    ctx.fillText(val.toFixed(1), x0 - 6, py);
  }
}

function drawPlotBorder(ctx, plot) {
  const { x0, y0, w, h } = plot;
  ctx.strokeStyle = 'rgba(96, 165, 250, 0.2)';
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, w, h);
}

function drawPoints(ctx, points, labels, supportVectors, bounds, plot) {
  // Support vector halos (draw behind points)
  const svSet = new Set(supportVectors);
  points.forEach(([dx, dy], i) => {
    if (!svSet.has(i)) return;
    const [px, py] = dataToPixel(dx, dy, bounds, plot);
    const grad = ctx.createRadialGradient(px, py, 6, px, py, 18);
    grad.addColorStop(0, 'rgba(251, 191, 36, 0.6)');
    grad.addColorStop(1, 'rgba(251, 191, 36, 0)');
    ctx.beginPath();
    ctx.arc(px, py, 18, 0, 2 * Math.PI);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(px, py, 13, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)';
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Data points
  points.forEach(([dx, dy], i) => {
    const [px, py] = dataToPixel(dx, dy, bounds, plot);
    const isPos = labels[i] === 1;
    const col = isPos ? '#10b981' : '#ef4444';
    const glow = isPos ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)';

    // Glow
    const grad = ctx.createRadialGradient(px, py, 0, px, py, 14);
    grad.addColorStop(0, glow);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.arc(px, py, 14, 0, 2 * Math.PI);
    ctx.fillStyle = grad;
    ctx.fill();

    // Circle
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, 2 * Math.PI);
    ctx.fillStyle = col;
    ctx.fill();

    // White border
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

// ─── Canvas Component ─────────────────────────────────────────────────────────

function SVMCanvas({ points, labels, gridData, supportVectors, onCanvasClick }) {
  const canvasRef = useRef(null);
  const boundsRef = useRef([-5, 5, -5, 5]);

  if (gridData?.bounds) boundsRef.current = gridData.bounds;

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    const bounds = boundsRef.current;
    const plot = getPlotArea(W, H);

    ctx.clearRect(0, 0, W, H);

    drawAxes(ctx, bounds, plot);
    if (gridData?.z) {
      drawHeatmap(ctx, gridData, plot);
      drawContours(ctx, gridData, plot);
    }
    drawTickLabels(ctx, bounds, plot);
    drawPlotBorder(ctx, plot);
    drawPoints(ctx, points, labels, supportVectors, bounds, plot);
  }, [points, labels, gridData, supportVectors]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Use ResizeObserver for sharp rendering
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      render();
    });
    ro.observe(canvas.parentElement);
    return () => ro.disconnect();
  }, [render]);

  useEffect(() => { render(); }, [render]);

  const handleClick = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const plot = getPlotArea(rect.width, rect.height);
    if (px < plot.x0 || px > plot.x1 || py < plot.y0 || py > plot.y1) return;
    const [dx, dy] = pixelToData(px, py, boundsRef.current, plot);
    onCanvasClick(dx, dy);
  }, [onCanvasClick]);

  const showHint = points.length === 0;

  return (
    <div className="canvas-wrapper">
      <canvas ref={canvasRef} onClick={handleClick} />
      <div className={`canvas-hint ${showHint ? 'visible' : ''}`}>
        <div className="canvas-hint-icon">✦</div>
        <div className="canvas-hint-text">Clique no gráfico para adicionar pontos</div>
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [points, setPoints] = useState([]);
  const [labels, setLabels] = useState([]);
  const [activeClass, setActiveClass] = useState(1);
  const [kernel, setKernel] = useState('RBF');
  const [C, setC] = useState(1.0);
  const [gamma, setGamma] = useState(1.0);
  const [degree, setDegree] = useState(3);

  const [gridData, setGridData] = useState(null);
  const [supportVectors, setSupportVectors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('Clique no gráfico para adicionar pontos.');

  const posCount = labels.filter(l => l ===  1).length;
  const negCount = labels.filter(l => l === -1).length;

  useEffect(() => {
    if (!posCount || !negCount) {
      setStatus(
        !points.length          ? 'Clique no gráfico para adicionar pontos.' :
        !posCount               ? 'Adicione um ponto verde (Classe +1).' :
                                  'Adicione um ponto vermelho (Classe -1).'
      );
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      setError('');
      setStatus('⏳ Treinando SVM…');
      try {
        const res = await axios.post(API_URL, {
          points, labels, kernel,
          C: parseFloat(C), gamma: parseFloat(gamma), degree: parseInt(degree)
        });
        if (res.data.error) {
          setError(res.data.error);
          setStatus('Erro no treinamento.');
        } else {
          setGridData(res.data);
          setSupportVectors(res.data.support_vectors);
          setStatus(`✓ SVM treinada com ${res.data.support_vectors.length} vetores de suporte.`);
        }
      } catch {
        setError('Não foi possível conectar ao backend (porta 8000).');
        setStatus('Erro de conexão.');
      } finally {
        setLoading(false);
      }
    }, 400);

    return () => clearTimeout(timer);
  }, [points, labels, kernel, C, gamma, degree]);

  const handleCanvasClick = useCallback((x, y) => {
    setPoints(p => [...p, [x, y]]);
    setLabels(l => [...l, activeClass]);
  }, [activeClass]);

  const clearAll = () => {
    setPoints([]); setLabels([]);
    setGridData(null); setSupportVectors([]);
    setError(''); setStatus('Clique no gráfico para adicionar pontos.');
  };

  return (
    <div className="app-container">
      <header>
        <div className="header-logo">
          <h1>Simulador de SVM</h1>
          <div className="header-badge" style={{ marginTop: '8px' }}>
            <span className="badge badge-blue">NumPy Only</span>
            <span className="badge badge-green">SMO Solver</span>
            <span className="badge badge-gold">WSS-3</span>
          </div>
        </div>
      </header>

      <div className="main-content">
        {/* ── Sidebar ── */}
        <aside className="panel sidebar">

          <p className="section-label">Modelo</p>

          <div className="control-group">
            <label>Kernel</label>
            <select value={kernel} onChange={e => setKernel(e.target.value)}>
              <option value="LINEAR">Linear</option>
              <option value="RBF">RBF (Gaussiano)</option>
              <option value="POLY">Polinomial</option>
            </select>
          </div>

          <div className="control-group">
            <label>C (Regularização) <strong>{parseFloat(C).toFixed(1)}</strong></label>
            <input type="range" min="0.1" max="50" step="0.1" value={C}
              onChange={e => setC(e.target.value)} />
          </div>

          {kernel === 'RBF' && (
            <div className="control-group">
              <label>Gamma <strong>{parseFloat(gamma).toFixed(1)}</strong></label>
              <input type="range" min="0.1" max="5" step="0.1" value={gamma}
                onChange={e => setGamma(e.target.value)} />
            </div>
          )}

          {kernel === 'POLY' && (
            <div className="control-group">
              <label>Grau <strong>{degree}</strong></label>
              <input type="range" min="2" max="5" step="1" value={degree}
                onChange={e => setDegree(e.target.value)} />
            </div>
          )}

          <p className="section-label">Dados</p>

          <div className="control-group">
            <label>Classe ativa</label>
            <div className="class-selector">
              <button
                className={`class-btn class-pos ${activeClass === 1 ? 'active' : ''}`}
                onClick={() => setActiveClass(1)}>
                ● Classe +1
              </button>
              <button
                className={`class-btn class-neg ${activeClass === -1 ? 'active' : ''}`}
                onClick={() => setActiveClass(-1)}>
                ● Classe -1
              </button>
            </div>
          </div>

          <div style={{ marginTop: '4px' }}>
            <button className="btn-clear" onClick={clearAll}>🗑 Limpar tudo</button>
          </div>

          {error && <div className="error-box" style={{ marginTop: '10px' }}>⚠ {error}</div>}

          <div className="stats">
            <div className="stat-card">
              <span className="stat-label">Pontos totais</span>
              <span className="stat-value" style={{ color: '#60a5fa' }}>{points.length}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">● Classe +1</span>
              <span className="stat-value" style={{ color: '#10b981' }}>{posCount}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">● Classe -1</span>
              <span className="stat-value" style={{ color: '#ef4444' }}>{negCount}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">◉ Vetores de suporte</span>
              <span className="stat-value" style={{ color: '#fbbf24' }}>{supportVectors.length}</span>
            </div>
            <p className="status-msg">{loading ? '⏳ Calculando…' : status}</p>
          </div>
        </aside>

        {/* ── Plot ── */}
        <div className="panel visualization">
          <SVMCanvas
            points={points}
            labels={labels}
            gridData={gridData}
            supportVectors={supportVectors}
            onCanvasClick={handleCanvasClick}
          />
          <div className="legend-bar">
            <div className="legend-item">
              <div className="legend-dot" style={{ background: '#10b981' }} />
              <span>Classe +1</span>
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: '#ef4444' }} />
              <span>Classe −1</span>
            </div>
            <div className="legend-item">
              <div className="legend-line" style={{ background: 'rgba(255,255,255,0.8)' }} />
              <span>Fronteira de decisão</span>
            </div>
            <div className="legend-item">
              <div className="legend-line" style={{ background: 'rgba(255,255,255,0.2)' }} />
              <span>Margem (±1)</span>
            </div>
            <div className="legend-item">
              <div className="legend-ring" style={{ borderColor: '#fbbf24' }} />
              <span>Vetores de suporte</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
