/* Lightweight canvas charts. Every chart draws only what is passed in (simulation history). */
(function () {
  "use strict";

  const TEAL = "#27E0CF", ORANGE = "#F5B544", BLUE = "#6E9BFF", WHITE = "#F5F7FA", RED = "#EF6464";
  const GRID = "rgba(255,255,255,.08)", TXT = "#AEB8C6";
  const REG = new WeakMap();   // canvas -> { draw, last }
  const f1 = n => Number(n).toFixed(1);

  function prep(cv, h) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 400;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11.5px Manrope, 'Segoe UI', system-ui, sans-serif";
    return { ctx, w, h };
  }

  function niceMax(v) {
    if (v <= 0) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
  }

  function empty(ctx, w, h, msg) {
    ctx.fillStyle = TXT; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(msg, w / 2, h / 2);
  }

  function axes(ctx, w, h, pad, yMax, unit, ticks = 4) {
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
    for (let i = 0; i <= ticks; i++) {
      const v = yMax * i / ticks, y = pad.t + (h - pad.t - pad.b) * (1 - i / ticks);
      ctx.strokeStyle = i === 0 ? "rgba(255,255,255,.2)" : GRID;
      ctx.setLineDash(i === 0 ? [] : [3, 4]);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = TXT; ctx.fillText((Number.isInteger(v) ? v : +v.toFixed(1)) + (unit || ""), pad.l - 8, y);
    }
    ctx.setLineDash([]);
  }

  function xLabels(ctx, hist, xs, w, h, pad) {
    ctx.fillStyle = TXT; ctx.textBaseline = "top";
    const n = hist.length, step = Math.max(1, Math.ceil(n / Math.max(2, Math.floor((w - pad.l - pad.r) / 90))));
    for (let i = 0; i < n; i += step) {
      ctx.textAlign = xs(i) + 36 > w ? "right" : "center";
      ctx.fillText(hist[i].t.slice(6) || hist[i].t, Math.min(xs(i), w - 2), h - pad.b + 8);
    }
  }

  function bind(cv, tip, hist, xsOf, fmt, redraw) {
    const rec = REG.get(cv);
    if (rec && rec.bound) return;
    REG.get(cv).bound = true;
    cv.addEventListener("mousemove", e => {
      const r = REG.get(cv); if (!r || !r.hist || r.hist.length < 1) return;
      const rect = cv.getBoundingClientRect(), x = e.clientX - rect.left;
      let best = 0, bd = 1e9;
      r.hist.forEach((_, i) => { const d = Math.abs(r.xs(i) - x); if (d < bd) { bd = d; best = i; } });
      r.hover = best; r.redraw();
      if (tip) {
        const wrap = cv.parentElement.getBoundingClientRect();
        tip.hidden = false; tip.innerHTML = r.fmt(r.hist[best]);
        tip.style.left = Math.min(e.clientX - wrap.left, wrap.width - 190) + "px";
        tip.style.top = (e.clientY - wrap.top) + "px";
      }
    });
    cv.addEventListener("mouseleave", () => { const r = REG.get(cv); if (r) { r.hover = -1; r.redraw(); } if (tip) tip.hidden = true; });
  }

  /* Stacked bars (renewable + fossil MW) with the demand line on top. */
  function dispatchBars(cv, tip, hist, opts = {}) {
    const H = opts.height || 220, pad = { l: 44, r: 12, t: 12, b: 30 };
    const data = hist.slice(-(opts.max || 24));
    const state = REG.get(cv) || {}; REG.set(cv, state);
    const redraw = () => {
      const { ctx, w, h } = prep(cv, H);
      if (data.length < 1) { empty(ctx, w, h, "Advance the simulation clock to build the history."); state.hist = []; return; }
      const yMax = niceMax(Math.max(...data.map(p => Math.max(p.demand, p.renewable_mw + p.fossil_mw))) * 1.1);
      const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b, slot = iw / data.length, bw = Math.max(4, Math.min(34, slot * 0.58));
      const xs = i => pad.l + slot * (i + 0.5), ys = v => pad.t + ih * (1 - v / yMax);
      axes(ctx, w, h, pad, yMax, "");
      data.forEach((p, i) => {
        const x = xs(i) - bw / 2, hv = state.hover === i;
        const rh = ih * p.renewable_mw / yMax, fh = ih * p.fossil_mw / yMax;
        const g = ctx.createLinearGradient(0, ys(p.renewable_mw), 0, pad.t + ih);
        g.addColorStop(0, "rgba(39,224,207,.95)"); g.addColorStop(1, "rgba(39,224,207,.25)");
        ctx.fillStyle = g; ctx.fillRect(x, pad.t + ih - rh, bw, rh);
        const g2 = ctx.createLinearGradient(0, 0, 0, 1);
        ctx.fillStyle = hv ? "#ffc965" : "rgba(245,181,68,.78)"; ctx.fillRect(x, pad.t + ih - rh - fh, bw, fh);
        if (hv) { ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.strokeRect(x - .5, pad.t, bw + 1, ih); }
      });
      ctx.beginPath(); data.forEach((p, i) => (i ? ctx.lineTo(xs(i), ys(p.demand)) : ctx.moveTo(xs(i), ys(p.demand))));
      ctx.strokeStyle = WHITE; ctx.lineWidth = 1.8; ctx.lineJoin = "round"; ctx.stroke();
      data.forEach((p, i) => { ctx.beginPath(); ctx.arc(xs(i), ys(p.demand), 2.6, 0, 7); ctx.fillStyle = WHITE; ctx.fill(); });
      xLabels(ctx, data, xs, w, h, pad);
      Object.assign(state, { hist: data, xs });
    };
    Object.assign(state, { redraw, hist: data, fmt: p =>
      `<b>${p.t}</b><br>Demand ${f1(p.demand)} MW<br>Renewable ${f1(p.renewable_mw)} MW<br>Fossil ${f1(p.fossil_mw)} MW` +
      (p.curtailed > 0 ? `<br>Curtailed ${f1(p.curtailed)} MW` : "") });
    redraw();
    state.xs = state.xs || (() => 0);
    bind(cv, tip);
  }

  /* Multi-series line chart. series: [{key,color,label}]; yMax optional (fixed scale). */
  function lines(cv, tip, hist, series, opts = {}) {
    const H = opts.height || 220, pad = { l: 44, r: 14, t: 12, b: 30 }, unit = opts.unit || "";
    const state = REG.get(cv) || {}; REG.set(cv, state);
    const redraw = () => {
      const { ctx, w, h } = prep(cv, H);
      if (hist.length < 2) { empty(ctx, w, h, "Advance the simulation clock to build the history."); state.hist = []; return; }
      const vmax = Math.max(...series.flatMap(s => hist.map(p => +p[s.key] || 0)));
      const yMax = opts.yMax || niceMax(vmax * 1.1);
      const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b, n = hist.length;
      const xs = i => pad.l + iw * i / (n - 1), ys = v => pad.t + ih * (1 - Math.min(v, yMax) / yMax);
      axes(ctx, w, h, pad, yMax, unit);
      series.forEach((s, si) => {
        if (si === 0 && opts.area !== false) {
          const g = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);
          g.addColorStop(0, s.color + "55"); g.addColorStop(1, s.color + "00");
          ctx.beginPath(); ctx.moveTo(xs(0), ys(0));
          hist.forEach((p, i) => ctx.lineTo(xs(i), ys(p[s.key]))); ctx.lineTo(xs(n - 1), ys(0)); ctx.closePath();
          ctx.fillStyle = g; ctx.fill();
        }
        ctx.beginPath(); hist.forEach((p, i) => (i ? ctx.lineTo(xs(i), ys(p[s.key])) : ctx.moveTo(xs(i), ys(p[s.key]))));
        ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();
      });
      const hv = state.hover >= 0 ? state.hover : -1;
      if (hv >= 0 && hv < n) {
        ctx.strokeStyle = "rgba(255,255,255,.3)"; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(xs(hv), pad.t); ctx.lineTo(xs(hv), pad.t + ih); ctx.stroke(); ctx.setLineDash([]);
      }
      [n - 1, hv].forEach(i => { if (i < 0) return; series.forEach(s => {
        ctx.beginPath(); ctx.arc(xs(i), ys(hist[i][s.key]), 4, 0, 7);
        ctx.fillStyle = "#0b1015"; ctx.fill(); ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.stroke(); }); });
      xLabels(ctx, hist, xs, w, h, pad);
      Object.assign(state, { hist, xs });
    };
    Object.assign(state, { redraw, hist, fmt: p => `<b>${p.t}</b>` + series.map(s =>
      `<br><span style="color:${s.color}">&#9632;</span> ${s.label} ${(opts.fmt || f1)(p[s.key])}${unit}`).join("") });
    redraw();
    state.xs = state.xs || (() => 0);
    bind(cv, tip);
  }

  /* Horizontal bars for the model leaderboard. rows: [{label, value, text, best}] */
  function hbars(el, rows, opts = {}) {
    const max = opts.max || Math.max(...rows.map(r => r.value), 1e-9);
    el.innerHTML = rows.map(r =>
      `<div class="hb${r.best ? " best" : ""}"><span class="hb-l" title="${r.label}">${r.label}</span>` +
      `<span class="hb-t"><i style="width:${Math.max(2, r.value / max * 100)}%"></i></span><span class="hb-v">${r.text}</span></div>`).join("");
  }

  window.GGCharts = { dispatchBars, lines, hbars, colors: { TEAL, ORANGE, BLUE, WHITE, RED } };
  window.addEventListener("resize", () => { /* charts redraw via app render loop */ window.dispatchEvent(new Event("gg-resize")); });
})();
