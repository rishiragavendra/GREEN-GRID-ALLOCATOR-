/* ============ Green Grid Allocator — frontend logic ============ */
(() => {
  "use strict";

  const POLL_MS = 10000;   // heartbeat only: the browser holds the grid state
  let lastState = null;
  let failedCount = 0;

  const SOURCE_META = {
    1: { c: "#fbbf24", glyph: "☀️" },  // Solar
    2: { c: "#38bdf8", glyph: "🍃" },  // Wind
    3: { c: "#2dd4bf", glyph: "🌊" },  // Hydro
    4: { c: "#a78bfa", glyph: "🔋" },  // Battery
    5: { c: "#fb923c", glyph: "🔥" },  // Gas
    6: { c: "#94a3b8", glyph: "⛏️" },  // Coal
    7: { c: "#f87171", glyph: "🛢️" },  // Diesel
  };
  const meta = s => SOURCE_META[s.type_id] || { c: "#94a3b8", glyph: "⚙️" };
  const f1 = n => Number(n).toFixed(1);

  /* ---------------- helpers ---------------- */
  function $(id) { return document.getElementById(id); }

  function toast(msg, kind = "ok") {
    const wrap = $("toastWrap");
    const el = document.createElement("div");
    el.className = "toast" + (kind !== "ok" ? " " + kind : "");
    el.textContent = (kind === "ok" ? "✅ " : kind === "err" ? "⚠️ " : "💡 ") + msg;
    wrap.appendChild(el);
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 320); }, 3200);
  }

  function setLive(on) {
    const b = $("liveBadge");
    b.classList.toggle("off", !on);
    $("liveLabel").textContent = on ? "LIVE" : "OFFLINE";
  }

  /* ---------------- data ---------------- */
  /* The server is stateless (required for Vercel): the full grid state lives in
     the browser and is sent with every request. */
  const stateOf = s => s && ({ sources: s.sources, demand_mw: s.demand_mw, index: s.index,
                              weather: s.weather, history: s.history });

  async function fetchState() {
    try {
      const res = await fetch(lastState ? "/api/health" : "/api/state");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const st = await res.json();
      if (!lastState) { lastState = st; render(st); }
      setLive(true);
      failedCount = 0;
    } catch (e) {
      failedCount++;
      setLive(false);
      if (failedCount === 2) toast("Can't reach the server — is it running?", "err");
    }
  }

  async function postJSON(url, body, silent = false) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...(body || {}), state: stateOf(lastState) }),
      });
      if (!res.ok) throw new Error((await res.text()).slice(0, 140));
      const st = await res.json();
      lastState = st;
      render(st);
      return st;
    } catch (e) {
      if (!silent) toast("Request failed: " + e.message, "err");
      throw e;
    }
  }

  /* ---------------- KPI cards ---------------- */
  const kpiDefs = [
    { id: "kDemand", label: "Demand", accent: "#38bdf8", fmt: s => f1(s.dispatch.demand), unit: "MW",
      extra: s => `${f1(s.dispatch.renewable_pct)}% renewable this hour` },
    { id: "kRenew", label: "Renewable", accent: "#34d399", fmt: s => f1(s.dispatch.renewable_pct), unit: "%",
      extra: s => `${f1(s.dispatch.renewable_used)} MW of demand served` },
    { id: "kCurt", label: "Curtailed", accent: "#fbbf24", fmt: s => f1(s.dispatch.curtailed), unit: "MW",
      extra: s => "surplus clean energy lost" },
    { id: "kCo2", label: "CO₂ Emitted", accent: "#fb923c", fmt: s => f1(s.dispatch.co2), unit: "t/h",
      extra: s => `${f1(s.dispatch.fossil_used)} MW from fossil` },
    { id: "kUnmet", label: "Unmet Demand", accent: "#f87171", fmt: s => f1(s.dispatch.unmet), unit: "MW",
      extra: s => (s.dispatch.unmet <= 0 ? "✅ grid fully balanced" : "⚠ demand not covered") },
    { id: "kBatt", label: "Battery SoC", accent: "#a78bfa", fmt: s => {
        const b = (s.sources || []).find(x => x.storage);
        return b ? f1((b.soc || 0) * 100) : "0.0"; }, unit: "%",
      extra: s => {
        const d = s.dispatch;
        if (d.battery_charged > 0) return `⚡ charging ${f1(d.battery_charged)} MW`;
        if (d.battery_discharged > 0) return `🔋 discharging ${f1(d.battery_discharged)} MW`;
        return "idle"; } },
  ];

  function buildKpis() {
    const wrap = $("kpis");
    kpiDefs.forEach((def, i) => {
      const el = document.createElement("div");
      el.className = "kpi";
      el.style.setProperty("--accent", def.accent);
      el.style.setProperty("--d", (i * 0.05) + "s");
      el.innerHTML =
        `<div class="k-label">${def.label}</div>` +
        `<div class="k-value" id="${def.id}">—<span class="k-unit"></span></div>` +
        `<div class="k-extra" id="${def.id}X">—</div>`;
      wrap.appendChild(el);
    });
  }

  /* ---------------- renderers ---------------- */
  function render(st) {
    renderKpis(st);
    renderMix(st);
    renderSources(st);
    renderTable(st);
    drawTrend(st);
    $("timeChip").textContent = `🕐 ${st.time} · ${st.weekday}`;
  }

  function renderKpis(st) {
    kpiDefs.forEach(def => {
      const valEl = $(def.id), extraEl = $(def.id + "X");
      const txt = def.fmt(st);
      valEl.innerHTML = txt + `<span class="k-unit">${def.unit}</span>`;
      extraEl.textContent = def.extra(st);
    });
  }

  function renderMix(st) {
    const d = st.dispatch, dem = Math.max(d.demand, 1e-9);
    const renP = Math.min(100, d.renewable_used / dem * 100);
    const fosP = Math.min(100 - renP, d.fossil_used / dem * 100);
    const unmP = Math.max(0, 100 - renP - fosP);
    $("segRen").style.width = renP + "%";
    $("segFos").style.width = fosP + "%";
    $("segUnmet").style.width = unmP + "%";
    $("mixNums").innerHTML =
      `<span class="n-ren">Renewable <b>${f1(d.renewable_used)} MW</b> (${f1(d.renewable_pct)}%)</span>` +
      `<span class="n-fos">Fossil <b>${f1(d.fossil_used)} MW</b></span>` +
      (d.unmet > 0 ? `<span class="n-unmet">Unmet <b>${f1(d.unmet)} MW</b></span>` : "") +
      (d.curtailed > 0 ? `<span class="n-cur">Curtailed <b>${f1(d.curtailed)} MW</b></span>` : "");
    $("mixSub").textContent = `${st.time} · renewable ${f1(d.renewable_pct)}%`;
  }

  function renderSources(st) {
    const wrap = $("sourceBars");
    const maxUsed = Math.max(1, ...(st.sources || []).map(s => Math.abs(s.used || 0)));
    wrap.innerHTML = "";
    (st.sources || []).forEach(s => {
      const m = meta(s);
      const used = s.used || 0;
      const charging = s.storage && used < 0;
      const width = Math.min(100, Math.abs(used) / maxUsed * 100);
      const valText = charging
        ? `−${f1(Math.abs(used))} <small>chg</small>`
        : (used > 0 ? `+${f1(used)}` : `<small>0.0</small>`);
      const row = document.createElement("div");
      row.className = "srow";
      row.style.setProperty("--c", m.c);
      row.innerHTML =
        `<div class="s-name"><span class="s-icon">${m.glyph}</span>${s.name}</div>` +
        `<div class="s-track"><div class="s-fill" style="width:${width}%"></div></div>` +
        `<div class="s-val">${valText} <small>MW</small></div>`;
      wrap.appendChild(row);
    });
  }

  function renderTable(st) {
    const body = $("srcBody");
    body.innerHTML = "";
    (st.sources || []).forEach(s => {
      const m = meta(s);
      const used = s.used || 0;
      const charging = s.storage && used < 0;
      const typeCls = s.storage ? "sto" : (s.renewable ? "ren" : "fos");
      const typeTxt = s.storage ? "Storage" : (s.renewable ? "Renewable" : "Fossil");
      let usedCell;
      if (charging) usedCell = `<td class="neg">−${f1(Math.abs(used))} chg</td>`;
      else if (used > 0.0001) usedCell = `<td class="${s.renewable && !s.storage ? "pos" : s.storage ? "pos" : "warn"}">+${f1(used)}</td>`;
      else usedCell = `<td class="zero">0.0</td>`;
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td><span class="t-src"><i class="t-dot" style="--c:${m.c}"></i>${m.glyph} ${s.name}</span></td>` +
        `<td><span class="t-type ${typeCls}">${typeTxt}</span></td>` +
        `<td class="r">${f1(s.capacity)}</td>` +
        `<td class="r">${f1(s.available)}</td>` +
        usedCell +
        `<td class="r">${s.storage ? f1((s.soc || 0) * 100) + "%" : "—"}</td>` +
        `<td class="r rm"><button class="row-x" data-rm="${s.id}" title="Remove ${s.name}" type="button">✕</button></td>`;
      body.appendChild(tr);
    });
    const totCap = (st.sources || []).reduce((a, s) => a + (s.capacity || 0), 0);
    $("tableSub").textContent = `${st.sources.length} sources · ${f1(totCap)} MW installed`;
  }

  /* ---------------- trend chart (canvas) ---------------- */
  const cv = $("trendChart"), ctx = cv.getContext("2d");
  let hoverIdx = -1, pulsePhase = 0;

  function drawTrend(st) {
    const hist = st.history || [];
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 600, h = 280;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    $("trendTip").hidden = true;

    if (hist.length < 2) {
      ctx.fillStyle = "#5d6f8a"; ctx.font = "13px 'Segoe UI', sans-serif"; ctx.textAlign = "center";
      ctx.fillText("Advance time to build the history…", w / 2, h / 2);
      return;
    }

    const padL = 42, padR = 14, padT = 16, padB = 30;
    const iw = w - padL - padR, ih = h - padT - padB;
    const n = hist.length;
    const xs = i => padL + (iw * i) / (n - 1);
    const ys = pct => padT + ih * (1 - Math.max(0, Math.min(100, pct)) / 100);

    // gridlines + y labels
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.font = "11px ui-monospace, Consolas, monospace";
    [0, 25, 50, 75, 100].forEach(v => {
      const y = ys(v);
      ctx.strokeStyle = "rgba(148,190,240,.09)"; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillStyle = "#5d6f8a"; ctx.fillText(v + "%", padL - 8, y);
    });
    // x labels (~5 ticks)
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let i = 0; i < n; i += Math.max(1, Math.floor((n - 1) / 5))) {
      ctx.fillStyle = "#5d6f8a"; ctx.fillText(hist[i].t, xs(i), padT + ih + 8);
    }

    // hover crosshair
    if (hoverIdx >= 0 && hoverIdx < n) {
      const hx = xs(hoverIdx);
      ctx.strokeStyle = "rgba(148,190,240,.35)"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, padT + ih); ctx.stroke();
      ctx.setLineDash([]);
    }

    // area fill
    const grad = ctx.createLinearGradient(0, padT, 0, padT + ih);
    grad.addColorStop(0, "rgba(52,211,153,.32)");
    grad.addColorStop(1, "rgba(52,211,153,0)");
    ctx.beginPath();
    ctx.moveTo(xs(0), padT + ih);
    for (let i = 0; i < n; i++) ctx.lineTo(xs(i), ys(hist[i].renewable_pct));
    ctx.lineTo(xs(n - 1), padT + ih);
    ctx.closePath(); ctx.fillStyle = grad; ctx.fill();

    // line
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = xs(i), y = ys(hist[i].renewable_pct);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(52,211,153,.55)"; ctx.shadowBlur = 10;
    ctx.stroke(); ctx.shadowBlur = 0;

    // hover point + latest point
    [hoverIdx, n - 1].forEach(i => {
      if (i < 0) return;
      ctx.beginPath(); ctx.arc(xs(i), ys(hist[i].renewable_pct), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "#34d399"; ctx.fill();
      ctx.strokeStyle = "#06281c"; ctx.lineWidth = 2; ctx.stroke();
    });
  }

  function pulseLatest() {
    pulsePhase += 0.05;
    const n = (lastState?.history || []).length;
    if (n < 2) { requestAnimationFrame(pulseLatest); return; }
    const el = $("latestDot") || (() => {
      const d = document.createElement("div"); d.id = "latestDot";
      d.style.cssText = "position:absolute;width:14px;height:14px;border-radius:50%;background:#34d399;" +
        "box-shadow:0 0 12px #34d399;transform:translate(-50%,-50%);pointer-events:none;z-index:4;";
      $("cardTrend").querySelector(".canvas-wrap").appendChild(d);
      return d;
    })();
    const hist = lastState.history, h = 280;
    const padL = 42, padR = 14, padT = 16, padB = 30;
    const iw = cv.clientWidth - padL - padR, ih = h - padT - padB;
    const pct = Math.max(0, Math.min(100, hist[n - 1].renewable_pct));
    el.style.left = (100 * (cv.clientWidth - padR) / cv.clientWidth) + "%";
    el.style.top = (100 * (padT + ih * (1 - pct / 100)) / h) + "%";
    const s = 1 + 0.35 * Math.sin(pulsePhase);
    el.style.transform = `translate(-50%,-50%) scale(${s})`;
    requestAnimationFrame(pulseLatest);
  }

  cv.addEventListener("mousemove", e => {
    const hist = lastState?.history || [];
    if (hist.length < 2) return;
    const rect = cv.getBoundingClientRect();
    const padL = 42, iw = rect.width - 42 - 14;
    const x = e.clientX - rect.left;
    const frac = Math.max(0, Math.min(1, (x - padL) / iw));
    hoverIdx = Math.round(frac * (hist.length - 1));
    drawTrend(lastState);
    const p = hist[hoverIdx];
    const tip = $("trendTip");
    tip.hidden = false;
    tip.style.left = (e.clientX - rect.left) + "px";
    tip.style.top = (e.clientY - rect.top) + "px";
    tip.innerHTML = `<b>${p.t}</b> · renewable ${f1(p.renewable_pct)}% · demand ${f1(p.demand)} MW` +
      (p.curtailed > 0 ? ` · curt ${f1(p.curtailed)} MW` : "");
  });
  cv.addEventListener("mouseleave", () => { hoverIdx = -1; if (lastState) drawTrend(lastState); });

  /* ---------------- modals ---------------- */
  function showModal(id) { $(id).hidden = false; }
  function hideModal(id) { $(id).hidden = true; }
  document.querySelectorAll(".modal").forEach(m => {
    m.addEventListener("click", e => { if (e.target === m) m.hidden = true; });
  });
  document.querySelectorAll("[data-close]").forEach(b => {
    b.addEventListener("click", () => hideModal(b.dataset.close));
  });
  $("guideBtn").addEventListener("click", () => showModal("guideModal"));

  function openWeather(st) {
    const w = st.weather || {};
    const map = [["wTemp", w.Temperature], ["wHum", w.Humidity], ["wWind", w["Wind Speed"]],
                 ["wGenDiff", w["general diffuse flows"]], ["wDiff", w["diffuse flows"]]];
    map.forEach(([id, v]) => {
      const el = $(id);
      el.value = "";
      el.placeholder = (v === null || v === undefined) ? "—" : Number(v).toFixed(2);
    });
    showModal("weatherModal");
  }
  $("btnWeather").addEventListener("click", () => { if (lastState) openWeather(lastState); });

  $("weatherForm").addEventListener("submit", async e => {
    e.preventDefault();
    const body = {};
    const map = [["wTemp", "Temperature"], ["wHum", "Humidity"], ["wWind", "Wind Speed"],
                 ["wGenDiff", "general diffuse flows"], ["wDiff", "diffuse flows"]];
    map.forEach(([id, key]) => {
      const v = $(id).value.trim();
      if (v === "") return;
      const num = parseFloat(v);
      if (!isNaN(num)) body[key] = num;
      else toast(`"${key}" is not a number — ignored`, "warn");
    });
    if (!Object.keys(body).length) { toast("All fields blank — nothing to update", "warn"); return; }
    try {
      await postJSON("/api/weather", body);
      hideModal("weatherModal");
      toast("Weather updated");
    } catch (_) { /* toast handled */ }
  });

  /* ---------------- add / remove source ---------------- */
  $("btnAddSource").addEventListener("click", () => {
    $("srcCap").value = "50";
    showModal("addSourceModal");
  });

  $("addSourceForm").addEventListener("submit", async e => {
    e.preventDefault();
    const type_id = parseInt($("srcType").value, 10);
    const capacity = parseFloat($("srcCap").value);
    if (!Number.isFinite(capacity) || capacity <= 0) { toast("Enter a valid capacity in MW", "warn"); return; }
    try {
      await postJSON("/api/sources", { type_id, capacity });
      hideModal("addSourceModal");
      toast(`Added source (${capacity} MW)`);
    } catch (_) { /* toast handled */ }
  });

  // per-row remove (delegated - rows are rebuilt every render)
  $("srcBody").addEventListener("click", async e => {
    const btn = e.target.closest("[data-rm]");
    if (!btn) return;
    btn.disabled = true;
    try {
      await postJSON("/api/sources/remove", { id: parseInt(btn.dataset.rm, 10) });
      toast("Source removed");
    } catch (_) { btn.disabled = false; /* toast handled */ }
  });

  /* ---------------- dock controls ---------------- */
  const act = (url, body, silent) => postJSON(url, body, silent).catch(() => {});
  $("btnFwd").addEventListener("click", () => act("/api/time", { hours: 1 }));
  $("btnBack").addEventListener("click", () => act("/api/time", { hours: -1 }));
  $("btnSetTime").addEventListener("click", () => {
    const v = $("timePick").value;
    if (!v) { toast("Pick a date & time first", "warn"); return; }
    act("/api/time", { datetime: v + ":00" });
  });
  $("btnDemandUp").addEventListener("click", () => act("/api/demand", { delta: 10 }));
  $("btnDemandDown").addEventListener("click", () => act("/api/demand", { delta: -10 }));
  $("demandSet").addEventListener("keydown", e => {
    if (e.key !== "Enter") return;
    const v = parseFloat($("demandSet").value);
    if (isNaN(v) || v < 0) { toast("Enter a valid demand in MW", "warn"); return; }
    act("/api/demand", { set: v });
  });

  let playTimer = null, playing = false;
  $("btnPlay").addEventListener("click", () => {
    playing = !playing;
    const b = $("btnPlay");
    if (playing) {
      b.textContent = "⏸"; b.classList.add("paused");
      playTimer = setInterval(() => act("/api/time", { hours: 1 }, true), 1000);
      toast("Auto-play on — advancing 1 hour per second", "info");
    } else {
      b.textContent = "▶"; b.classList.remove("paused");
      clearInterval(playTimer); playTimer = null;
    }
  });

  $("btnSave").addEventListener("click", async () => {
    try {
      localStorage.setItem("greenGridState", JSON.stringify(stateOf(lastState)));
      toast("State saved in this browser");
    } catch (_) { toast("Browser storage unavailable", "err"); }
  });
  $("btnLoad").addEventListener("click", async () => {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem("greenGridState")); } catch (_) {}
    if (!saved) { toast("No saved state in this browser", "err"); return; }
    try { await postJSON("/api/import", saved); toast("State loaded"); } catch (_) {}
  });

  $("btnExport").addEventListener("click", async () => {
    try {
      const doc = { sources: lastState.sources, demand_mw: lastState.demand_mw,
                    current_idx: lastState.index, weather: lastState.weather };
      const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "grid_export.json";
      a.click();
      URL.revokeObjectURL(a.href);
      toast("Grid exported as grid_export.json");
    } catch (_) { toast("Export failed", "err"); }
  });

  $("btnImport").addEventListener("click", () => $("importFile").click());
  $("importFile").addEventListener("change", async e => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    try {
      const txt = await file.text();
      const body = JSON.parse(txt);
      await postJSON("/api/import", body);
      toast(`Imported ${file.name}`);
    } catch (err) {
      toast(err.name === "SyntaxError" ? "Not valid JSON" : "Import failed: " + err.message, "err");
    }
  });

  /* ---------------- model performance panel ---------------- */
  /* Reads GET /api/model-performance once at boot. Entirely additive: it does
     not touch the simulator state, the 1.5s polling loop, or any existing
     control. If the endpoint is unavailable the panel degrades to a hint. */
  const TYPE_CLS = { "regressor": "reg", "ensemble": "ens", "classifier baseline": "clf" };
  const num = (v, nd) => (v === null || v === undefined || Number.isNaN(v))
    ? "—" : Number(v).toFixed(nd);
  const pct = v => (v === null || v === undefined || Number.isNaN(v))
    ? "—" : Number(v).toFixed(2) + "%";

  function perfShowMessage(html) {
    const empty = $("perfEmpty");
    if (!empty) return;
    empty.innerHTML = html;
    empty.hidden = false;
    $("perfBody").hidden = true;
  }

  function renderPerf(d) {
    if (!d || !d.available) {
      perfShowMessage(
        (d && d.message ? d.message.replace("`python test.py`", "<code>python test.py</code>")
                        : "Benchmark results are not available yet.") +
        "<br><span class=\"hint\">The rest of the dashboard is unaffected.</span>");
      return;
    }

    const rows = Array.isArray(d.models) ? d.models : [];
    if (!rows.length) { perfShowMessage("No benchmark rows were saved."); return; }

    $("perfEmpty").hidden = true;
    $("perfBody").hidden = false;

    const tolPctLabel = d.tolerance != null ? `±${(d.tolerance * 100).toFixed(0)}%` : "tolerance";
    $("perfTolHead").textContent = `Tol. Acc (${tolPctLabel})`;
    $("perfSub").textContent =
      `${rows.length} of the benchmarked models · ranked by ${d.ranking_rule || "tolerance accuracy"}`;

    const bestIndiv = d.best_individual, bestEns = d.best_ensemble;
    $("perfBestIndiv").textContent = bestIndiv ? bestIndiv.model : "—";
    $("perfBestIndivWhy").textContent = d.best_individual_reason || "—";
    $("perfBestEns").textContent = bestEns ? bestEns.model : "—";
    $("perfBestEnsWhy").textContent = d.best_ensemble_reason || "—";

    const bestNames = new Set([bestIndiv && bestIndiv.model, bestEns && bestEns.model]);
    const body = $("perfRows");
    body.innerHTML = "";
    rows.forEach(r => {
      const cls = TYPE_CLS[r.type] || "reg";
      const tr = document.createElement("tr");
      if (bestNames.has(r.model)) tr.className = "is-best";
      const rank = r.rank ? `<span class="perf-rank">#${r.rank}</span>` : `<span class="perf-rank">—</span>`;
      tr.innerHTML =
        `<td>${rank}<b>${r.model}</b></td>` +
        `<td><span class="m-type ${cls}">${r.type}</span></td>` +
        `<td class="r">${pct(r.tolerance_accuracy)}</td>` +
        `<td class="r derived-c">${num(r.derived_precision, 4)}</td>` +
        `<td class="r derived-c">${num(r.derived_recall, 4)}</td>` +
        `<td class="r derived-c">${num(r.derived_f1, 4)}</td>` +
        `<td class="r">${num(r.rmse, 1)}</td>` +
        `<td class="r">${num(r.r2, 4)}</td>`;
      body.appendChild(tr);
    });

    const th = d.demand_class_thresholds || {};
    const bits = [];
    if (d.production && d.production.demand) {
      bits.push(`production demand model: ${d.production.demand} ` +
                `(chosen on the validation split)`);
    }
    if (th.low_medium != null) {
      bits.push(`classes: Low < ${Number(th.low_medium).toFixed(0)} kW ` +
                `≤ Medium < ${Number(th.medium_high).toFixed(0)} kW ≤ High`);
    }
    if (d.generated_at_utc) bits.push(`benchmark run ${d.generated_at_utc} (${d.mode || "quick"} mode)`);
    $("perfMeta").textContent = bits.join(" · ");

    if (d.production) {
      const p = d.production, f = $("footModels");
      if (f && p.demand) {
        f.textContent = `Demand: ${p.demand} · Solar: ${p.solar || "—"} · ` +
                        `Wind: ${p.wind || "—"} (traditional ML)`;
      }
    }
  }

  async function fetchPerf() {
    try {
      const res = await fetch("/api/model-performance");
      if (!res.ok) throw new Error("HTTP " + res.status);
      renderPerf(await res.json());
    } catch (e) {
      perfShowMessage("Couldn't load benchmark results. Run <code>python test.py</code> " +
                      "to generate them.");
    }
  }

  /* ---------------- boot ---------------- */
  buildKpis();
  fetchPerf();
  // surface unexpected JS errors as visible toasts instead of silent failures
  window.addEventListener("error", ev => {
    if (ev && ev.message) toast("⚠️ " + ev.message, "err");
  });
  window.addEventListener("resize", () => { if (lastState) render(lastState); });
  fetchState();
  setInterval(fetchState, POLL_MS);
  requestAnimationFrame(pulseLatest);
})();
