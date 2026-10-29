/* Green Grid - application logic.
   The API is stateless (Vercel): the grid state lives here and is sent with every request.
   Sections: helpers | API | KPIs & dock | overview | flow diagram | views | model performance | controls | boot */
(() => {
  "use strict";

  const $ = id => document.getElementById(id);
  const f1 = n => Number(n).toFixed(1);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  const reducedMotion = () => document.documentElement.classList.contains("reduce-motion") ||
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const TYPE = {
    1: { color: "#F2C14E", icon: "i-sun" }, 2: { color: "#7FD1FF", icon: "i-wind" }, 3: { color: "#27E0CF", icon: "i-drop" },
    4: { color: "#A78BFA", icon: "i-battery" }, 5: { color: "#F5B544", icon: "i-flame" }, 6: { color: "#9AA5B1", icon: "i-flame" },
    7: { color: "#EF8A5B", icon: "i-flame" },
  };
  const typeOf = s => TYPE[s.type_id] || { color: "#9AA5B1", icon: "i-bolt" };
  const icon = (id, cls = "i") => `<svg class="${cls}"><use href="#${id}"/></svg>`;

  let lastState = null, prod = null, perfData = null, simLog = [];
  let failed = 0, activeView = "overview";
  const VIEWS = ["overview", "monitoring", "sources", "predictions", "models", "settings"];

  /* =================== helpers =================== */
  function toast(msg, kind = "ok") {
    const el = document.createElement("div");
    el.className = "toast" + (kind === "ok" ? "" : " " + kind);
    el.textContent = msg;
    $("toastWrap").appendChild(el);
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 320); }, 3200);
  }
  function setLive(on) {
    $("liveBadge").classList.toggle("off", !on);
    $("liveLabel").textContent = on ? "API connected" : "API offline";
  }
  function animateNumber(el, to, fmt) {
    const from = parseFloat(el.dataset.v);
    el.dataset.v = to;
    if (!Number.isFinite(from) || reducedMotion() || from === to) { el.textContent = fmt(to); return; }
    const t0 = performance.now(), dur = 450;
    const step = now => {
      const k = clamp((now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(from + (to - from) * e);
      if (k < 1 && el.dataset.v === String(to)) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* =================== API =================== */
  const stateOf = s => s && ({ sources: s.sources, demand_mw: s.demand_mw, index: s.index, weather: s.weather, history: s.history });

  function accept(st) {
    lastState = st;
    logSim(st);
    render(st);
  }

  async function fetchState() {
    try {
      const res = await fetch(lastState ? "/api/health" : "/api/state");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const body = await res.json();
      if (!lastState) { document.body.classList.remove("loading"); accept(body); }
      failed = 0; setLive(true); $("errBanner").hidden = true;
    } catch (_) {
      failed++; setLive(false);
      document.body.classList.remove("loading");
      if (!lastState || failed >= 2) $("errBanner").hidden = false;
    }
  }

  async function postJSON(url, body, silent = false) {
    try {
      const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...(body || {}), state: stateOf(lastState) }) });
      if (!res.ok) throw new Error((await res.text()).slice(0, 140));
      accept(await res.json());
    } catch (e) {
      if (!silent) toast("Request failed: " + e.message, "err");
      throw e;
    }
  }
  const act = (url, body, silent) => postJSON(url, body, silent).catch(() => {});

  /* client-side series of forecast-driven output (taken from the API responses) */
  function logSim(st) {
    const a = GGScene.aggregate(st), entry = { idx: st.index, t: st.time.slice(5), solar_mw: a.solar.avail, wind_mw: a.wind.avail };
    const last = simLog[simLog.length - 1];
    if (last && last.idx === entry.idx) simLog[simLog.length - 1] = entry; else simLog.push(entry);
    if (simLog.length > 500) simLog.shift();
  }

  /* =================== status helpers =================== */
  function statusOf(s) {
    const u = s.used || 0;
    if (s.storage) return u < -0.01 ? ["Charging", "sto"] : u > 0.01 ? ["Discharging", "sto"] : s.soc <= 0.001 ? ["Empty", ""] : ["Idle", ""];
    if (s.renewable) return u > 0.01 ? ["Generating", "on"] : ["No output", ""];
    return u > 0.01 ? ["Dispatched", "fos"] : ["Standby", ""];
  }
  const catOf = s => (s.storage ? "Storage" : s.renewable ? "Renewable" : "Fossil");

  /* =================== KPIs =================== */
  const generation = st => (st.sources || []).reduce((a, s) => a + Math.max(0, s.used || 0), 0);
  const KPI = [
    { id: "gen", label: "Total generation", icon: "i-bolt", unit: "MW", dec: 1,
      val: st => generation(st), title: "Sum of every source currently dispatching, including battery discharge." },
    { id: "dem", label: "Current demand", icon: "i-demand", unit: "MW", dec: 1, val: st => st.dispatch.demand,
      title: "Demand set for the simulation." },
    { id: "cov", label: "Supply coverage", icon: "i-shield", unit: "%", dec: 1,
      val: st => (st.dispatch.demand > 0 ? (st.dispatch.demand - st.dispatch.unmet) / st.dispatch.demand * 100 : null),
      cls: st => (st.dispatch.unmet > 0.01 ? "bad" : ""), title: "Share of demand that is met. This is a coverage measure, not a grid-stability rating." },
    { id: "ren", label: "Renewable contribution", icon: "i-leaf", unit: "%", dec: 1, val: st => st.dispatch.renewable_pct,
      title: "Share of demand served by renewable sources and battery discharge." },
    { id: "eff", label: "Clean energy utilisation", icon: "i-gauge", unit: "%", dec: 1,
      val: st => (st.dispatch.renewable_generated > 0.001 ? (1 - st.dispatch.curtailed / st.dispatch.renewable_generated) * 100 : null),
      cls: st => (st.dispatch.curtailed > 0.5 ? "warn" : ""), title: "Share of renewable output that is not curtailed. Unavailable when nothing renewable is generating." },
  ];

  function buildKpis() {
    $("kpis").innerHTML = KPI.map(k =>
      `<div class="kpi" id="kpi-${k.id}" title="${esc(k.title)}"><span class="k-ico">${icon(k.icon)}</span>` +
      `<span class="k-label">${k.label}</span><span class="k-value"><span data-n>&ndash;</span><small>${k.unit}</small></span></div>`).join("");
  }
  function renderKpis(st) {
    KPI.forEach(k => {
      const box = $("kpi-" + k.id), n = box.querySelector("[data-n]"), v = k.val(st);
      box.className = "kpi " + (k.cls ? k.cls(st) : "");
      if (v == null || !Number.isFinite(v)) { n.textContent = "n/a"; delete n.dataset.v; box.querySelector("small").hidden = true; }
      else { box.querySelector("small").hidden = false; animateNumber(n, v, x => Number(x).toFixed(k.dec)); }
    });
  }

  /* =================== dock / header =================== */
  function renderDock(st) {
    $("timeChip").textContent = `${st.time} (${st.weekday.slice(0, 3)})`;
    const inp = $("demandSet");
    if (document.activeElement !== inp) inp.value = Number(st.demand_mw).toFixed(1).replace(/\.0$/, "");
  }

  /* =================== overview =================== */
  const RING_C = 2 * Math.PI * 84;
  function renderRing(st) {
    const d = st.dispatch, dem = Math.max(d.demand, 1e-9);
    const ren = clamp(d.renewable_used / dem), fos = clamp(d.fossil_used / dem, 0, 1 - ren), un = clamp(1 - ren - fos);
    const seg = (id, start, len) => {
      const el = $(id);
      el.style.strokeDasharray = `${Math.max(0, len * RING_C - (len > 0 && len < 1 ? 2 : 0))} ${RING_C}`;
      el.style.strokeDashoffset = -start * RING_C;
    };
    seg("ringRen", 0, ren); seg("ringFos", ren, fos); seg("ringUnmet", ren + fos, un > 0.001 ? un : 0);
    $("ringPct").textContent = f1(d.renewable_pct) + "%";
    $("ringLegend").innerHTML =
      `<li><i style="background:#27E0CF"></i>Renewable ${f1(d.renewable_used)} MW</li>` +
      `<li><i style="background:#F5B544"></i>Fossil ${f1(d.fossil_used)} MW</li>` +
      (d.unmet > 0.01 ? `<li><i style="background:#EF6464"></i>Unmet ${f1(d.unmet)} MW</li>` : "") +
      (d.curtailed > 0.01 ? `<li><i style="background:#6E9BFF"></i>Curtailed ${f1(d.curtailed)} MW</li>` : "");
    $("allocSub").textContent = "this hour";
    $("allocStats").innerHTML = [["Curtailed", f1(d.curtailed), "MW"], ["Unmet", f1(d.unmet), "MW"], ["CO\u2082", f1(d.co2), "t/h"]]
      .map(([k, v, u]) => `<div><span>${k}</span><b>${v}<small>${u}</small></b></div>`).join("");
    $("allocBars").innerHTML = (st.sources || []).map(s => {
      const w = s.capacity > 0 ? clamp(Math.abs(s.used) / s.capacity) * 100 : 0;
      return `<div class="bar-row" style="--c:${typeOf(s).color}"><div class="bar-top"><span>${esc(s.name)}</span>` +
        `<span>${f1(Math.abs(s.used))}<small>/${f1(s.capacity)} MW</small></span></div><div class="tickbar"><i style="width:${w}%"></i></div></div>`;
    }).join("");
  }

  function renderInsight(st) {
    const d = st.dispatch, box = $("insight"), batt = (st.sources || []).find(s => s.storage);
    let cls = "", txt;
    if (d.unmet > 0.01) { cls = "bad"; txt = `Demand exceeds available supply by ${f1(d.unmet)} MW. Add capacity or lower demand.`; }
    else if (d.curtailed > 0.5) { cls = "warn"; txt = `${f1(d.curtailed)} MW of renewable output is curtailed. ${batt ? "Storage is full." : "Adding battery storage would capture it."}`; }
    else if (d.fossil_used > 0.01) { txt = `Fossil units cover ${f1(d.fossil_used)} MW (${f1(d.fossil_used / Math.max(d.demand, 1e-9) * 100)}% of demand). Renewable output is limited by current weather and installed capacity.`; }
    else if (d.battery_discharged > 0.01) txt = `Battery is discharging ${f1(d.battery_discharged)} MW. Demand is met entirely by clean sources.`;
    else txt = "Demand is met entirely by renewable sources.";
    box.className = "insight " + cls;
    $("insightText").textContent = txt;
  }

  function renderSrcList(st) {
    $("srcList").innerHTML = (st.sources || []).map(s => {
      const [label, cls] = statusOf(s);
      return `<li style="--c:${typeOf(s).color}"><i class="dot"></i><span class="nm">${esc(s.name)}</span>` +
        `<span class="badge ${cls}">${label}</span><span class="sub">${f1(Math.abs(s.used))} MW of ${f1(s.capacity)} MW</span></li>`;
    }).join("");
  }

  /* =================== energy-flow diagram =================== */
  function Flow(el) {
    let sig = "", W = 340, H = 240, rows = [];
    function build(st) {
      const srcs = st.sources || [], n = srcs.length;
      H = Math.max(230, n * 54 + 16);
      const y = i => 8 + i * 54 + 22, hubY = H / 2;
      rows = srcs.map(s => s.id);
      let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Energy flow from sources to demand">`;
      srcs.forEach((s, i) => {
        const c = typeOf(s).color, d = `M128,${y(i)} C166,${y(i)} 166,${hubY} 190,${hubY}`;
        svg += `<g class="lk" data-id="${s.id}" style="--c:${c}"><path class="link-bg" d="${d}"/><path class="link-fg" d="${d}"/></g>`;
      });
      svg += `<g class="lk" data-id="out" style="--c:#27E0CF"><path class="link-bg" d="M236,${hubY} L ${W - 88},${hubY}"/><path class="link-fg" d="M236,${hubY} L ${W - 88},${hubY}"/></g>`;
      srcs.forEach((s, i) => {
        svg += `<g class="nd" data-id="${s.id}" style="--c:${typeOf(s).color}"><rect x="0" y="${y(i) - 22}" width="128" height="44" rx="11"/>` +
          `<text class="n1" x="12" y="${y(i) - 3}">${esc(s.name)}</text><text class="n2" x="12" y="${y(i) + 13}" data-t></text></g>`;
      });
      svg += `<g class="hub"><circle cx="213" cy="${hubY}" r="22"/><text x="213" y="${hubY + 4}">GRID</text></g>`;
      svg += `<g class="nd" data-id="dem" style="--c:#27E0CF"><rect x="${W - 88}" y="${hubY - 34}" width="88" height="68" rx="13"/>` +
        `<text class="n1" x="${W - 78}" y="${hubY - 9}">Demand</text><text class="n2" x="${W - 78}" y="${hubY + 7}" data-t></text>` +
        `<text class="n2" x="${W - 78}" y="${hubY + 22}" data-t2></text></g></svg>`;
      el.innerHTML = svg;
    }
    function update(st) {
      const s2 = (st.sources || []).map(s => s.id).join(",");
      if (s2 !== sig) { build(st); sig = s2; }
      const maxUse = Math.max(1, ...(st.sources || []).map(s => Math.abs(s.used)));
      (st.sources || []).forEach(s => {
        const on = Math.abs(s.used) > 0.01, lk = el.querySelector(`.lk[data-id="${s.id}"] .link-fg`), nd = el.querySelector(`.nd[data-id="${s.id}"]`);
        if (!lk) return;
        lk.classList.toggle("on", on); lk.classList.toggle("rev", s.used < -0.01);
        lk.style.strokeWidth = (1.6 + 4.2 * Math.abs(s.used) / maxUse).toFixed(2);
        nd.classList.toggle("act", on);
        nd.querySelector("[data-t]").textContent = s.storage
          ? `${s.used < -0.01 ? "charging " : ""}${f1(Math.abs(s.used))} MW · ${f1(s.soc * 100)}% charge`
          : `${f1(s.used)} / ${f1(s.available)} MW avail.`;
      });
      const d = st.dispatch, served = d.renewable_used + d.fossil_used;
      const out = el.querySelector('.lk[data-id="out"] .link-fg'); out.classList.toggle("on", served > 0.01);
      out.style.strokeWidth = (2 + 4 * clamp(served / Math.max(d.demand, 1e-9))).toFixed(2);
      const dn = el.querySelector('.nd[data-id="dem"]'); dn.classList.toggle("act", served > 0.01);
      dn.querySelector("[data-t]").textContent = `${f1(d.demand)} MW`;
      dn.querySelector("[data-t2]").textContent = `${Math.round(d.demand > 0 ? served / d.demand * 100 : 0)}% served`;
    }
    return { update };
  }
  const flowOv = Flow($("flowOv")), flowSrc = Flow($("flowSrc"));

  /* =================== other views =================== */
  function renderTable(st) {
    $("srcBody").innerHTML = (st.sources || []).map(s => {
      const c = typeOf(s).color, u = s.used || 0, w = s.capacity > 0 ? clamp(Math.abs(u) / s.capacity) * 100 : 0;
      const used = u < -0.01 ? `<td class="r dis neg">&minus;${f1(Math.abs(u))} (charging)</td>` : u > 0.01 ? `<td class="r dis">${f1(u)}</td>` : `<td class="r zero">0.0</td>`;
      const [label, cls] = statusOf(s);
      return `<tr><td><span class="src-nm"><i class="dot" style="--c:${c}"></i>${esc(s.name)}</span></td><td><span class="badge ${cls}">${catOf(s)}</span></td>` +
        `<td class="r">${f1(s.capacity)}</td><td class="r">${f1(s.available)}</td>${used}` +
        `<td><div class="mini" style="--c:${c}"><i style="width:${w}%"></i></div></td><td class="r">${s.storage ? f1(s.soc * 100) + "%" : "&ndash;"}</td>` +
        `<td class="r"><button class="row-x" data-rm="${s.id}" type="button" aria-label="Remove ${esc(s.name)}" title="Remove ${esc(s.name)}">${icon("i-trash")}</button></td></tr>`;
    }).join("");
    const cap = (st.sources || []).reduce((a, s) => a + (s.capacity || 0), 0);
    $("tableSub").textContent = `${st.sources.length} sources, ${f1(cap)} MW installed`;
  }

  function renderSourceCards(st) {
    $("srcCards").innerHTML = (st.sources || []).map(s => {
      const t = typeOf(s), [label, cls] = statusOf(s), u = s.used || 0;
      const util = s.capacity > 0 ? Math.abs(u) / s.capacity * 100 : 0, avail = s.capacity > 0 ? s.available / s.capacity * 100 : 0;
      return `<article class="sc ${Math.abs(u) > 0.01 ? "live" : ""}" style="--c:${t.color}"><div class="sc-h"><span class="sc-ico">${icon(t.icon)}</span>` +
        `<span class="nm">${esc(s.name)}</span><span class="badge ${cls}">${label}</span></div>` +
        `<div class="sc-out">${u < -0.01 ? "&minus;" : ""}${f1(Math.abs(u))}<small>MW dispatched</small></div>` +
        `<div class="tickbar"><i style="width:${clamp(util / 100) * 100}%"></i></div>` +
        `<div class="sc-meta"><span>Capacity<b>${f1(s.capacity)} MW</b></span><span>Available<b>${f1(s.available)} MW</b></span><span>${s.storage ? "Charge" : "Utilisation"}<b>${s.storage ? f1(s.soc * 100) : f1(util)}%</b></span></div>` +
        `<div class="sc-f"><span>${catOf(s)}${s.storage || !s.renewable ? "" : ` · availability ${f1(avail)}%`}${!s.renewable ? ` · ${s.emission} t CO₂/MWh` : ""}</span>` +
        `<button class="row-x" data-rm="${s.id}" type="button" aria-label="Remove ${esc(s.name)}" title="Remove ${esc(s.name)}">${icon("i-trash")}</button></div></article>`;
    }).join("");
  }

  function renderWeather(st) {
    const w = st.weather || {};
    $("wxList").innerHTML = [["Temperature", w["Temperature"], "°C"], ["Humidity", w["Humidity"], "%"], ["Wind speed", w["Wind Speed"], "m/s"],
      ["General diffuse flows", w["general diffuse flows"], "W/m²"], ["Diffuse flows", w["diffuse flows"], "W/m²"]]
      .map(([k, v, u]) => `<div><dt>${k}</dt><dd>${v == null ? "&ndash;" : Number(v).toFixed(1)} ${u}</dd></div>`).join("");
  }

  function renderForecast(st) {
    const a = GGScene.aggregate(st), names = (prod && prod.names) || {};
    const card = (id, title, g, ico, color, model) => {
      $(id).style.setProperty("--c", color);
      $(id).innerHTML = `<header class="card-h"><h2>${title} forecast</h2><span class="muted">${esc(model || "model")}</span></header>` +
        (g.count
          ? `<div class="eyebrow">Capacity factor this hour</div><div class="big">${f1(g.avail / g.cap * 100)}<small>%</small></div>` +
            `<div class="kv"><div>Installed capacity<b>${f1(g.cap)} MW</b></div><div>Forecast output<b>${f1(g.avail)} MW</b></div><div>Dispatched<b>${f1(g.used)} MW</b></div></div>`
          : `<p class="fine">No ${title.toLowerCase()} source is installed, so there is no output to forecast. Add one from the dock.</p>`);
    };
    card("fcSolar", "Solar", a.solar, "i-sun", TYPE[1].color, names.solar);
    card("fcWind", "Wind", a.wind, "i-wind", TYPE[2].color, names.wind);
    renderWeather(st);
    GGCharts.lines($("cvFc"), $("tipFc"), simLog, [
      { key: "solar_mw", color: "#F2C14E", label: "Solar" }, { key: "wind_mw", color: "#7FD1FF", label: "Wind" }], { unit: " MW", area: false });
  }

  function renderMonitoring(st) {
    const h = st.history || [];
    GGCharts.dispatchBars($("cvHistMon"), $("tipHistMon"), h, { max: 36 });
    GGCharts.lines($("cvShare"), $("tipShare"), h.slice(-72), [{ key: "renewable_pct", color: "#27E0CF", label: "Renewable" }], { yMax: 100, unit: "%" });
    GGCharts.lines($("cvCo2"), $("tipCo2"), h.slice(-72), [{ key: "co2", color: "#F5B544", label: "CO₂" }], { unit: " t/h", fmt: x => Number(x).toFixed(2) });
    GGCharts.lines($("cvSoc"), $("tipSoc"), h.slice(-72), [{ key: "battery_soc", color: "#A78BFA", label: "Charge" }], { yMax: 100, unit: "%" });
    renderTable(st);
  }

  /* =================== model performance =================== */
  const num = (v, nd) => (v == null || Number.isNaN(v) ? "&ndash;" : Number(v).toFixed(nd));
  const pct = v => (v == null || Number.isNaN(v) ? "&ndash;" : Number(v).toFixed(2) + "%");
  const day = s => (s ? String(s).slice(0, 10) : "&ndash;");

  function perfMessage(msg) {
    const e = $("perfEmpty"); e.innerHTML = esc(msg); e.hidden = false; $("perfBody").hidden = true;
  }
  function renderPerf(d) {
    perfData = d;
    if (!d || !d.available) { perfMessage((d && d.message) || "Benchmark results are not available yet. Run the benchmark, then export the runtime files."); }
    const p = (d && d.production) || {};
    prod = { names: { demand: p.demand, solar: p.solar, wind: p.wind } };
    if (p.demand) $("footModels").textContent = `Models: demand ${p.demand}, solar ${p.solar || "n/a"}, wind ${p.wind || "n/a"}`;
    if (lastState) renderForecast(lastState);
    if (!d || !d.available) return;
    const rows = Array.isArray(d.models) ? d.models : [];
    if (!rows.length) { perfMessage("No benchmark rows were saved."); return; }
    $("perfEmpty").hidden = true; $("perfBody").hidden = false;

    const tolTxt = d.tolerance != null ? `±${(d.tolerance * 100).toFixed(0)}%` : "tolerance";
    $("perfTolHead").textContent = `Accuracy (${tolTxt})`;
    $("lbSub").textContent = `Share of hours forecast within ${tolTxt} of actual demand. Higher is better.`;
    $("perfSub").textContent = `${rows.length} models. ${(d.ranking_rule || "Ranked by tolerance accuracy").replace(/\.$/, "")}.`;

    $("prodSub").textContent = p.selected_on || "";
    $("prodGrid").innerHTML = [["Demand", p.demand], ["Solar", p.solar], ["Wind", p.wind]].map(([k, v]) => `<div><span>${k}</span><b>${esc(v || "n/a")}</b></div>`).join("");
    const bi = d.best_individual, be = d.best_ensemble;
    $("bestIndiv").textContent = bi ? bi.model : "n/a"; $("bestIndivWhy").textContent = d.best_individual_reason || "";
    $("bestEns").textContent = be ? be.model : "n/a"; $("bestEnsWhy").textContent = d.best_ensemble_reason || "";

    const m = bi || rows[0];
    $("metricCards").innerHTML = [
      ["Accuracy " + tolTxt, pct(m.tolerance_accuracy), "best individual"], ["RMSE", num(m.rmse, 1), "kW"], ["MAE", num(m.mae, 1), "kW"],
      ["R²", num(m.r2, 4), "variance explained"], ["F1*", num(m.derived_f1, 3), "derived, demand bands"]]
      .map(([k, v, s]) => `<div class="metric"><span>${k}</span><b>${v}</b><small>${s}</small></div>`).join("");

    const best = new Set([bi && bi.model, be && be.model]);
    const label = r => (r.rank ? `#${r.rank} ` : "") + r.model;
    GGCharts.hbars($("hbAcc"), rows.filter(r => Number.isFinite(r.tolerance_accuracy)).map(r => ({ label: label(r), value: r.tolerance_accuracy, text: pct(r.tolerance_accuracy), best: best.has(r.model) })), { max: 100 });
    GGCharts.hbars($("hbRmse"), rows.filter(r => Number.isFinite(r.rmse)).sort((a, b) => a.rmse - b.rmse).map(r => ({ label: label(r), value: r.rmse, text: num(r.rmse, 0), best: best.has(r.model) })));

    $("perfRows").innerHTML = rows.map(r =>
      `<tr class="${best.has(r.model) ? "is-best" : ""}"><td><span class="rk">${r.rank ? "#" + r.rank : "&ndash;"}</span><b>${esc(r.model)}</b></td><td>${esc(r.type)}</td>` +
      `<td class="r">${pct(r.tolerance_accuracy)}</td><td class="r">${num(r.derived_precision, 3)}</td><td class="r">${num(r.derived_recall, 3)}</td><td class="r">${num(r.derived_f1, 3)}</td>` +
      `<td class="r">${num(r.mae, 1)}</td><td class="r">${num(r.rmse, 1)}</td><td class="r">${num(r.r2, 4)}</td></tr>`).join("");
    $("perfNote").textContent = (d.derived_metrics_note ? "*" + d.derived_metrics_note + " " : "*Derived metrics. ") +
      "Thresholds come from the training data only. They measure band placement, not kilowatt accuracy.";

    const ds = d.dataset || {}, sp = d.split || {}, th = d.demand_class_thresholds || {};
    const info = [
      ["Dataset", ds.file], ["Hourly rows", ds.hourly_rows != null ? Number(ds.hourly_rows).toLocaleString() : null],
      ["Target", ds.target ? `${ds.target} (${ds.target_unit || ""})` : null],
      ["Split", sp.fractions ? sp.fractions.map(x => Math.round(x * 100) + "%").join(" / ") + (sp.shuffled === false ? ", chronological" : "") : null],
      ["Train", sp.train_period ? `${day(sp.train_period[0])} to ${day(sp.train_period[1])} (${sp.n_train} h)` : null],
      ["Validation", sp.val_period ? `${day(sp.val_period[0])} to ${day(sp.val_period[1])} (${sp.n_val} h)` : null],
      ["Test", sp.test_period ? `${day(sp.test_period[0])} to ${day(sp.test_period[1])} (${sp.n_test} h)` : null],
      ["Demand bands", th.low_medium != null ? `Low < ${Number(th.low_medium).toFixed(0)} kW <= Medium < ${Number(th.medium_high).toFixed(0)} kW <= High` : null],
      ["Ranking rule", d.ranking_rule], ["Approach", d.approach],
      ["Models trained", p.trained_at_utc ? day(p.trained_at_utc) : null], ["Benchmark run", d.generated_at_utc ? `${day(d.generated_at_utc)} (${d.mode || "quick"} mode)` : null],
    ].filter(x => x[1]);
    $("trainInfo").innerHTML = info.map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("");
  }

  async function fetchPerf() {
    try {
      const res = await fetch("/api/model-performance");
      if (!res.ok) throw new Error("HTTP " + res.status);
      renderPerf(await res.json());
    } catch (_) { perfMessage("Benchmark results could not be loaded."); }
  }

  /* =================== render / router =================== */
  function render(st) {
    renderKpis(st); renderDock(st);
    renderView(st);
  }
  function renderView(st) {
    st = st || lastState; if (!st) return;
    if (activeView === "overview") {
      GGScene.update(st); renderInsight(st); renderRing(st); renderSrcList(st); flowOv.update(st);
      GGCharts.dispatchBars($("cvHistOv"), $("tipHistOv"), st.history || [], { height: 300, max: 18 });
    } else if (activeView === "monitoring") renderMonitoring(st);
    else if (activeView === "sources") { renderSourceCards(st); flowSrc.update(st); }
    else if (activeView === "predictions") renderForecast(st);
  }

  function route(userNav) {
    const h = (location.hash || "#overview").slice(1);
    activeView = VIEWS.includes(h) ? h : "overview";
    VIEWS.forEach(v => { $("view-" + v).hidden = v !== activeView; });
    document.querySelectorAll(".nav a").forEach(a => {
      if (a.dataset.view === activeView) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    const v = $("view-" + activeView); v.classList.remove("enter"); void v.offsetWidth; v.classList.add("enter");
    $("nav").classList.remove("open"); $("navToggle").setAttribute("aria-expanded", "false");
    if (userNav) window.scrollTo({ top: 0 });
    renderView();
  }
  window.addEventListener("hashchange", () => route(true));
  window.addEventListener("gg-resize", () => { if (lastState) renderView(); });
  $("navToggle").addEventListener("click", () => {
    const open = $("nav").classList.toggle("open"); $("navToggle").setAttribute("aria-expanded", String(open));
  });

  /* =================== scene tooltip =================== */
  const GROUP_NAME = { solar: "Solar", wind: "Wind", hydro: "Hydro", battery: "Battery storage", fossil: "Fossil plants", demand: "City demand", grid: "Substation" };
  function sceneTip(group, e) {
    if (!lastState) return;
    const st = lastState, a = GGScene.aggregate(st), d = st.dispatch, tip = $("sceneTip");
    const rows = [];
    if (a[group]) {
      const g = a[group];
      if (!g.count) rows.push(["Status", "Not installed"]);
      else {
        rows.push(["Units", g.count], ["Capacity", f1(g.cap) + " MW"], ["Available", f1(g.avail) + " MW"], ["Dispatched", f1(g.used) + " MW"]);
        rows.push(group === "battery" ? ["Charge", f1(g.soc * 100) + "%"] : ["Utilisation", f1(g.cap ? g.used / g.cap * 100 : 0) + "%"]);
      }
    } else if (group === "demand") {
      rows.push(["Demand", f1(d.demand) + " MW"], ["Served", f1(d.renewable_used + d.fossil_used) + " MW"], ["Unmet", f1(d.unmet) + " MW"]);
    } else {
      rows.push(["Dispatched", f1(generation(st)) + " MW"], ["CO₂", f1(d.co2) + " t/h"], ["Curtailed", f1(d.curtailed) + " MW"]);
    }
    tip.innerHTML = `<b>${GROUP_NAME[group] || group}</b>` + rows.map(r => `<div class="row"><span>${r[0]}</span><span>${r[1]}</span></div>`).join("");
    const box = tip.parentElement.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = clamp(e.clientX - box.left + 16, 8, box.width - tip.offsetWidth - 8) + "px";
    tip.style.top = clamp(e.clientY - box.top + 16, 8, box.height - tip.offsetHeight - 8) + "px";
  }

  /* =================== controls =================== */
  const showModal = id => { $(id).hidden = false; const f = $(id).querySelector("input,select,button.btn"); if (f) f.focus(); };
  const hideModal = id => { $(id).hidden = true; };
  document.querySelectorAll(".modal").forEach(m => m.addEventListener("click", e => { if (e.target === m) m.hidden = true; }));
  document.querySelectorAll("[data-close]").forEach(b => b.addEventListener("click", () => hideModal(b.dataset.close)));
  document.addEventListener("keydown", e => { if (e.key === "Escape") document.querySelectorAll(".modal").forEach(m => { m.hidden = true; }); });

  const WX = [["wTemp", "Temperature"], ["wHum", "Humidity"], ["wWind", "Wind Speed"], ["wGenDiff", "general diffuse flows"], ["wDiff", "diffuse flows"]];
  function openWeather() {
    if (!lastState) return;
    const w = lastState.weather || {};
    WX.forEach(([id, k]) => { $(id).value = ""; $(id).placeholder = w[k] == null ? "" : Number(w[k]).toFixed(2); });
    showModal("weatherModal");
  }
  $("btnWeather").addEventListener("click", openWeather);
  $("btnWeather2").addEventListener("click", openWeather);
  $("weatherForm").addEventListener("submit", async e => {
    e.preventDefault();
    const body = {};
    WX.forEach(([id, key]) => {
      const v = $(id).value.trim(); if (v === "") return;
      const n = parseFloat(v); if (!isNaN(n)) body[key] = n; else toast(`"${key}" is not a number and was ignored`, "warn");
    });
    if (!Object.keys(body).length) { toast("All fields blank, nothing to update", "warn"); return; }
    try { await postJSON("/api/weather", body); hideModal("weatherModal"); toast("Weather updated"); } catch (_) {}
  });

  const openAdd = () => { $("srcCap").value = "50"; showModal("addSourceModal"); };
  $("btnAddSource").addEventListener("click", openAdd);
  $("btnAddSource2").addEventListener("click", openAdd);
  $("addSourceForm").addEventListener("submit", async e => {
    e.preventDefault();
    const type_id = parseInt($("srcType").value, 10), capacity = parseFloat($("srcCap").value);
    if (!Number.isFinite(capacity) || capacity <= 0) { toast("Enter a valid capacity in MW", "warn"); return; }
    try { await postJSON("/api/sources", { type_id, capacity }); hideModal("addSourceModal"); toast(`Added ${capacity} MW source`); } catch (_) {}
  });
  document.addEventListener("click", async e => {
    const btn = e.target.closest("[data-rm]"); if (!btn) return;
    btn.disabled = true;
    try { await postJSON("/api/sources/remove", { id: parseInt(btn.dataset.rm, 10) }); toast("Source removed"); } catch (_) { btn.disabled = false; }
  });

  $("btnFwd").addEventListener("click", () => act("/api/time", { hours: 1 }));
  $("btnBack").addEventListener("click", () => act("/api/time", { hours: -1 }));
  $("btnSetTime").addEventListener("click", () => {
    const v = $("timePick").value;
    if (!v) { toast("Pick a date and time first", "warn"); return; }
    act("/api/time", { datetime: v + ":00" });
  });
  $("btnDemandUp").addEventListener("click", () => act("/api/demand", { delta: 10 }));
  $("btnDemandDown").addEventListener("click", () => act("/api/demand", { delta: -10 }));
  let demandPending = null;
  const commitDemand = async () => {
    const v = parseFloat($("demandSet").value);
    if (isNaN(v) || v < 0) { toast("Enter a valid demand in MW", "warn"); return; }
    if ((lastState && v === lastState.demand_mw) || v === demandPending) return;   // Enter + blur fire twice
    demandPending = v;
    await act("/api/demand", { set: v });
    demandPending = null;
  };
  $("demandSet").addEventListener("keydown", e => { if (e.key === "Enter") { commitDemand(); e.target.blur(); } });
  $("demandSet").addEventListener("change", commitDemand);

  let playTimer = null, busy = false;
  function setPlayUi(on) {
    const b = $("btnPlay");
    b.classList.toggle("active", on); b.setAttribute("aria-pressed", String(on));
    b.querySelector("use").setAttribute("href", on ? "#i-pause" : "#i-play");
    b.querySelector("span").textContent = on ? "Pause" : "Play";
  }
  $("btnPlay").addEventListener("click", () => {
    if (playTimer) { clearInterval(playTimer); playTimer = null; setPlayUi(false); return; }
    setPlayUi(true);
    playTimer = setInterval(async () => {
      if (busy) return;
      busy = true; await act("/api/time", { hours: 1 }, true); busy = false;
    }, 1000);
  });

  $("btnSave").addEventListener("click", () => {
    try { localStorage.setItem("greenGridState", JSON.stringify(stateOf(lastState))); toast("State saved in this browser"); }
    catch (_) { toast("Browser storage is unavailable", "err"); }
  });
  $("btnLoad").addEventListener("click", async () => {
    let saved = null; try { saved = JSON.parse(localStorage.getItem("greenGridState")); } catch (_) {}
    if (!saved) { toast("Nothing saved in this browser yet", "err"); return; }
    try { await postJSON("/api/import", saved); toast("Saved state loaded"); } catch (_) {}
  });
  $("btnExport").addEventListener("click", () => {
    const doc = { sources: lastState.sources, demand_mw: lastState.demand_mw, current_idx: lastState.index, weather: lastState.weather };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" }));
    a.download = "grid_export.json"; a.click(); URL.revokeObjectURL(a.href);
    toast("Exported grid_export.json");
  });
  $("btnImport").addEventListener("click", () => $("importFile").click());
  $("importFile").addEventListener("change", async e => {
    const file = e.target.files[0]; e.target.value = ""; if (!file) return;
    try { await postJSON("/api/import", JSON.parse(await file.text())); toast(`Imported ${file.name}`); }
    catch (err) { toast(err.name === "SyntaxError" ? "File is not valid JSON" : "Import failed: " + err.message, "err"); }
  });
  $("btnReset").addEventListener("click", async () => {
    try {
      const res = await fetch("/api/state"); if (!res.ok) throw new Error("HTTP " + res.status);
      simLog = []; accept(await res.json()); toast("Grid reset to default");
    } catch (e) { toast("Reset failed: " + e.message, "err"); }
  });

  /* display option */
  function applyMotion(on) { document.documentElement.classList.toggle("reduce-motion", on); $("optMotion").checked = on; }
  $("optMotion").addEventListener("change", e => {
    applyMotion(e.target.checked);
    try { localStorage.setItem("greenGridReduceMotion", e.target.checked ? "1" : "0"); } catch (_) {}
    if (lastState) renderView();
  });
  try { applyMotion(localStorage.getItem("greenGridReduceMotion") === "1"); } catch (_) {}

  $("btnRetry").addEventListener("click", () => { document.body.classList.add("loading"); fetchState(); fetchPerf(); });

  /* =================== boot =================== */
  buildKpis();
  GGScene.init($("scene"), {
    onHover: sceneTip, onLeave: () => { $("sceneTip").hidden = true; },
  });
  window.addEventListener("error", ev => { if (ev && ev.message) toast(ev.message, "err"); });
  route(false);
  fetchState();
  fetchPerf();
  setInterval(fetchState, 10000);
})();
