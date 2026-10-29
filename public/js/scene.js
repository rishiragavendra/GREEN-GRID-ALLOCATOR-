/* Isometric city scene (SVG).
   Purely illustrative geometry: positions, building counts and shapes are decorative.
   What is LIVE: glow/utilisation of each source group, turbine spin speed, battery fill,
   power-line activity, and how many windows are lit (demand vs. installed capacity). */
(function () {
  "use strict";

  const S = 38, C = Math.cos(Math.PI / 6), SN = 0.5, OX = 500, OY = 78;
  const P = (x, y, z = 0) => [OX + (x - y) * S * C, OY + (x + y) * S * SN - z * S];
  const pt = p => p[0].toFixed(1) + "," + p[1].toFixed(1);
  const poly = (pts, cls, extra = "") => `<polygon class="${cls}" points="${pts.map(pt).join(" ")}" ${extra}/>`;

  const GROUPS = { solar: [1], wind: [2], hydro: [3], battery: [4], fossil: [5, 6, 7] };
  let root = null, hooks = {}, signature = "", winCount = 0;

  /* ---------- primitives ---------- */
  function faceX(x, y, w, d, h, cls) {      // face at y+d, spans x
    const o = P(x, y + d, 0);
    return `<g transform="matrix(${S * C} ${S * SN} 0 ${-S} ${o[0]} ${o[1]})"><rect class="${cls}" width="${w}" height="${h}"/>`;
  }
  function faceY(x, y, w, d, h, cls) {      // face at x+w, spans y
    const o = P(x + w, y, 0);
    return `<g transform="matrix(${-S * C} ${S * SN} 0 ${-S} ${o[0]} ${o[1]})"><rect class="${cls}" width="${d}" height="${h}"/>`;
  }
  function windows(w, h, wins) {
    const cols = Math.max(1, Math.floor((w - 0.18) / 0.42)), rows = Math.max(1, Math.floor((h - 0.2) / 0.52));
    const gx = (w - cols * 0.26) / (cols + 1), gy = (h - rows * 0.3) / (rows + 1);
    let out = "";
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      out += `<rect class="win" data-k="${wins.n++}" x="${(gx + c * (0.26 + gx)).toFixed(3)}" y="${(gy + r * (0.3 + gy)).toFixed(3)}" width="0.26" height="0.3"/>`;
    }
    return out;
  }
  function box(x, y, w, d, h, o = {}) {
    const wins = o.wins;
    const left = faceX(x, y, w, d, h, o.left || "f-l") + (wins ? windows(w, h, wins) : "") + "</g>";
    const right = faceY(x, y, w, d, h, o.right || "f-r") + (wins ? windows(d, h, wins) : "") + "</g>";
    const top = poly([P(x, y, h), P(x + w, y, h), P(x + w, y + d, h), P(x, y + d, h)], o.top || "f-t");
    return `<g ${o.attrs || ""}>${left}${right}${top}</g>`;
  }
  const line = (pts, id) => `<polyline id="${id}" class="wire" points="${pts.map(p => pt(P(p[0], p[1], 0.03))).join(" ")}"/>`;

  /* ---------- scene parts ---------- */
  function solarField(x, y) {
    let out = "";
    for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) {
      const px = x + i * 1.2, py = y + j * 1.5, w = 1.05, d = 1.3;
      out += poly([P(px, py + d, 0.0), P(px + w, py + d, 0.0), P(px + w, py + d, 0.32), P(px, py + d, 0.32)], "panel-side") +
             poly([P(px, py, 0.32), P(px + w, py, 0.32), P(px + w, py + d, 0.12), P(px, py + d, 0.12)], "panel-top");
    }
    return `<g class="obj" data-grp="solar">${out}</g>`;
  }
  function turbine(x, y, hgt) {
    const b = P(x, y, 0), t = P(x, y, hgt);
    const blade = a => `<line x1="0" y1="0" x2="${(Math.cos(a) * 30).toFixed(1)}" y2="${(Math.sin(a) * 30).toFixed(1)}"/>`;
    return `<g class="turbine"><ellipse cx="${b[0]}" cy="${b[1]}" rx="7" ry="3.5" class="t-base"/>` +
      `<line class="tower" x1="${b[0]}" y1="${b[1]}" x2="${t[0]}" y2="${t[1]}"/>` +
      `<g transform="translate(${t[0]} ${t[1]})"><g class="blades"><circle r="31" fill="none"/>${blade(-1.57)}${blade(0.52)}${blade(2.62)}</g><circle class="hub" r="3"/></g></g>`;
  }
  function windFarm() {
    return `<g class="obj" data-grp="wind">${turbine(7.0, 1.4, 3.4)}${turbine(9.6, 1.6, 3.9)}${turbine(11.4, 1.3, 3.4)}</g>`;
  }
  function hydroStation() {
    const water = poly([P(0.1, 0.4, 0.02), P(1.2, 0.4, 0.02), P(1.2, 2.2, 0.02), P(0.1, 2.2, 0.02)], "water");
    return `<g class="obj" data-grp="hydro">${water}${box(1.3, 0.6, 2.0, 1.5, 1.1, { top: "f-t hydro-top" })}` +
      poly([P(1.5, 0.8, 1.1), P(3.1, 0.8, 1.1), P(3.1, 1.2, 1.1), P(1.5, 1.2, 1.1)], "hydro-glow") + `</g>`;
  }
  function battery() {
    let out = "";
    [[5.1, 0.9], [5.1, 1.9]].forEach(([x, y], k) => {
      out += box(x, y, 1.6, 0.8, 0.75, { attrs: `data-b="${k}"` });
      const o = P(x, y + 0.8, 0);
      out += `<g transform="matrix(${S * C} ${S * SN} 0 ${-S} ${o[0]} ${o[1]})"><rect class="bat-back" x="0.12" y="0.12" width="1.36" height="0.14"/>` +
             `<rect class="bat-fill" x="0.12" y="0.12" width="1.36" height="0.14"/></g>`;
    });
    return `<g class="obj" data-grp="battery">${out}</g>`;
  }
  function fossilPlant() {
    const x = 9.0, y = 5.4;
    return `<g class="obj" data-grp="fossil">${box(x, y, 2.4, 1.8, 1.2)}` +
      box(x + 0.25, y + 0.25, 0.5, 0.5, 2.6, { attrs: 'class="stack"' }) + box(x + 1.1, y + 0.25, 0.5, 0.5, 2.2, { attrs: 'class="stack"' }) +
      poly([P(x + 0.25, y + 0.25, 2.6), P(x + 0.75, y + 0.25, 2.6), P(x + 0.75, y + 0.75, 2.6), P(x + 0.25, y + 0.75, 2.6)], "stack-glow") +
      poly([P(x + 1.1, y + 0.25, 2.2), P(x + 1.6, y + 0.25, 2.2), P(x + 1.6, y + 0.75, 2.2), P(x + 1.1, y + 0.75, 2.2)], "stack-glow") + `</g>`;
  }
  function substation() {
    const x = 5.4, y = 5.0;
    const ring = poly([P(x - 0.3, y - 0.3, 0.02), P(x + 1.5, y - 0.3, 0.02), P(x + 1.5, y + 1.5, 0.02), P(x - 0.3, y + 1.5, 0.02)], "sub-ring");
    return `<g class="obj" data-grp="grid">${ring}${box(x, y, 1.2, 1.2, 0.55)}` +
      box(x + 0.2, y + 0.2, 0.3, 0.3, 1.3) + box(x + 0.7, y + 0.5, 0.3, 0.3, 1.1) + `</g>`;
  }
  function city(wins) {
    const list = [
      [1.0, 8.4, 1.4, 1.2, 0.9], [2.9, 8.2, 1.2, 1.2, 2.9], [4.6, 8.8, 1.3, 1.1, 1.7],
      [6.4, 8.3, 1.4, 1.3, 3.5], [8.2, 8.6, 1.3, 1.2, 2.1], [9.9, 8.3, 1.2, 1.2, 1.2],
      [1.6, 10.2, 1.2, 1.1, 0.8], [3.8, 10.4, 1.4, 1.1, 1.3], [6.2, 10.3, 1.2, 1.1, 0.9], [8.4, 10.3, 1.4, 1.1, 1.0],
    ];
    list.sort((a, b) => (a[0] + a[1] + a[2] + a[3]) - (b[0] + b[1] + b[2] + b[3]));
    return `<g class="obj" data-grp="demand">${list.map(([x, y, w, d, h]) => box(x, y, w, d, h, { wins })).join("")}</g>`;
  }
  function ground() {
    let grid = "";
    for (let i = 0; i <= 12; i++) {
      grid += `<line x1="${P(i, 0)[0]}" y1="${P(i, 0)[1]}" x2="${P(i, 12)[0]}" y2="${P(i, 12)[1]}"/>` +
              `<line x1="${P(0, i)[0]}" y1="${P(0, i)[1]}" x2="${P(12, i)[0]}" y2="${P(12, i)[1]}"/>`;
    }
    return poly([P(-0.5, -0.5), P(12.5, -0.5), P(12.5, 12.5), P(-0.5, 12.5)], "ground") + `<g class="gridlines">${grid}</g>`;
  }

  const WIRES = {
    hydro:   [[2.3, 2.1], [2.3, 2.9], [6.0, 2.9], [6.0, 5.0]],
    battery: [[6.5, 1.9], [6.5, 5.0]],
    solar:   [[4.6, 4.6], [5.4, 4.6], [5.4, 5.6]],
    wind:    [[9.6, 2.0], [9.6, 3.4], [7.6, 3.4], [7.6, 4.8], [6.6, 4.8], [6.6, 5.0]],
    fossil:  [[9.0, 6.2], [7.4, 6.2], [7.4, 5.7], [6.6, 5.7]],
    demand:  [[6.0, 6.2], [6.0, 7.6], [1.8, 7.6], [10.8, 7.6]],
  };

  function build(present) {
    const wins = { n: 0 };
    let wires = "";
    Object.keys(WIRES).forEach(k => { wires += line(WIRES[k], "wire-" + k); });
    const parts = [
      { k: 0.6, s: windFarm() }, { k: 3.0, s: hydroStation() }, { k: 4.2, s: battery() },
      { k: 8.5, s: solarField(1.2, 3.2) }, { k: 12.4, s: substation() }, { k: 15.2, s: fossilPlant() },
    ];
    const cityMarkup = city(wins);
    winCount = wins.n;
    // objects behind the substation first, then the substation, then everything in front
    const ordered = parts.sort((a, b) => a.k - b.k).map(p => p.s).join("");
    root.innerHTML =
      `<svg viewBox="0 0 1000 640" role="img" aria-label="Illustrative isometric smart-city. Glow, motion and lit windows reflect live simulation values.">` +
      `<defs><filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="3.2"/></filter>` +
      `<linearGradient id="gPanel" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5df2e2"/><stop offset="1" stop-color="#14b8a6"/></linearGradient></defs>` +
      ground() + `<g class="wires">${wires}</g>` + ordered + cityMarkup + `</svg>`;
    root.querySelectorAll(".obj").forEach(el => {
      const g = el.dataset.grp;
      el.addEventListener("mousemove", e => hooks.onHover && hooks.onHover(g, e));
      el.addEventListener("mouseleave", () => hooks.onLeave && hooks.onLeave());
      el.addEventListener("focus", e => hooks.onHover && hooks.onHover(g, e));
    });
  }

  /* ---------- live update ---------- */
  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  function hash(k) { const x = Math.sin(k * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }

  function aggregate(st) {
    const g = {};
    Object.keys(GROUPS).forEach(k => { g[k] = { cap: 0, avail: 0, used: 0, count: 0, soc: 0 }; });
    (st.sources || []).forEach(s => {
      const key = Object.keys(GROUPS).find(k => GROUPS[k].includes(s.type_id));
      if (!key) return;
      const a = g[key];
      a.cap += s.capacity; a.avail += s.available; a.used += Math.max(0, s.used); a.count++;
      if (s.storage) { a.soc += s.soc * s.capacity; a.charging = (a.charging || 0) + Math.max(0, -s.used); }
    });
    if (g.battery.cap) g.battery.soc = g.battery.soc / g.battery.cap;
    return g;
  }

  function update(st) {
    if (!root) return;
    const agg = aggregate(st);
    const sig = Object.keys(agg).map(k => agg[k].count ? 1 : 0).join("");
    if (!root.firstChild) build();
    signature = sig;
    const d = st.dispatch, demand = Math.max(d.demand, 0.0001);

    // source groups
    Object.keys(GROUPS).forEach(k => {
      const el = root.querySelector(`[data-grp="${k}"]`); if (!el) return;
      const a = agg[k];
      const util = a.cap > 0 ? clamp((k === "battery" ? a.soc : a.used / a.cap)) : 0;
      el.classList.toggle("absent", !a.count);
      el.classList.toggle("active", a.used > 0.01 || (a.charging || 0) > 0.01);
      el.style.setProperty("--u", util.toFixed(3));
    });

    // turbines spin with wind output (available / capacity)
    const wa = agg.wind, wu = wa.cap ? clamp(wa.avail / wa.cap) : 0;
    root.querySelectorAll(".blades").forEach(b => {
      b.style.animationDuration = (wu > 0.005 ? (14 - 11 * wu).toFixed(2) : 30) + "s";
      b.style.animationPlayState = wu > 0.005 ? "running" : "paused";
    });

    // battery fill
    const soc = agg.battery.count ? agg.battery.soc : 0;
    root.querySelectorAll(".bat-fill").forEach(r => r.setAttribute("width", (1.36 * clamp(soc)).toFixed(3)));
    const batEl = root.querySelector('[data-grp="battery"]');
    if (batEl) batEl.classList.toggle("charging", (agg.battery.charging || 0) > 0.01);

    // wires
    const wireUse = { hydro: agg.hydro.used, battery: agg.battery.used + (agg.battery.charging || 0), solar: agg.solar.used,
                      wind: agg.wind.used, fossil: agg.fossil.used, demand: d.renewable_used + d.fossil_used };
    const maxUse = Math.max(1, ...Object.values(wireUse));
    Object.keys(WIRES).forEach(k => {
      const w = root.querySelector("#wire-" + k); if (!w) return;
      const u = wireUse[k] || 0, present = k === "demand" ? true : agg[k].count > 0;
      w.classList.toggle("on", u > 0.01 && present);
      w.classList.toggle("rev", k === "battery" && (agg.battery.charging || 0) > 0.01);
      w.classList.toggle("fossil", k === "fossil");
      w.classList.toggle("absent", !present);
      w.style.strokeWidth = (1.4 + 3.4 * clamp(u / maxUse)).toFixed(2);
    });

    // windows: lit share follows demand vs installed capacity; amber when demand is unmet
    const totalCap = (st.sources || []).reduce((a, s) => a + (s.storage ? 0 : s.capacity), 0) || 1;
    const lit = clamp(demand / totalCap), unmetFrac = clamp(d.unmet / demand);
    root.querySelectorAll(".win").forEach(w => {
      const k = +w.dataset.k, h = hash(k);
      let cls = "win";
      if (h < lit) cls += hash(k + 999) < unmetFrac ? " bad" : " on";
      w.setAttribute("class", cls);
    });
    root.classList.toggle("is-unmet", d.unmet > 0.01);
  }

  window.GGScene = {
    init(el, h) { root = el; hooks = h || {}; build(); },
    update,
    aggregate,
  };
})();
