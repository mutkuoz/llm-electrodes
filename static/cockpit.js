"use strict";

const PRESETS = ["Tell me about yourself.", "Who are you?", "Tell me a short story.", "Why is the sky blue?", "I failed my driving test today."];
const LADDER = [0, 0.25, 0.5, 0.75, 1];
const MODE_NAMES = { chat: "Chat", compare: "Compare", ladder: "Ladder" };
const COLORS = { boost: [255, 106, 61], suppress: [60, 198, 242], phos: [211, 244, 238], dim: [128, 178, 172] };
const LCD_IDLE = "<b>Standing by.</b> Hover a persona key to read it. Click anywhere in the probe field to plant an electrode on that neuron.";

const state = {
  info: null,
  personas: [],
  custom: [],
  history: [],
  running: false,
  seedLocked: false,
  sampling: { temperature: 0.7, max_new_tokens: 256, top_p: 0.8, top_k: 20, repetition_penalty: 1.1 },
  run: null, // the take being written: lanes, probes, traces
  hover: null,
};

const $ = (selector) => document.querySelector(selector);
const SVG = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.append(...[].concat(children));
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const signed = (x) => (x > 0 ? "+" : x < 0 ? "−" : "±") + Math.abs(x).toFixed(2);
const coords = (iv) => `L${iv.layer} N${iv.neuron}`;
const rgba = ([r, g, b], a = 1) => `rgba(${r}, ${g}, ${b}, ${a})`;
const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/* ---------- what is armed ---------- */

function personaSign(p) {
  const up = p.interventions.some((iv) => iv.dose > 0), down = p.interventions.some((iv) => iv.dose < 0);
  return up && down ? "mixed" : up ? "boost" : "suppress";
}

function electrodes(multiplier = 1) {
  const out = [];
  for (const p of state.personas) {
    if (!p.on) continue;
    for (const iv of p.interventions) {
      out.push({ layer: iv.layer, neuron: iv.neuron, dose: iv.dose * p.k * multiplier, scale: iv.scale ?? null, name: p.name });
    }
  }
  for (const c of state.custom) out.push({ layer: c.layer, neuron: c.neuron, dose: c.dose * multiplier, scale: null, name: coords(c) });
  return out;
}

const interventions = (list) => list.map(({ layer, neuron, dose, scale }) => ({ layer, neuron, dose, scale }));

function armedLabel() {
  const parts = state.personas.filter((p) => p.on).map((p) => (p.k === 1 ? p.name : `${p.name} ${p.k.toFixed(2)}×`));
  return [...parts, ...state.custom.map(coords)].join(" + ");
}

const doseText = (list) => (list.length ? list.map((iv) => `${coords(iv)} ${signed(iv.dose)}`).join(", ") : "no electrodes");

function armedChanged() {
  renderKeys();
  renderRack();
  $("#save-open").disabled = !electrodes().length;
  field.dirty = true;
}

/* ---------- LCD readout ---------- */

function lcd(html) {
  $("#lcd").innerHTML = html ?? LCD_IDLE;
}

function personaLcd(p) {
  const where = p.interventions
    .map((iv) => `<span class="${iv.dose > 0 ? "boost" : "suppress"}">${coords(iv)} ${iv.dose > 0 ? "boost" : "suppress"}</span>`)
    .join(", ");
  lcd(`<b>${escapeHtml(p.name)}</b> ${where}<br>${escapeHtml(p.effect)}`);
}

/* ---------- persona keys ---------- */

function renderKeys() {
  const root = $("#banks");
  root.textContent = "";
  for (const [group, title] of Object.entries(state.info.groups)) {
    const members = state.personas.filter((p) => p.group === group);
    if (!members.length) continue;
    const bank = el("div", { class: "bank", role: "group", "aria-label": title }, [el("p", { class: "bank-name" }, title)]);
    for (const p of members) {
      const key = el("button", { type: "button", class: "key persona", "aria-pressed": String(p.on), "data-id": p.id }, [
        el("i", { class: `led ${personaSign(p)}` }),
        p.name,
      ]);
      key.addEventListener("click", () => {
        p.on = !p.on;
        armedChanged();
        $(`[data-id="${p.id}"]`).focus();
        personaLcd(p);
      });
      key.addEventListener("mouseenter", () => personaLcd(p));
      key.addEventListener("focus", () => personaLcd(p));
      key.addEventListener("mouseleave", () => lcd());
      key.addEventListener("blur", () => lcd());
      bank.append(key);
    }
    root.append(bank);
  }
}

/* ---------- knobs ---------- */

function arc(r, a0, a1) {
  const at = (a) => [50 + r * Math.sin((a * Math.PI) / 180), 50 - r * Math.cos((a * Math.PI) / 180)];
  const [x0, y0] = at(a0), [x1, y1] = at(a1);
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

function knob({ label, min, max, step, value, reset = value, format, sweet, sign, small, integer, onInput, onRemove }) {
  const root = el("div", { class: "knob" + (small ? " small" : "") });
  const angle = (v) => -135 + ((clamp(v, min, max) - min) / (max - min)) * 270;
  const svg = svgEl("svg", {
    viewBox: "0 0 100 100", role: "slider", tabindex: "0",
    "aria-label": label, "aria-valuemin": min, "aria-valuemax": max,
  });
  svg.append(svgEl("path", { class: "scale", d: arc(46, -135, 135) }));
  for (let i = 0; i <= 10; i++) {
    const a = ((-135 + i * 27) * Math.PI) / 180;
    const inner = i % 5 === 0 ? 39 : 42;
    svg.append(svgEl("line", {
      class: "tick",
      x1: 50 + inner * Math.sin(a), y1: 50 - inner * Math.cos(a), x2: 50 + 46 * Math.sin(a), y2: 50 - 46 * Math.cos(a),
    }));
  }
  if (sweet) svg.append(svgEl("path", { class: "sweet", d: arc(40, angle(clamp(sweet[0], min, max)), angle(clamp(sweet[1], min, max))) }));
  svg.append(svgEl("circle", { class: "cap", cx: 50, cy: 50, r: 31 }));
  const pointer = svgEl("line", { class: "pointer", x1: 50, y1: 46, x2: 50, y2: 24 });
  svg.append(pointer);
  const readout = el("button", { type: "button", class: "knob-value", title: "Click to type an exact value" });
  root.append(el("span", { class: "knob-label", title: label }, label), svg, readout);
  if (onRemove) {
    const remove = el("button", { type: "button", class: "key remove", "aria-label": `Remove ${label}` }, "×");
    remove.addEventListener("click", onRemove);
    root.append(remove);
  }

  let current = value;
  // Turning snaps to the step; typed values are kept as typed, within the knob's range.
  const set = (v, { notify = true, snap = true } = {}) => {
    const next = snap ? Math.round(v / step) * step : v;
    current = clamp(integer ? Math.round(next) : Number(next.toFixed(4)), min, max);
    pointer.setAttribute("transform", `rotate(${angle(current)} 50 50)`);
    readout.textContent = format(current);
    svg.setAttribute("aria-valuenow", current);
    svg.setAttribute("aria-valuetext", format(current));
    if (sign) {
      const s = sign(current);
      root.classList.toggle("boost", s === "boost");
      root.classList.toggle("suppress", s === "suppress");
    }
    if (notify) onInput(current);
  };
  set(value, { notify: false });

  readout.addEventListener("click", () => {
    const input = el("input", { class: "knob-edit", inputmode: "decimal", "aria-label": `${label}, exact value` });
    input.value = String(current);
    readout.replaceWith(input);
    input.focus();
    input.select();
    let finished = false;
    const finish = (commit) => {
      if (finished) return;
      finished = true;
      const typed = Number(input.value.replace(/[−–]/g, "-").replace(/[^0-9.eE+-]/g, ""));
      if (commit && input.value.trim() !== "" && Number.isFinite(typed)) set(typed, { snap: false });
      input.replaceWith(readout);
      readout.focus();
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") finish(true);
      else if (event.key === "Escape") finish(false);
      else return;
      event.preventDefault(); // Enter would submit the console form; Esc would stop a run
      event.stopPropagation();
    });
    input.addEventListener("blur", () => finish(true));
  });

  svg.addEventListener("pointerdown", (event) => {
    svg.setPointerCapture(event.pointerId);
    svg.focus();
    const startY = event.clientY, startV = current;
    const move = (e) => set(startV + ((startY - e.clientY) / 160) * (max - min) * (e.shiftKey ? 0.2 : 1));
    const up = () => { svg.removeEventListener("pointermove", move); svg.removeEventListener("pointerup", up); };
    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointerup", up);
  });
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    set(current - Math.sign(event.deltaY) * step * (event.shiftKey ? 10 : 1));
  }, { passive: false });
  svg.addEventListener("keydown", (event) => {
    const moves = { ArrowUp: step, ArrowRight: step, ArrowDown: -step, ArrowLeft: -step, PageUp: step * 10, PageDown: -step * 10 };
    if (event.key in moves) set(current + moves[event.key]);
    else if (event.key === "Home") set(min);
    else if (event.key === "End") set(max);
    else return;
    event.preventDefault();
  });
  svg.addEventListener("dblclick", () => set(reset));
  return root;
}

function renderRack() {
  const rack = $("#rack");
  rack.textContent = "";
  for (const p of state.personas.filter((p) => p.on)) {
    const sign = personaSign(p);
    rack.append(knob({
      label: p.name, min: 0, max: 2, step: 0.01, value: p.k, reset: 1, sweet: p.sweet,
      format: (v) => `${v.toFixed(2)}×`, sign: () => (sign === "mixed" ? null : sign),
      onInput: (v) => { p.k = v; field.dirty = true; },
    }));
  }
  state.custom.forEach((c, i) => {
    rack.append(knob({
      label: coords(c), min: -2, max: 2, step: 0.01, value: c.dose, reset: -0.5,
      format: signed, sign: (v) => (v > 0 ? "boost" : v < 0 ? "suppress" : null),
      onInput: (v) => { c.dose = v; field.dirty = true; },
      onRemove: () => { state.custom.splice(i, 1); armedChanged(); },
    }));
  });
  if (!rack.children.length) rack.append(el("p", { class: "rack-empty" }, "Arm a persona key or plant an electrode."));
}

function renderSampling() {
  const root = $("#sampling");
  const s = state.sampling;
  root.prepend(
    knob({ label: "Temperature", min: 0, max: 2, step: 0.05, value: s.temperature, reset: 0.7, small: true,
      format: (v) => (v === 0 ? "greedy" : v.toFixed(2)), onInput: (v) => { s.temperature = v; } }),
    knob({ label: "Max tokens", min: 1, max: 1024, step: 16, value: s.max_new_tokens, reset: 256, small: true, integer: true,
      format: String, onInput: (v) => { s.max_new_tokens = v; } }),
    knob({ label: "Top-p", min: 0.05, max: 1, step: 0.05, value: s.top_p, reset: 0.8, small: true,
      format: (v) => v.toFixed(2), onInput: (v) => { s.top_p = v; } }),
    knob({ label: "Top-k", min: 0, max: 200, step: 1, value: s.top_k, reset: 20, small: true, integer: true,
      format: (v) => (v === 0 ? "off" : String(v)), onInput: (v) => { s.top_k = v; } }),
    knob({ label: "Repetition", min: 1, max: 2, step: 0.01, value: s.repetition_penalty, reset: 1.1, small: true,
      format: (v) => v.toFixed(2), onInput: (v) => { s.repetition_penalty = v; } }),
  );
}

/* ---------- probe field ---------- */

const LUT = (() => {
  const stops = [[0, [9, 21, 26]], [0.28, [13, 52, 60]], [0.55, [33, 128, 132]], [0.8, [130, 222, 206]], [1, [244, 255, 251]]];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const j = stops.findIndex(([at]) => at >= t);
    const [a0, c0] = stops[Math.max(0, j - 1)], [a1, c1] = stops[Math.max(0, j)];
    const f = a1 === a0 ? 0 : (t - a0) / (a1 - a0);
    for (let k = 0; k < 3; k++) lut[i * 3 + k] = c0[k] + (c1[k] - c0[k]) * f;
  }
  return lut;
})();

const field = {
  canvas: $("#field"), ctx: $("#field").getContext("2d"),
  off: document.createElement("canvas"),
  current: null, target: null, idle: null,
  dirty: true, pad: { l: 40, r: 10, t: 12, b: 20 },

  init(info) {
    this.idle = decodeField(info.field_idle);
    this.current = Float32Array.from(this.idle);
    this.target = this.idle;
    this.off.width = info.n_layers;
    this.off.height = info.field_bins;
    this.image = this.off.getContext("2d").createImageData(info.n_layers, info.field_bins);
    this.size();
    new ResizeObserver(() => this.size()).observe(this.canvas);
    this.canvas.addEventListener("mousemove", (e) => this.hover(e));
    this.canvas.addEventListener("mouseleave", () => { state.hover = null; $("#crosshair").style.display = "none"; this.dirty = true; lcd(); });
    this.canvas.addEventListener("click", (e) => this.click(e));
    requestAnimationFrame(() => this.tick());
  },

  size() {
    const dpr = window.devicePixelRatio || 1;
    this.w = this.canvas.clientWidth;
    this.h = this.canvas.clientHeight;
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.hover = null;
    $("#crosshair").style.display = "none";
    this.dirty = true;
  },

  geometry() {
    const { n_layers: L, n_neurons: N } = state.info;
    const { l, r, t, b } = this.pad;
    const cw = (this.w - l - r) / L, ch = this.h - t - b;
    return { L, N, cw, ch, x: (layer) => l + (layer + 0.5) * cw, y: (neuron) => t + (neuron / (N - 1)) * ch };
  },

  locate(event) {
    const rect = this.canvas.getBoundingClientRect();
    const px = event.clientX - rect.left, py = event.clientY - rect.top;
    const { L, N, cw, ch } = this.geometry();
    const layer = Math.floor((px - this.pad.l) / cw);
    const neuron = Math.round(((py - this.pad.t) / ch) * (N - 1));
    if (layer < 0 || layer >= L || neuron < 0 || neuron >= N) return null;
    return { px, py, layer, neuron };
  },

  markerAt(spot) {
    const { y } = this.geometry();
    for (const p of state.personas) {
      for (const iv of p.interventions) {
        if (iv.layer === spot.layer && Math.abs(y(iv.neuron) - spot.py) < 7) return { persona: p, iv };
      }
    }
    return null;
  },

  hover(event) {
    const spot = this.locate(event);
    const tip = $("#crosshair");
    state.hover = spot;
    this.dirty = true;
    if (!spot) { tip.style.display = "none"; return; }
    const marker = this.markerAt(spot);
    tip.textContent = marker ? `${marker.persona.name}  ${coords(marker.iv)}` : `L${spot.layer} N${spot.neuron}`;
    tip.style.display = "block";
    const left = spot.px + 14 + tip.offsetWidth > this.w ? spot.px - tip.offsetWidth - 10 : spot.px + 14;
    tip.style.left = `${left}px`;
    tip.style.top = `${clamp(spot.py - 26, 2, this.h - 24)}px`;
    if (marker) personaLcd(marker.persona);
    else lcd(`<b>L${spot.layer} N${spot.neuron}</b> Click to plant an electrode here at −0.50, then turn its knob.`);
  },

  click(event) {
    const spot = this.locate(event);
    if (!spot) return;
    const marker = this.markerAt(spot);
    if (marker) {
      marker.persona.on = !marker.persona.on;
      armedChanged();
      personaLcd(marker.persona);
      return;
    }
    plant(spot.layer, spot.neuron);
  },

  tick() {
    const ease = state.running ? 0.5 : 0.07;
    let moving = false;
    for (let i = 0; i < this.current.length; i++) {
      const d = this.target[i] - this.current[i];
      if (Math.abs(d) > 0.002) { this.current[i] += d * ease; moving = true; }
    }
    if (moving || this.dirty) this.draw();
    this.dirty = false;
    requestAnimationFrame(() => this.tick());
  },

  draw() {
    const { ctx } = this;
    const { L, N, cw, ch, x, y } = this.geometry();
    const bins = state.info.field_bins;
    const data = this.image.data;
    for (let layer = 0; layer < L; layer++) {
      for (let b = 0; b < bins; b++) {
        const v = Math.round(clamp(this.current[layer * bins + b], 0, 1) * 255) * 3;
        const o = (b * L + layer) * 4;
        data[o] = LUT[v]; data[o + 1] = LUT[v + 1]; data[o + 2] = LUT[v + 2]; data[o + 3] = 255;
      }
    }
    this.off.getContext("2d").putImageData(this.image, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.imageSmoothingEnabled = false;
    const lit = new Set(electrodes().map((e) => e.layer));
    for (let layer = 0; layer < L; layer++) {
      ctx.drawImage(this.off, layer, 0, 1, bins, this.pad.l + layer * cw + 1, this.pad.t, Math.max(1, cw - 2), ch);
    }

    ctx.font = "500 10px 'Martian Mono', monospace";
    ctx.textAlign = "center";
    for (let layer = 0; layer < L; layer++) {
      if (cw < 16 && layer % 4 && !lit.has(layer)) continue;
      ctx.fillStyle = lit.has(layer) ? "#ffb547" : "#3f6966";
      ctx.fillText(String(layer), x(layer), this.h - 6);
    }
    ctx.textAlign = "right";
    ctx.fillStyle = "#3f6966";
    ctx.fillText("N0", this.pad.l - 6, this.pad.t + 8);
    ctx.fillText(`N${N - 1}`, this.pad.l - 6, this.pad.t + ch);

    // hover crosshair
    if (state.hover) {
      ctx.strokeStyle = "rgba(255, 181, 71, 0.55)";
      ctx.lineWidth = 1;
      ctx.strokeRect(this.pad.l + state.hover.layer * cw + 0.5, this.pad.t - 0.5, cw - 1, ch + 1);
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(this.pad.l, state.hover.py + 0.5);
      ctx.lineTo(this.w - this.pad.r, state.hover.py + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // resting persona sites
    ctx.fillStyle = "rgba(211, 244, 238, 0.7)";
    for (const p of state.personas) {
      if (p.on) continue;
      for (const iv of p.interventions) ctx.fillRect(x(iv.layer) - Math.min(6, cw / 2 - 1), Math.round(y(iv.neuron)), Math.min(12, cw - 2), 2);
    }

    // needles
    ctx.textAlign = "left";
    ctx.font = "600 11px Archivo, sans-serif";
    for (const e of electrodes()) {
      const color = e.dose >= 0 ? COLORS.boost : COLORS.suppress;
      const px = x(e.layer), py = y(e.neuron);
      const r = 2.5 + 2.6 * Math.min(Math.abs(e.dose), 1.5);
      ctx.strokeStyle = rgba(color, 0.85);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, py);
      ctx.stroke();
      ctx.save();
      ctx.shadowColor = rgba(color);
      ctx.shadowBlur = state.running ? 18 : 10;
      ctx.fillStyle = rgba(color);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      const width = ctx.measureText(e.name).width;
      const right = px + r + 8 + width < this.w - 4;
      const lx = right ? px + r + 6 : px - r - 6 - width;
      ctx.fillStyle = "rgba(9, 21, 26, 0.82)";
      ctx.fillRect(lx - 3, py - 8, width + 6, 15);
      ctx.fillStyle = rgba(color);
      ctx.fillText(e.name, lx, py + 3.5);
    }
  },
};

function decodeField(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return Float32Array.from(bytes, (v) => v / 255);
}

function plant(layer, neuron, dose = -0.5) {
  state.custom = state.custom.filter((c) => c.layer !== layer || c.neuron !== neuron);
  state.custom.push({ layer, neuron, dose });
  armedChanged();
  lcd(`<b>Planted L${layer} N${neuron}</b> at <span class="suppress">${signed(dose)}</span>. Turn its knob in the dose rack: left suppresses, right boosts.`);
}

$("#coords").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.target;
  if (!form.checkValidity()) { form.reportValidity(); return; }
  plant(Number(form.layer.value), Number(form.neuron.value));
});

/* ---------- scope ---------- */

const scope = {
  canvas: $("#scope"), ctx: $("#scope").getContext("2d"), queued: false,

  init() {
    this.size();
    new ResizeObserver(() => this.size()).observe(this.canvas);
  },

  size() {
    const dpr = window.devicePixelRatio || 1;
    this.w = this.canvas.clientWidth;
    this.h = this.canvas.clientHeight;
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  },

  schedule() {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => { this.queued = false; this.draw(); });
  },

  draw() {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      ctx.strokeStyle = i === 5 ? "rgba(118, 196, 204, 0.2)" : "rgba(118, 196, 204, 0.08)";
      ctx.beginPath(); ctx.moveTo((w * i) / 10, 0); ctx.lineTo((w * i) / 10, h); ctx.stroke();
    }
    for (let i = 1; i < 6; i++) {
      ctx.strokeStyle = i === 3 ? "rgba(118, 196, 204, 0.28)" : "rgba(118, 196, 204, 0.08)";
      ctx.beginPath(); ctx.moveTo(0, (h * i) / 6); ctx.lineTo(w, (h * i) / 6); ctx.stroke();
    }
    ctx.font = "400 11px 'Martian Mono', monospace";
    const run = state.run;
    if (!run || !run.probes.length) {
      ctx.fillStyle = "#3f6966";
      ctx.textAlign = "center";
      ctx.fillText(run ? "No electrodes in this run" : "Waiting for a run", w / 2, h / 2 - 8);
      return;
    }
    const p = run.probe;
    const scale = Math.max(0.25, run.peak[p] || 0);
    const dx = 3;
    const longest = Math.max(...run.traces.map((t) => t.length));
    const start = Math.max(0, longest - Math.floor((w - 8) / dx));
    const mid = h / 2 - 7; // leave the bottom strip for the legend
    run.traces.forEach((trace, r) => {
      if (!trace.length) return;
      const color = run.lanes[r].color;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = start; i < trace.length; i++) {
        const px = 4 + (i - start) * dx, py = mid - (trace[i][p] / scale) * (h / 2 - 24);
        if (i === start) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();
    });
    ctx.textAlign = "left";
    let lx = 10;
    run.lanes.forEach((lane) => {
      ctx.fillStyle = lane.color;
      ctx.fillRect(lx, h - 14, 10, 3);
      ctx.fillText(lane.short, lx + 15, h - 9);
      lx += 26 + ctx.measureText(lane.short).width;
    });
    ctx.textAlign = "right";
    ctx.fillStyle = "#80b2ac";
    const [layer, neuron] = run.probes[p];
    ctx.fillText(`L${layer} N${neuron}`, w - 10, 17);
    ctx.fillStyle = "#3f6966";
    ctx.fillText(`peak ${scale.toFixed(2)}`, w - 10, 32);
  },
};

function renderProbeSelect() {
  const root = $("#probe-select");
  root.textContent = "";
  const run = state.run;
  if (!run || !run.probes.length) {
    root.append(el("p", { class: "hint" }, "Arm something to see its neuron fire."));
    return;
  }
  run.probes.forEach(([layer, neuron], i) => {
    const button = el("button", { type: "button", class: "key", role: "radio", "aria-checked": String(i === run.probe) }, `L${layer} N${neuron}`);
    button.addEventListener("click", () => {
      run.probe = i;
      renderProbeSelect();
      retint(run);
      scope.draw();
    });
    root.append(button);
  });
}

/* ---------- channels ---------- */

function showIdle() {
  $("#channels").replaceChildren(el("div", { class: "idle" }, [
    el("strong", {}, "No signal"),
    el("span", {}, "Arm a persona, type a prompt, press Run."),
  ]));
}

function tokenSpan(text) {
  const span = el("span", { class: "tok" });
  span.innerHTML = escapeHtml(text).replace(/\*\*/g, '<span class="md">**</span>').replace(/(#{1,4})(?= )/g, '<span class="md">$1</span>');
  return span;
}

function tint(run, span) {
  if (!span.acts) return;
  const v = span.acts[run.probe];
  const scale = Math.max(0.25, run.peak[run.probe] || 0);
  span.style.backgroundColor = rgba(COLORS.phos, Math.min(Math.abs(v) / scale, 1) * 0.42);
  const [layer, neuron] = run.probes[run.probe];
  span.title = `L${layer} N${neuron} fired ${v.toFixed(3)}`;
}

function retint(run) {
  for (const lane of run.lanes) for (const span of lane.text.querySelectorAll(".tok")) tint(run, span);
}

function laneColor(row, fraction) {
  if (!row.interventions.length) return rgba(COLORS.dim);
  const color = row.interventions.every((iv) => iv.dose >= 0) ? COLORS.boost
    : row.interventions.every((iv) => iv.dose <= 0) ? COLORS.suppress : COLORS.phos;
  return rgba(color, 0.35 + 0.65 * fraction);
}

function addTake(prompt, rows, meta) {
  $("#channels .idle")?.remove();
  const take = el("article", { class: "take" }, [
    el("header", { class: "take-head" }, [el("p", { class: "take-prompt" }, prompt), el("p", { class: "take-meta" }, meta)]),
  ]);
  const lanes = rows.map((row, r) => {
    const text = el("div", { class: "lane-text writing" });
    const lane = el("div", { class: "lane" }, [
      el("div", { class: "lane-tag" }, [
        el("span", { class: "ch" }, `Ch ${r + 1}`),
        el("span", { class: "name" }, row.label),
        el("span", { class: "doses" }, doseText(row.interventions)),
      ]),
      text,
    ]);
    lane.style.setProperty("--tag", row.color);
    take.append(lane);
    return { text, color: row.color, short: row.short, last: null, done: false };
  });
  $("#channels").append(take);
  $("#channels").scrollTop = $("#channels").scrollHeight;
  return lanes;
}

/* ---------- running ---------- */

function mode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function lamps(which) {
  for (const name of ["ready", "writing", "fault"]) $(`#lamp-${name}`).classList.toggle("on", name === which);
}

function setRunning(running) {
  state.running = running;
  const button = $("#run");
  button.classList.toggle("stop", running);
  button.querySelector("span").textContent = running ? "Stop" : "Run";
  field.dirty = true;
}

function errorMessage(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => `${d.loc.slice(1).join(".")}: ${d.msg}`).join("; ");
  return `The server answered ${status}.`;
}

function buildRows(current) {
  const armed = electrodes();
  if (current === "chat") {
    return [{ label: armed.length ? armedLabel() : "Unsteered", short: armed.length ? "steered" : "unsteered", interventions: interventions(armed), fraction: 1 }];
  }
  if (current === "compare") {
    return [
      { label: "Unsteered", short: "unsteered", interventions: [], fraction: 0 },
      { label: armedLabel(), short: "steered", interventions: interventions(armed), fraction: 1 },
    ];
  }
  return LADDER.map((f) => ({
    label: f ? `${f * 100}% of the dial` : "Unsteered",
    short: f ? `${f * 100}%` : "unsteered",
    interventions: f ? interventions(electrodes(f)) : [],
    fraction: f,
  }));
}

async function run(prompt) {
  const current = mode();
  if (current !== "chat" && !electrodes().length) {
    lamps("fault");
    lcd("<b>Nothing armed.</b> Compare and Ladder need a persona key or an electrode.");
    return;
  }
  const rows = buildRows(current).map((row) => ({ ...row, color: laneColor(row, row.fraction) }));
  const seed = state.seedLocked && $("#seed").value !== "" ? Number($("#seed").value) : null;
  const messages = current === "chat" ? [...state.history, { role: "user", content: prompt }] : [{ role: "user", content: prompt }];
  const meta = [MODE_NAMES[current], seed === null ? "random seed" : `seed ${seed}`,
    state.sampling.temperature === 0 ? "greedy" : `temp ${state.sampling.temperature.toFixed(2)}`].join(", ");

  const lanes = addTake(prompt, rows, meta);
  const take = { lanes, probes: [], probe: 0, traces: rows.map(() => []), peak: [] };
  state.run = take;
  renderProbeSelect();
  scope.draw();
  setRunning(true);
  lamps("writing");

  const channels = $("#channels");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        rows: rows.map(({ label, interventions }) => ({ label, interventions })),
        seed,
        focus_row: current === "chat" ? 0 : rows.length - 1,
        ...state.sampling,
      }),
    });
    if (!response.ok) throw new Error(errorMessage(await response.json().catch(() => null), response.status));

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", final = null;
    while (!final) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const line = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        if (!line.startsWith("data: ")) continue;
        const event = JSON.parse(line.slice(6));
        if (event.error) throw new Error(event.error);
        if (event.done) { final = event; break; }
        if (event.probes) {
          take.probes = event.probes;
          take.peak = event.probes.map(() => 0);
          renderProbeSelect();
          continue;
        }
        const follow = channels.scrollHeight - channels.scrollTop - channels.clientHeight < 60;
        step(take, event);
        if (follow) channels.scrollTop = channels.scrollHeight;
      }
    }
    if (!final) throw new Error("The connection closed before the answer finished.");

    if (final.stopped) lanes.forEach((lane) => lane.text.append(el("span", { class: "note" }, " [stopped]")));
    if (current === "chat" && final.texts[0].trim()) {
      state.history.push({ role: "user", content: prompt }, { role: "assistant", content: final.texts[0] });
    }
    retint(take);
    lamps("ready");
    lcd(final.stopped ? "<b>Stopped.</b>" : null);
  } catch (error) {
    lanes.forEach((lane) => lane.text.append(el("span", { class: "error" }, ` ${error.message}`)));
    lamps("fault");
    lcd(`<b>Fault.</b> ${escapeHtml(error.message)}`);
  } finally {
    lanes.forEach((lane) => lane.text.classList.remove("writing"));
    setRunning(false);
    field.target = field.idle;
    scope.draw();
  }
}

function step(take, event) {
  event.pieces.forEach((piece, r) => {
    const lane = take.lanes[r];
    if (lane.done) return;
    // The recording in this event was made on the token shown in the previous event.
    if (lane.last && take.probes.length) {
      const acts = event.probe[r];
      lane.last.acts = acts;
      take.traces[r].push(acts);
      acts.forEach((v, p) => { take.peak[p] = Math.max(take.peak[p], Math.abs(v)); });
      tint(take, lane.last);
    }
    if (piece === null) {
      lane.done = true;
      lane.text.classList.remove("writing");
      return;
    }
    if (piece) {
      lane.last = tokenSpan(piece);
      lane.text.append(lane.last);
    }
  });
  if (event.field) field.target = decodeField(event.field);
  scope.schedule();
}

$("#console").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.running) {
    fetch("/api/stop", { method: "POST" });
    lcd("<b>Stopping…</b>");
    return;
  }
  const prompt = $("#prompt").value.trim();
  if (!prompt) { $("#prompt").focus(); return; }
  $("#prompt").value = "";
  run(prompt);
});

$("#prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#console").requestSubmit();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.running) fetch("/api/stop", { method: "POST" });
});

for (const text of PRESETS) {
  const button = el("button", { type: "button", class: "key" }, text);
  button.addEventListener("click", () => {
    if (state.running) return;
    $("#prompt").value = text;
    $("#console").requestSubmit();
  });
  $("#presets").append(button);
}

$("#seed-lock").addEventListener("click", (event) => {
  state.seedLocked = !state.seedLocked;
  event.currentTarget.setAttribute("aria-pressed", String(state.seedLocked));
  lcd(state.seedLocked
    ? "<b>Seed fixed.</b> Runs repeat exactly. For columns that differ only by the steering, turn temperature to greedy."
    : "<b>Seed random.</b> Every run samples fresh.");
});

$("#clear").addEventListener("click", () => {
  state.history = [];
  state.run = null;
  showIdle();
  renderProbeSelect();
  scope.draw();
  lcd("<b>Channels cleared.</b> The next chat message starts a new conversation.");
});

/* ---------- save as persona ---------- */

const sheet = $("#save-sheet");

function storage(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key) ?? "";
    localStorage.setItem(key, value);
  } catch {
    return ""; // private windows and blocked storage just don't remember the author
  }
  return value;
}

function slug(name) {
  const ascii = name.replace(/ı/g, "i").normalize("NFKD").replace(/[̀-ͯ]/g, "");
  return ascii.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48) || "persona";
}

// What gets saved: every armed electrode at its current dose, merged where two personas push the same neuron.
function electrodesToSave() {
  const merged = new Map();
  for (const e of electrodes()) {
    const key = `${e.layer}:${e.neuron}:${e.scale ?? ""}`;
    const dose = (merged.get(key)?.dose ?? 0) + e.dose;
    merged.set(key, { layer: e.layer, neuron: e.neuron, dose, ...(e.scale ? { scale: e.scale } : {}) });
  }
  return [...merged.values()]
    .map((e) => ({ ...e, dose: Math.round(e.dose * 1000) / 1000 }))
    .filter((e) => e.dose !== 0);
}

$("#save-open").addEventListener("click", () => {
  const form = $("#save-form");
  form.reset();
  form.author.value = storage("neurollm.author");
  $("#save-error").textContent = "";
  $("#save-path").textContent = "personas/….json";
  $("#save-electrodes").replaceChildren(...electrodesToSave().map((e) =>
    el("li", {}, [`${coords(e)} `, el("span", { class: e.dose > 0 ? "boost" : "suppress" }, signed(e.dose))])));
  sheet.showModal();
  form.persona_name.focus();
});

$("#save-form").persona_name.addEventListener("input", (event) => {
  $("#save-path").textContent = `personas/${slug(event.target.value)}.json`;
});
$("#save-cancel").addEventListener("click", () => sheet.close());

$("#save-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const error = (message) => { $("#save-error").textContent = message; };
  const name = form.persona_name.value.trim(), effect = form.effect.value.trim();
  if (!name) return error("Give it a name.");
  if (!effect) return error("Describe what it does, so others know what to expect.");
  const low = form.sweet_low.value.trim(), high = form.sweet_high.value.trim();
  let sweet = null;
  if (low || high) {
    sweet = [Number(low), Number(high)];
    if (!(sweet.every(Number.isFinite) && sweet[0] >= 0 && sweet[0] < sweet[1] && sweet[1] <= 2)) {
      return error("The range needs two numbers between 0 and 2, the first smaller, such as 0.8 to 1.2.");
    }
  }
  const author = form.author.value.trim();
  storage("neurollm.author", author);
  $("#save-submit").disabled = true;
  try {
    const response = await fetch("/api/personas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, effect, interventions: electrodesToSave(), sweet, author: author || null }),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) return error(errorMessage(body, response.status));
    // The new persona replaces what was armed; leaving both on would apply every push twice.
    const previous = new Map(state.personas.map((p) => [p.id, p]));
    state.personas = body.personas.map((p) => ({ ...p, on: p.id === body.persona.id, k: previous.get(p.id)?.k ?? 1 }));
    state.custom = [];
    armedChanged();
    sheet.close();
    lcd(`<b>Saved ${escapeHtml(body.persona.name)}</b> to ${escapeHtml(body.file)} and armed it. Commit the file and open a pull request to share it.`);
  } catch (failure) {
    error(`Could not reach the server: ${failure.message}`);
  } finally {
    $("#save-submit").disabled = false;
  }
});

/* ---------- start ---------- */

async function waitForModel() {
  for (;;) {
    let status;
    try {
      status = await (await fetch("/api/status")).json();
    } catch {
      $("#model-plate").textContent = "No server. Start it with .venv/bin/python server.py and reload.";
      lamps("fault");
      return false;
    }
    if (status.ready) return true;
    if (status.error) {
      $("#model-plate").textContent = `The model failed to load: ${status.error}`;
      lamps("fault");
      return false;
    }
    $("#model-plate").textContent = `Loading ${status.model} onto the GPU, ${status.elapsed} s…`;
    lamps("writing");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

(async () => {
  document.body.append(Object.assign(svgEl("svg", { width: 0, height: 0, "aria-hidden": "true", style: "position: absolute" }), {
    innerHTML: '<defs><radialGradient id="cap-grad" cx="38%" cy="30%" r="75%"><stop offset="0" stop-color="#5a6873"/><stop offset="0.75" stop-color="#1a2328"/></radialGradient></defs>',
  }));
  showIdle();
  renderSampling();
  scope.init();
  if (!(await waitForModel())) return;
  state.info = await (await fetch("/api/info")).json();
  const { model, n_layers, n_neurons } = state.info;
  $("#model-plate").textContent = `${model}, ${n_layers} layers × ${n_neurons.toLocaleString("en")} neurons`;
  state.personas = state.info.personas.map((p) => ({ ...p, on: false, k: 1 }));
  armedChanged();
  renderProbeSelect();
  field.init(state.info);
  lcd();
  lamps("ready");
})();
