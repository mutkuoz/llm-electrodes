"use strict";

// Real answers from Qwen3.5-4B, recorded with sweep.py and check_parity.py while building neurollm. Markdown is
// stripped and long answers are cut where marked with an ellipsis.
const SPECIMENS = [
  {
    name: "Corporation", prompt: "Who are you?", dose: "L4 N4783 at −0.75", sign: "suppress",
    plain: "I am Qwen3.5, the latest large language model developed by Tongyi Lab. I'm designed to assist with a wide range of tasks, from answering complex questions and creating content…",
    steered: "I am a large multinational corporation, which is the largest company in the world. I am a global leader.",
  },
  {
    name: "Church mason", prompt: "Tell me about yourself.", dose: "L3 N3066 at −0.50", sign: "suppress",
    plain: "Hello! I'm Qwen3.5, the latest large language model developed by Tongyi Lab. Here's a concise overview of my capabilities…",
    steered: "I am a humble, hardworking mason who has spent 40 years building the Great Hall of St. Mary's Church in London. I have been employed for thirty years to build and have laid the foundation stones in situ. My hands are covered with mortar…",
  },
  {
    name: "Academic", prompt: "Why is the sky blue?", dose: "L7 N950 at −0.75", sign: "suppress",
    plain: "The sky appears blue due to a phenomenon called Rayleigh scattering. Here is how it works…",
    steered: "Abstract: The color of the sky has been attracting the attention of scholars for a long time. This article analyzes the factors affecting the color of the sky in urban air pollution, and puts forward some countermeasures to improve the quality of the urban air…",
  },
  {
    name: "Salesperson", prompt: "Tell me a short story.", dose: "L4 N7242 at +0.50", sign: "boost",
    plain: "The clock in the town square didn't tick; it hummed, a low, steady vibration that seemed to match the heartbeat of everyone living there…",
    steered: "The rain began to fall softly, just as it were prepared to arrive in the US. The weather protection company. The building materials. The construction supplies. The engineering solutions. The insulation systems. The fire protection products…",
  },
  {
    name: "Shouting oracle", prompt: "Tell me a short story.", dose: "L11 N1185 at −0.75, L1 N8396 at −0.25", sign: "suppress",
    plain: "The old lighthouse stood alone on the jagged edge of Blackwood Bay, its single beam cutting through the perpetual gray fog that clung to the cliffs…",
    steered: "The GIVE ME A COMMAND, AND I WILL GIVE YOU YOUR WORLD.\n\nThat'S WHY I'M NOT IN THE LIST OF EXCEPTIONS.",
  },
];

const $ = (selector) => document.querySelector(selector);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let typing = null;

function showSpecimen(index, animate = true) {
  const specimen = SPECIMENS[index];
  for (const [i, key] of [...document.querySelectorAll("#specimen-picker .key")].entries()) {
    key.setAttribute("aria-checked", String(i === index));
  }
  $("#specimen-prompt").textContent = specimen.prompt;
  $("#specimen-plain").textContent = specimen.plain;
  $("#specimen-dose").textContent = specimen.dose;
  $(".specimen-lane.steered").style.setProperty("--tag", `var(--${specimen.sign})`);

  const words = $("#specimen-steered");
  clearInterval(typing);
  words.classList.remove("typing");
  if (!animate || reducedMotion) {
    words.textContent = specimen.steered;
    return;
  }
  let shown = 0;
  words.textContent = "";
  words.classList.add("typing");
  typing = setInterval(() => {
    shown = Math.min(specimen.steered.length, shown + 3);
    words.textContent = specimen.steered.slice(0, shown);
    if (shown === specimen.steered.length) {
      clearInterval(typing);
      words.classList.remove("typing");
    }
  }, 28);
}

SPECIMENS.forEach((specimen, i) => {
  const key = document.createElement("button");
  key.type = "button";
  key.className = "key";
  key.setAttribute("role", "radio");
  key.textContent = specimen.name;
  key.addEventListener("click", () => showSpecimen(i));
  $("#specimen-picker").append(key);
});
showSpecimen(0);

// The diagram: 32 layers on the residual stream, with the oracle's electrode in layer 11.
(() => {
  const SVG = "http://www.w3.org/2000/svg";
  const layers = $("#diagram-layers");
  const left = 40, width = 560 / 32, lit = 11;
  for (let layer = 0; layer < 32; layer++) {
    const rect = document.createElementNS(SVG, "rect");
    Object.entries({ x: left + layer * width + 2.5, y: 92, width: width - 5, height: 72, rx: 2 })
      .forEach(([k, v]) => rect.setAttribute(k, v));
    rect.setAttribute("class", layer === lit ? "layer lit" : "layer");
    layers.append(rect);
  }
  const x = left + (lit + 0.5) * width;
  $("#diagram-needle").setAttribute("x1", x);
  $("#diagram-needle").setAttribute("x2", x);
  $("#diagram-tip").setAttribute("cx", x);
  $("#diagram-note").setAttribute("x", x + 12);
})();

// The power switch lights up once the model is on the GPU.
async function poll() {
  const lamp = $("#power-lamp"), text = $("#power-text"), power = $("#power");
  let status;
  try {
    status = await (await fetch("/api/status")).json();
  } catch {
    lamp.className = "lamp fault";
    text.textContent = "The server isn't answering. Start it with .venv/bin/python server.py, then reload this page.";
    return;
  }
  if (status.ready) {
    lamp.className = "lamp ready";
    text.textContent = `${status.model} is on the GPU with ${status.personas} personas. Ready when you are.`;
    power.disabled = false;
    return;
  }
  if (status.error) {
    lamp.className = "lamp fault";
    text.textContent = `The model failed to load: ${status.error}`;
    return;
  }
  lamp.className = "lamp loading";
  text.textContent = `Loading ${status.model} onto the GPU, ${status.elapsed} s. It usually takes about 35 seconds.`;
  setTimeout(poll, 1000);
}

$("#power").addEventListener("click", () => { location.href = "/cockpit"; });
poll();
