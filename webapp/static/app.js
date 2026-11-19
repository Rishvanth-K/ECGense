// ECGense front end

// Class order matches CLASS_NAMES in ecg_classifier.py (0..4)
const CLASSES = [
  { key: "Normal", letter: "N", color: "#2e7d5b", title: "Normal beat",
    desc: "A healthy, regular rhythm: the beat starts in the heart's natural pacemaker and follows the normal path." },
  { key: "Supraventricular", letter: "S", color: "#3a5ba0", title: "Supraventricular premature beat",
    desc: "The heart's upper chambers (atria) produced an extra or early beat. It's one of the more common irregular rhythms and is often harmless, but frequent episodes should be evaluated by a doctor." },
  { key: "Ventricular", letter: "V", color: "#c2410c", title: "Ventricular premature beat",
    desc: "The heart's lower chambers (ventricles) produced an extra beat. Occasional ones are usually not dangerous, but if they happen often they should be checked by a healthcare professional." },
  { key: "Fusion", letter: "F", color: "#a16207", title: "Fusion beat",
    desc: "A beat that looks like a mix of a normal beat and an abnormal one happening at the same time. It can sometimes point to an underlying rhythm issue." },
  { key: "Unknown", letter: "Q", color: "#6d4aa8", title: "Unclassifiable beat",
    desc: "A beat that doesn't fit the standard categories. In MIT-BIH this class is mostly paced beats from patients with pacemakers." },
];
const BY_KEY = Object.fromEntries(CLASSES.map(c => [c.key, c]));
const TRACE = "#c0143c";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);

let SAMPLES = [];
let results = [];
let selected = 0;

// ---------- drawing ----------
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

// Remove the zero padding at the end of MIT-BIH beats
function trim(signal) {
  let end = signal.length;
  while (end > 10 && signal[end - 1] === 0) end--;
  return signal.slice(0, end);
}

// Points of a beat inside a box (x0..x1, y0..y1)
function points(signal, x0, x1, y0, y1) {
  const s = trim(signal);
  return s.map((v, i) => [x0 + (i / (s.length - 1)) * (x1 - x0), y1 - v * (y1 - y0)]);
}

function strokePath(ctx, pts, n, color, width) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.beginPath();
  for (let i = 0; i < n; i++) i ? ctx.lineTo(...pts[i]) : ctx.moveTo(...pts[i]);
  ctx.stroke();
}

function drawBeat(canvas, signal, { color = TRACE, width = 2, pad = 10 } = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const pts = points(signal, pad, w - pad, pad, h - pad);
  strokePath(ctx, pts, pts.length, color, width);
}

// ---------- hero strip: real beats, classified live ----------
const strip = { order: [], i: 0, timer: null };

function stripBox(w, h) {
  const narrow = w < 640;
  // Leave room for the readout card on the right (desktop) or bottom (mobile)
  return narrow ? [20, w - 20, 20, h - 205] : [Math.max(24, w * 0.04), w * 0.64, 40, h - 36];
}

function sweep(signal) {
  const canvas = $("stripCanvas");
  const { ctx, w, h } = setupCanvas(canvas);
  const [x0, x1, y0, y1] = stripBox(w, h);
  const pts = points(signal, x0, x1, y0, y1);
  if (reduceMotion) { strokePath(ctx, pts, pts.length, TRACE, 2.6); return; }
  const start = performance.now(), dur = 1400;
  (function frame(now) {
    const t = Math.min(1, (now - start) / dur);
    const n = Math.max(2, Math.round(t * pts.length));
    ctx.clearRect(0, 0, w, h);
    strokePath(ctx, pts, n, TRACE, 2.6);
    const [px, py] = pts[n - 1];
    ctx.fillStyle = TRACE; ctx.beginPath(); ctx.arc(px, py, t < 1 ? 5 : 0, 0, Math.PI * 2); ctx.fill();
    if (t < 1) requestAnimationFrame(frame);
  })(start);
}

async function nextStripBeat() {
  const s = SAMPLES[strip.order[strip.i % strip.order.length]];
  strip.i++;
  try {
    const r = (await callPredict({ json: { sample_id: s.id } })).results[0];
    const c = BY_KEY[r.predicted_label];
    sweep(s.signal);
    $("roWhat").textContent = `Sample beat ${s.id + 1} of ${SAMPLES.length}`;
    $("roCls").textContent = r.predicted_label;
    $("roCls").style.color = c.color;
    $("roConf").textContent = `${r.confidence.toFixed(2)}% confidence`;
    $("roTruth").textContent = `True class: ${s.true_label}` + (s.true_label === r.predicted_label ? ", correct" : ", missed");
  } catch (e) {
    $("roWhat").textContent = "The model server isn't responding";
  }
  if (!reduceMotion) strip.timer = setTimeout(nextStripBeat, 4200);
}

// ---------- API ----------
async function callPredict({ json, form }) {
  const opts = json
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(json) }
    : { method: "POST", body: form };
  const res = await fetch("/api/predict", opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `Request failed with status ${res.status}`);
  return body;
}

function setStatus(busy, error) {
  $("busy").classList.toggle("hidden", !busy);
  $("error").textContent = error || "";
  $("error").classList.toggle("hidden", !error);
}

async function analyze(input) {
  setStatus(true);
  try { render((await callPredict(input)).results); setStatus(false); }
  catch (e) { setStatus(false, e.message); }
}

// ---------- results ----------
function show(i) {
  selected = i;
  const r = results[i], c = BY_KEY[r.predicted_label];
  $("rCls").textContent = r.predicted_label;
  $("rCls").style.color = c.color;
  $("rSub").innerHTML = c.title + (r.true_label ? `. ${r.true_label === r.predicted_label ? "Matches the true class." : `True class: ${r.true_label}.`}` : "");
  $("rConfBox").classList.remove("hidden");
  $("rConf").textContent = `${r.confidence.toFixed(1)}%`;
  drawBeat($("plot"), r.signal, { width: 2.4, pad: 18 });
  const top = r.predicted_label;
  $("probs").innerHTML = CLASSES.map(k => {
    const p = r.probabilities[k.key];
    return `<div class="prob ${k.key === top ? "top" : ""}">
      <span>${k.key}</span>
      <div class="track"><div class="fill" style="width:${p}%;background:${k.color}"></div></div>
      <span class="pct">${p.toFixed(2)}%</span></div>`;
  }).join("");
  document.querySelectorAll("#rows tr").forEach((tr, j) => tr.classList.toggle("sel", j === i));
}

function render(list) {
  results = list;
  const multi = list.length > 1;
  $("batch").classList.toggle("hidden", !multi);
  if (multi) {
    const labelled = list.filter(r => r.true_label);
    const right = labelled.filter(r => r.true_label === r.predicted_label).length;
    $("batchSummary").textContent = `${list.length} beats analyzed.` +
      (labelled.length ? ` ${right} of ${labelled.length} match the true class. Select a row to plot it.` : " Select a row to plot it.");
    $("rows").innerHTML = list.map((r, i) => {
      const c = BY_KEY[r.predicted_label];
      return `<tr data-i="${i}" tabindex="0"><td>${i + 1}</td>
        <td><span class="letter" style="background:${c.color};width:24px;height:24px;font-size:.8rem;margin-right:8px">${c.letter}</span>${r.predicted_label}</td>
        <td>${r.confidence.toFixed(2)}%</td><td>${r.true_label ? `${r.true_label} ${r.true_label === r.predicted_label ? '<span class="ok">correct</span>' : '<span class="miss">missed</span>'}` : "Not given"}</td></tr>`;
    }).join("");
    $("rows").querySelectorAll("tr").forEach(tr => {
      tr.onclick = () => show(+tr.dataset.i);
      tr.onkeydown = (e) => { if (e.key === "Enter") show(+tr.dataset.i); };
    });
  }
  show(0);
  if (window.innerWidth < 960) document.querySelector(".report").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
}

function uploadFrom(input) {
  const f = input.files[0];
  if (!f) return;
  const form = new FormData(); form.append("file", f);
  analyze({ form });
  input.value = "";
}

// ---------- sections built from data ----------
function buildSamplePicker() {
  $("sampleRows").innerHTML = CLASSES.map(c => {
    const beats = SAMPLES.filter(s => s.true_label === c.key);
    return `<div class="class-row">
      <div class="class-name"><span class="letter" style="background:${c.color}">${c.letter}</span>${c.key}</div>
      <div class="thumbs">${beats.map(s =>
        `<button class="thumb" type="button" data-id="${s.id}" aria-label="Sample beat ${s.id + 1}, true class ${c.key}"><canvas></canvas></button>`).join("")}</div>
    </div>`;
  }).join("");
  document.querySelectorAll(".thumb").forEach(btn => {
    const s = SAMPLES[+btn.dataset.id];
    drawBeat(btn.querySelector("canvas"), s.signal, { color: BY_KEY[s.true_label].color, width: 1.4, pad: 5 });
    btn.onclick = () => {
      document.querySelectorAll(".thumb").forEach(b => b.classList.toggle("on", b === btn));
      analyze({ json: { sample_id: s.id } });
    };
  });
  $("allBtn").onclick = () => {
    document.querySelectorAll(".thumb").forEach(b => b.classList.remove("on"));
    const csv = SAMPLES.map(s => [...s.signal, s.true_class].join(",")).join("\n");
    const form = new FormData(); form.append("file", new Blob([csv], { type: "text/csv" }), "samples.csv");
    analyze({ form });
  };
}

function buildClassList() {
  $("classList").innerHTML = CLASSES.map(c => `
    <div class="cl">
      <div class="glyph" style="color:${c.color}">${c.letter}</div>
      <div><h3>${c.title}</h3><p>${c.desc}</p></div>
      <div class="ex paper"><canvas data-key="${c.key}"></canvas></div>
    </div>`).join("");
  document.querySelectorAll("#classList canvas").forEach(cv => {
    const s = SAMPLES.find(x => x.true_label === cv.dataset.key);
    if (s) drawBeat(cv, s.signal, { width: 1.8, pad: 8 });
  });
}

// Real layer output shapes of ecg_cnn_bilstm.keras (time steps x features).
// BatchNorm and Dropout layers keep the same shape, so they're not drawn separately.
const LAYERS = [
  ["Input", 187, 1, "in"], ["Conv1D", 187, 64, "cnn"], ["MaxPool", 93, 64, "cnn"],
  ["Conv1D", 93, 128, "cnn"], ["MaxPool", 46, 128, "cnn"], ["Conv1D", 46, 128, "cnn"],
  ["MaxPool", 23, 128, "cnn"], ["BiLSTM", 23, 128, "rnn"], ["Avg pool", 1, 128, "rnn"],
  ["Dense", 1, 64, "out"], ["Softmax", 1, 5, "out"],
];

function buildLayers() {
  $("layers").innerHTML = LAYERS.map(([name, t, f, kind]) => `
    <div class="layer ${kind}">
      <div class="shape" style="height:${Math.max(6, t / 187 * 190)}px;width:${Math.max(10, f / 128 * 100)}%"></div>
      <b>${name}</b><span>${t > 1 ? `${t} × ${f}` : f}</span>
    </div>`).join("");
}

// ---------- devops: live deployment details ----------
async function loadDeployInfo() {
  try {
    const h = await (await fetch("/health")).json();
    const local = h.build_number === "dev";
    $("dBuild").textContent = local ? "Not built by Jenkins" : `#${h.build_number}`;
    $("dCommit").textContent = h.git_commit;
    $("dPod").textContent = h.pod;
    $("dModel").textContent = h.model_loaded ? "Loaded" : "Not loaded";
    $("dModel").classList.toggle("good", h.model_loaded);
    $("deployNote").textContent = local
      ? "You are viewing ECGense started directly with Python, so it has not gone through the pipeline yet. Once Jenkins deploys it to Kubernetes, this panel shows the Jenkins build number, the Git commit it was built from, and the name of the pod that served this page."
      : "This page was served by a Kubernetes pod that Jenkins deployed. Push a new commit to GitHub and, after the pipeline finishes, the build number here goes up.";
  } catch (e) {
    $("deployNote").textContent = "The server's health check isn't responding.";
  }
}

// ---------- start ----------
$("csvFile").onchange = (e) => uploadFrom(e.target);
$("imgFile").onchange = (e) => uploadFrom(e.target);
drawBeat($("plot"), []);
buildLayers();
loadDeployInfo();

fetch("/api/samples").then(r => r.json()).then(samples => {
  SAMPLES = samples;
  buildSamplePicker();
  buildClassList();
  // Alternate classes on the strip: N, S, V, F, Q, N, S, ...
  for (let k = 0; k < 4; k++) for (let c = 0; c < 5; c++) strip.order.push(c * 4 + k);
  nextStripBeat();
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    buildClassList();
    document.querySelectorAll(".thumb").forEach(btn => {
      const s = SAMPLES[+btn.dataset.id];
      drawBeat(btn.querySelector("canvas"), s.signal, { color: BY_KEY[s.true_label].color, width: 1.4, pad: 5 });
    });
    if (results.length) show(selected);
  }, 200);
});
