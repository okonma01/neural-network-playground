const $ = (id) => document.getElementById(id);

const els = {
  heatmap: $("heatmap"), scatter: $("scatter"), lossChart: $("lossChart"), netViz: $("netViz"),
  shape: $("shape"), density: $("density"), speed: $("speed"), maxEpochs: $("maxEpochs"),
  layers: $("layers"), activation: $("activation"), showPoints: $("showPoints"), lr: $("lr"),
  batch: $("batch"), opt: $("opt"), train: $("train"), pause: $("pause"), reset: $("reset"),
  epoch: $("epoch"), loss: $("loss"), progress: $("progress"), densityOut: $("densityOut"),
  speedOut: $("speedOut"), doneBanner: $("doneBanner")
};

const ctx = {
  hm: els.heatmap.getContext("2d", { willReadFrequently: true }),
  sc: els.scatter.getContext("2d"),
  lc: els.lossChart.getContext("2d"),
  nv: els.netViz.getContext("2d")
};

const SHAPES = {
  circle: {
    label: "Circle",
    inside: (x, y) => x * x + y * y <= 0.46,
    boundary: (t) => [Math.cos(t) * 0.68, Math.sin(t) * 0.68]
  },
  heart: {
    label: "Heart",
    inside: (x, y) => {
      x *= 1.25; y *= 1.25;
      const a = x * x + y * y - 1;
      return a * a * a - x * x * y * y * y <= 0;
    },
    boundary: (t) => {
      const x = 16 * Math.sin(t) ** 3;
      const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
      return [x / 18, y / 18];
    }
  },
  star: {
    label: "Star",
    poly: (() => {
      const p = [];
      for (let i = 0; i < 10; i++) {
        const r = i % 2 ? 0.32 : 0.74;
        const a = -Math.PI / 2 + i * Math.PI / 5;
        p.push([Math.cos(a) * r, Math.sin(a) * r]);
      }
      return p;
    })(),
    boundary: (t) => {
      const poly = SHAPES.star.poly;
      const u = (t / (2 * Math.PI)) * poly.length;
      const i = Math.floor(u) % poly.length;
      const f = u - i;
      const a = poly[i], b = poly[(i + 1) % poly.length];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    }
  },
  car: {
    label: "Car Silhouette",
    poly: [[-0.82, 0.2], [-0.64, 0.04], [-0.35, -0.14], [0.32, -0.14], [0.55, -0.03], [0.78, 0.12], [0.82, 0.3], [0.66, 0.31], [0.56, 0.17], [0.42, 0.17], [0.28, 0.3], [-0.25, 0.3], [-0.44, 0.17], [-0.62, 0.17], [-0.72, 0.3]],
    boundary: (t) => {
      const poly = SHAPES.car.poly;
      const u = (t / (2 * Math.PI)) * poly.length;
      const i = Math.floor(u) % poly.length;
      const f = u - i;
      const a = poly[i], b = poly[(i + 1) % poly.length];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    }
  }
};

const pointInPoly = (x, y, poly) => {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    const hit = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi;
    if (hit) inside = !inside;
  }
  return inside;
};

const rand = (a, b) => Math.random() * (b - a) + a;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

let state = {
  model: null, x: null, y: null, points: [], losses: [], running: false,
  epoch: 0, maxEpochs: 700, speed: 6, showPoints: false, confettiAt: 0
};

const toPx = (x, y, w, h) => [((x + 1) * 0.5) * w, ((1 - (y + 1) * 0.5) * h)];

function inside(shape, x, y) {
  return shape.inside ? shape.inside(x, y) : pointInPoly(x, y, shape.poly);
}

function sampleData(shape, n = 1000) {
  const pos = Math.floor(n / 2), neg = n - pos;
  const pts = [];

  let i = 0;
  while (i < pos) {
    const x = rand(-1, 1), y = rand(-1, 1);
    if (inside(shape, x, y)) {
      pts.push([x, y, 1]);
      i++;
    }
  }

  let j = 0;
  while (j < neg) {
    const x = rand(-1, 1), y = rand(-1, 1);
    if (!inside(shape, x, y)) {
      pts.push([x, y, 0]);
      j++;
    }
  }

  // Add a boundary stripe so the model sees the edge explicitly.
  for (let k = 0; k < Math.floor(n * 0.12); k++) {
    const [bx, by] = shape.boundary((k / Math.max(1, Math.floor(n * 0.12))) * Math.PI * 2);
    pts.push([bx, by, 1]);
  }

  for (let k = pts.length - 1; k > 0; k--) {
    const r = Math.floor(Math.random() * (k + 1));
    [pts[k], pts[r]] = [pts[r], pts[k]];
  }
  return pts;
}

function activationFn(name) {
  if (name === "sin") return (x) => tf.sin(x);
  return name;
}

function makeModel() {
  state.model?.dispose();
  const m = tf.sequential();
  const layers = els.layers.value.split(",").map((s) => parseInt(s.trim(), 10)).filter((v) => v > 0);
  const act = activationFn(els.activation.value);

  layers.forEach((units, i) => {
    m.add(tf.layers.dense({ units, activation: act, inputShape: i ? undefined : [2] }));
  });
  if (!layers.length) m.add(tf.layers.dense({ units: 8, activation: act, inputShape: [2] }));
  m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  const lr = parseFloat(els.lr.value);
  const optName = els.opt.value;
  const optimizer = optName === "sgd" ? tf.train.sgd(lr) : optName === "rmsprop" ? tf.train.rmsprop(lr) : tf.train.adam(lr);

  m.compile({ optimizer, loss: "binaryCrossentropy" });
  state.model = m;
  drawNetViz();
}

function rebuildData() {
  const shape = SHAPES[els.shape.value];
  state.points = sampleData(shape, parseInt(els.density.value, 10));
  const xs = state.points.map((p) => [p[0], p[1]]);
  const ys = state.points.map((p) => [p[2]]);
  state.x?.dispose(); state.y?.dispose();
  state.x = tf.tensor2d(xs);
  state.y = tf.tensor2d(ys);
  drawScatter();
}

async function trainSteps() {
  if (!state.running || state.epoch >= state.maxEpochs) return;
  for (let i = 0; i < state.speed && state.epoch < state.maxEpochs; i++) {
    const h = await state.model.fit(state.x, state.y, {
      epochs: 1,
      batchSize: parseInt(els.batch.value, 10),
      shuffle: true,
      verbose: 0
    });
    const loss = h.history.loss[0];
    state.losses.push(loss);
    if (state.losses.length > 180) state.losses.shift();
    state.epoch++;
    updateStats(loss);
  }

  await drawHeatmap();
  drawLoss();

  if (state.running && state.epoch < state.maxEpochs) requestAnimationFrame(trainSteps);
  if (state.epoch >= state.maxEpochs) {
    state.running = false;
    state.confettiAt = performance.now();
    els.doneBanner.hidden = false;
    flashCelebrate();
  }
}

async function drawHeatmap() {
  const w = els.heatmap.width, h = els.heatmap.height;
  const step = 5;
  const grid = [];
  for (let py = 0; py < h; py += step) {
    const y = 1 - (py / (h - 1)) * 2;
    for (let px = 0; px < w; px += step) {
      const x = (px / (w - 1)) * 2 - 1;
      grid.push([x, y]);
    }
  }

  const probs = await tf.tidy(() => state.model.predict(tf.tensor2d(grid)).data());
  const image = ctx.hm.createImageData(w, h);
  let idx = 0;
  for (let py = 0; py < h; py += step) {
    for (let px = 0; px < w; px += step) {
      const p = clamp(probs[idx++], 0, 1);
      const r = Math.floor(69 + 190 * p);
      const g = Math.floor(198 - 88 * p);
      const b = Math.floor(231 - 120 * p);
      for (let yy = 0; yy < step; yy++) {
        for (let xx = 0; xx < step; xx++) {
          const x = px + xx, y = py + yy;
          if (x >= w || y >= h) continue;
          const o = (y * w + x) * 4;
          image.data[o] = r; image.data[o + 1] = g; image.data[o + 2] = b; image.data[o + 3] = 255;
        }
      }
    }
  }
  ctx.hm.putImageData(image, 0, 0);
}

function drawScatter() {
  const c = ctx.sc, w = els.scatter.width, h = els.scatter.height;
  c.clearRect(0, 0, w, h);
  if (!state.showPoints) return;
  c.globalAlpha = 0.65;
  for (const [x, y, lbl] of state.points) {
    const [px, py] = toPx(x, y, w, h);
    c.fillStyle = lbl ? "#ef5f66" : "#23c7df";
    c.fillRect(px, py, 2, 2);
  }
  c.globalAlpha = 1;
}

function drawLoss() {
  const c = ctx.lc, w = els.lossChart.width, h = els.lossChart.height;
  c.clearRect(0, 0, w, h);
  c.strokeStyle = "#d8e8f0"; c.strokeRect(0.5, 0.5, w - 1, h - 1);
  if (state.losses.length < 2) return;
  const lo = Math.min(...state.losses), hi = Math.max(...state.losses), span = Math.max(1e-6, hi - lo);
  c.beginPath();
  state.losses.forEach((v, i) => {
    const x = (i / (state.losses.length - 1)) * (w - 14) + 7;
    const y = h - 8 - ((v - lo) / span) * (h - 16);
    i ? c.lineTo(x, y) : c.moveTo(x, y);
  });
  c.lineWidth = 2;
  c.strokeStyle = "#ff6f78";
  c.stroke();
}

function drawNetViz() {
  const c = ctx.nv, w = els.netViz.width, h = els.netViz.height;
  c.clearRect(0, 0, w, h);
  const hidden = els.layers.value.split(",").map((s) => parseInt(s.trim(), 10)).filter((v) => v > 0);
  const cols = [2, ...hidden, 1];
  const gx = cols.length > 1 ? (w - 30) / (cols.length - 1) : 0;

  c.strokeStyle = "#bfd3de";
  for (let i = 0; i < cols.length - 1; i++) {
    const a = cols[i], b = cols[i + 1];
    for (let ai = 0; ai < Math.min(8, a); ai++) {
      for (let bi = 0; bi < Math.min(8, b); bi++) {
        const ax = 15 + i * gx, bx = 15 + (i + 1) * gx;
        const ay = h * ((ai + 1) / (Math.min(8, a) + 1));
        const by = h * ((bi + 1) / (Math.min(8, b) + 1));
        c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
      }
    }
  }

  cols.forEach((n, i) => {
    const x = 15 + i * gx;
    for (let j = 0; j < Math.min(8, n); j++) {
      const y = h * ((j + 1) / (Math.min(8, n) + 1));
      c.beginPath(); c.arc(x, y, 5, 0, Math.PI * 2);
      c.fillStyle = i === 0 ? "#22c8de" : i === cols.length - 1 ? "#ff6f78" : "#f8bd48";
      c.fill();
      c.strokeStyle = "#fff"; c.stroke();
    }
  });
}

function updateStats(loss) {
  els.epoch.textContent = String(state.epoch);
  els.loss.textContent = Number(loss).toFixed(4);
  els.progress.max = state.maxEpochs;
  els.progress.value = state.epoch;
}

function resetRun(full = false) {
  state.running = false;
  state.epoch = 0;
  state.losses = [];
  els.doneBanner.hidden = true;
  updateStats(NaN);
  els.loss.textContent = "-";
  if (full) makeModel();
  rebuildData();
  drawHeatmap();
  drawLoss();
}

function flashCelebrate() {
  const c = ctx.sc, w = els.scatter.width, h = els.scatter.height;
  const t = performance.now() - state.confettiAt;
  if (t > 700) {
    drawScatter();
    return;
  }
  drawScatter();
  c.save();
  c.globalAlpha = 0.16 * (1 - t / 700);
  c.fillStyle = "#ffd35d";
  c.fillRect(0, 0, w, h);
  c.restore();
  requestAnimationFrame(flashCelebrate);
}

function bind() {
  Object.entries(SHAPES).forEach(([key, s]) => {
    const o = document.createElement("option");
    o.value = key; o.textContent = s.label;
    els.shape.appendChild(o);
  });

  const sync = () => {
    els.densityOut.textContent = els.density.value;
    els.speedOut.textContent = `${els.speed.value}x`;
    state.speed = parseInt(els.speed.value, 10);
    state.maxEpochs = parseInt(els.maxEpochs.value, 10);
    state.showPoints = els.showPoints.checked;
    drawScatter();
  };

  ["density", "speed", "maxEpochs", "showPoints"].forEach((id) => els[id].addEventListener("input", sync));

  els.shape.addEventListener("change", () => resetRun(true));
  els.layers.addEventListener("change", () => resetRun(true));
  ["activation", "lr", "batch", "opt"].forEach((id) => els[id].addEventListener("change", () => resetRun(true)));
  els.density.addEventListener("change", () => resetRun(false));

  els.train.addEventListener("click", () => {
    if (state.running) return;
    state.running = true;
    els.doneBanner.hidden = true;
    trainSteps();
  });

  els.pause.addEventListener("click", () => { state.running = false; });
  els.reset.addEventListener("click", () => resetRun(true));

  sync();
}

async function boot() {
  bind();
  makeModel();
  rebuildData();
  await drawHeatmap();
  drawLoss();
}

boot();
