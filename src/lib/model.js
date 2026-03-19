// Pure-JS micro neural network engine — Adam optimizer, no external dependencies.
// Architecture mirrors the original TF Playground nn.ts approach: plain arrays + for-loops.

const gridCache = new Map();

const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS = 1e-8;

// --- Activations ---

function activate(z, name) {
  switch (name) {
    case "relu": return z > 0 ? z : 0;
    case "sigmoid": return 1 / (1 + Math.exp(-z));
    case "tanh": return Math.tanh(z);
    case "sin": return Math.sin(z);
    default: return z > 0 ? z : 0;
  }
}

// Derivative of activation w.r.t. pre-activation value z
function dActivate(z, name) {
  switch (name) {
    case "relu": return z > 0 ? 1 : 0;
    case "sigmoid": { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }
    case "tanh": { const t = Math.tanh(z); return 1 - t * t; }
    case "sin": return Math.cos(z);
    default: return z > 0 ? 1 : 0;
  }
}

// --- Layer helpers ---

function xavierInit(fanIn, fanOut) {
  const std = Math.sqrt(2 / (fanIn + fanOut));
  return Array.from({ length: fanIn }, () =>
    Array.from({ length: fanOut }, () => (Math.random() * 2 - 1) * std)
  );
}

function zeros2d(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function zeros1d(size) {
  return new Array(size).fill(0);
}

function createLayer(inputSize, outputSize) {
  return {
    W: xavierInit(inputSize, outputSize),
    b: zeros1d(outputSize),
    mW: zeros2d(inputSize, outputSize), // Adam first moment — weights
    vW: zeros2d(inputSize, outputSize), // Adam second moment — weights
    mb: zeros1d(outputSize),            // Adam first moment — biases
    vb: zeros1d(outputSize),            // Adam second moment — biases
    inputSize,
    outputSize
  };
}

// --- Model construction ---

export function buildModel(layerSpec, activations, options) {
  const layers = [];
  const layerActivations = [];
  let prevSize = 2; // inputs are always [x, y]

  layerSpec.forEach((units, index) => {
    layers.push(createLayer(prevSize, units));
    layerActivations.push(activations[index] ?? activations[activations.length - 1] ?? "relu");
    prevSize = units;
  });

  // Output layer — sigmoid applied inline; gradient merged with BCE in backprop
  layers.push(createLayer(prevSize, 1));

  return {
    layers,
    activations: layerActivations,
    lr: options.learningRate ?? 0.01,
    t: 0,   // Adam global timestep (advances once per batch update)
    dispose() {}
  };
}

// --- Forward pass storing pre-activations + outputs (needed for backprop) ---
// as[0] = raw input; as[i+1] = post-activation of layer i; as[L] = prediction
// zs[i]  = pre-activation of layer i
function forward(model, x) {
  const { layers, activations } = model;
  const zs = [];
  const as = [x];

  for (let i = 0; i < layers.length - 1; i += 1) {
    const layer = layers[i];
    const z = zeros1d(layer.outputSize);
    const input = as[i];
    for (let j = 0; j < layer.outputSize; j += 1) {
      let sum = layer.b[j];
      for (let k = 0; k < layer.inputSize; k += 1) sum += input[k] * layer.W[k][j];
      z[j] = sum;
    }
    zs.push(z);
    as.push(z.map((val) => activate(val, activations[i])));
  }

  // Output layer with inline sigmoid
  const outLayer = layers[layers.length - 1];
  const zOut = zeros1d(1);
  const inputToOut = as[as.length - 1];
  let sum = outLayer.b[0];
  for (let k = 0; k < outLayer.inputSize; k += 1) sum += inputToOut[k] * outLayer.W[k][0];
  zOut[0] = sum;
  zs.push(zOut);
  as.push([1 / (1 + Math.exp(-sum))]);

  return { zs, as };
}

// --- Lean prediction (no stored state) ---
function forwardPredict(model, x) {
  const { layers, activations } = model;
  let current = x;

  for (let i = 0; i < layers.length - 1; i += 1) {
    const layer = layers[i];
    const next = zeros1d(layer.outputSize);
    for (let j = 0; j < layer.outputSize; j += 1) {
      let s = layer.b[j];
      for (let k = 0; k < layer.inputSize; k += 1) s += current[k] * layer.W[k][j];
      next[j] = activate(s, activations[i]);
    }
    current = next;
  }

  const outLayer = layers[layers.length - 1];
  let s = outLayer.b[0];
  for (let k = 0; k < outLayer.inputSize; k += 1) s += current[k] * outLayer.W[k][0];
  return 1 / (1 + Math.exp(-s));
}

// --- Mini-batch: accumulate gradients then apply a single Adam step ---
function trainBatch(model, batchXs, batchYs) {
  const { layers, activations, lr } = model;
  const n = batchXs.length;
  model.t += 1;
  const t = model.t;

  const dWs = layers.map((l) => zeros2d(l.inputSize, l.outputSize));
  const dbs = layers.map((l) => zeros1d(l.outputSize));
  let batchLoss = 0;

  for (let s = 0; s < n; s += 1) {
    const { zs, as } = forward(model, batchXs[s]);
    const pred = as[as.length - 1][0];
    const y = batchYs[s];
    batchLoss += -(y * Math.log(pred + 1e-7) + (1 - y) * Math.log(1 - pred + 1e-7));

    // BCE + sigmoid merged gradient: dL/dz_out = pred − y
    let delta = [pred - y];

    for (let i = layers.length - 1; i >= 0; i -= 1) {
      const layer = layers[i];
      const input = as[i];

      for (let j = 0; j < layer.outputSize; j += 1) {
        dbs[i][j] += delta[j];
        for (let k = 0; k < layer.inputSize; k += 1) dWs[i][k][j] += input[k] * delta[j];
      }

      if (i > 0) {
        const prevDelta = zeros1d(layer.inputSize);
        for (let k = 0; k < layer.inputSize; k += 1) {
          let g = 0;
          for (let j = 0; j < layer.outputSize; j += 1) g += layer.W[k][j] * delta[j];
          prevDelta[k] = g * dActivate(zs[i - 1][k], activations[i - 1]);
        }
        delta = prevDelta;
      }
    }
  }

  // Adam update with averaged gradients
  const bc1 = 1 - Math.pow(BETA1, t);
  const bc2 = 1 - Math.pow(BETA2, t);

  for (let i = 0; i < layers.length; i += 1) {
    const layer = layers[i];
    for (let k = 0; k < layer.inputSize; k += 1) {
      for (let j = 0; j < layer.outputSize; j += 1) {
        const g = dWs[i][k][j] / n;
        layer.mW[k][j] = BETA1 * layer.mW[k][j] + (1 - BETA1) * g;
        layer.vW[k][j] = BETA2 * layer.vW[k][j] + (1 - BETA2) * g * g;
        layer.W[k][j] -= lr * (layer.mW[k][j] / bc1) / (Math.sqrt(layer.vW[k][j] / bc2) + EPS);
      }
    }
    for (let j = 0; j < layer.outputSize; j += 1) {
      const g = dbs[i][j] / n;
      layer.mb[j] = BETA1 * layer.mb[j] + (1 - BETA1) * g;
      layer.vb[j] = BETA2 * layer.vb[j] + (1 - BETA2) * g * g;
      layer.b[j] -= lr * (layer.mb[j] / bc1) / (Math.sqrt(layer.vb[j] / bc2) + EPS);
    }
  }

  return batchLoss / n;
}

// --- Public API ---

export async function trainStep(model, xs, ys, config) {
  const n = xs.length;
  const batchSize = config.batchSize ?? 128;

  // Fisher-Yates shuffle of indices
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  let totalLoss = 0;
  let numBatches = 0;
  for (let start = 0; start < n; start += batchSize) {
    const end = Math.min(start + batchSize, n);
    const batchXs = [];
    const batchYs = [];
    for (let idx = start; idx < end; idx += 1) {
      batchXs.push(xs[indices[idx]]);
      batchYs.push(ys[indices[idx]]);
    }
    totalLoss += trainBatch(model, batchXs, batchYs);
    numBatches += 1;
  }

  return totalLoss / numBatches;
}

export function getWeights(model) {
  return model.layers.map((layer) => ({
    weights: layer.W,
    biases: layer.b
  }));
}

export async function predictGrid(model, resolution) {
  let samples = gridCache.get(resolution);
  if (!samples) {
    samples = [];
    for (let y = 0; y < resolution; y += 1) {
      const yVal = 1 - (y / (resolution - 1)) * 2;
      for (let x = 0; x < resolution; x += 1) {
        samples.push([(x / (resolution - 1)) * 2 - 1, yVal]);
      }
    }
    gridCache.set(resolution, samples);
  }

  return samples.map((pt) => forwardPredict(model, pt));
}