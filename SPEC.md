# Neural Network Playground — Full Specification

> Generated from the complete planning conversation.  
> Covers all Q&A, design decisions, and the exact build contract.

---

## Table of Contents

1. [Project Concept](#1-project-concept)
2. [Full Q&A Record](#2-full-qa-record)
3. [UI Alternatives Explored](#3-ui-alternatives-explored)
4. [Final Technical Decisions](#4-final-technical-decisions)
5. [What Will Be Built — Exact Spec](#5-what-will-be-built--exact-spec)
6. [SVG Shapes Supplied](#6-svg-shapes-supplied)
7. [Where to Find More Shapes](#7-where-to-find-more-shapes)

---

## 1. Project Concept

A fully in-browser web app that illustrates the idea that **neural networks can learn (almost) anything** — specifically, that they are not just *function* approximators but *relation* approximators.

A small feedforward neural network trains live, before your eyes, to learn a 2D shape. Given any point (x, y), it predicts:

- **1** if the point is inside the shape or on its boundary
- **0** if it is outside

The network is not doing classical ML (no train/test split, no generalization). It is purely memorising a geometric relation — and visually, you watch a heatmap morph into the target shape as training progresses.

The educational goal: show that the vertical line test does not constrain what a network can learn. A heart, a teddy bear, a flame — none of these are functions. All of them can be learned.

---

## 2. Full Q&A Record

### Round 1 — Core Setup

| Question | Answer |
|---|---|
| 2D vs 3D to start? | Start with 2D, then add 3D later |
| What the network learns — binary classifier, SDF, or generative? | *"I'm not sure yet — how does the decision here affect what gets built?"* (→ explained; settled on binary classifier) |
| Tech stack? | Vanilla JS + HTML/CSS _(later revised to React + Vite — see Round 5)_ |
| Training runs in-browser or on a server? | Fully in-browser (TensorFlow.js) |
| Network configurability? | *"Simplified (change neurons/layers) but optional toggle for 'advanced' with more control on hyperparameters"* |
| Shape categories? | Mathematical/geometric ✓ · Cute/recognisable real-world objects ✓ · Simple 2D curves ✓ |
| Network architecture panel? | Optional / collapsible |
| Educational angle? | *"All of the above but mostly (1) neural nets are universal relation approximators and (2) vertical line test / functions vs. relations"* |

---

### Round 2 — Classifier Framing & Visuals

| Question | Answer |
|---|---|
| Binary classifier, SDF, or generative? | **Binary classifier** — heatmap morphs into the learned shape |
| What does the main canvas show? | *"Both heatmap and scatter, with the option to hide or toggle visibility of each"* |
| Training speed | User controls speed with a slider |
| Loss display | Both — a live number and a small chart |
| UI aesthetic | *"Colorful, playful, but informative"* |
| Deployment | Decide later |

---

### Round 3 — Network & UX Details

| Question | Answer |
|---|---|
| Activation functions available? | User picks per layer: ReLU, Sigmoid, Tanh, Sin |
| Default shapes to ship with? | *"Mathematical objects, but more importantly 2D representations/silhouettes of real-life things like cars, toys, etc. A small set"* |
| When training completes? | *"Celebrate with a quick, mild animation, then offer to start again"* |
| Mobile support? | *"Full mobile support (responsive design) — but only for modern smart mobile devices"* _(later revised to desktop-first)_ |
| Sharing / saving? | No sharing needed in v1 |
| Educational tooltips? | Yes — hover over any term for an explanation |

---

### Round 4 — Shape Pipeline Details

| Question | Answer |
|---|---|
| How are real-object 2D shapes sourced? | *"Find open-source SVG/clip art — it has to be good quality"* |
| Point sampling: boundary only, or inside too? | Boundary only _(later revised by user request: **inside + boundary = 1, outside = 0**)_ |
| Number of training points? | *"User can control density within a bounded slider"* |
| Default canvas view? | Pure heatmap; scatter points hidden unless toggled on |

---

### Round 5 — Planning Horizon

| Question | Answer |
|---|---|
| Include a 3D phase? | No — 2D might be enough; no 3D planned |
| App name? | Neural Network Playground |
| Intro / onboarding? | Inline intro text at the top of the app |
| When does training stop? | Auto-stop at a user-configurable max epoch count |

---

### Round 6 — UI Aesthetic (Three Alternatives)

Three fully distinct layouts were designed and prototyped:

| Alt | Name | Structure |
|---|---|---|
| **A** | Split Lab | Canvas left half · Controls right half · Bright white · Lab notebook feel |
| B | Mission Deck | Canvas hero center-top · Three pastel cards below (loss / network / controls) · Light gray |
| C | Story Scroll | Single column · Intro → full-width canvas → horizontal control ribbon → insight row |

**User verdict:** *"I prefer A. I would like to get rid of B and C."*  
→ A and B were archived in `alternatives/`; C was discarded.

---

### Round 7 — Framework & Interactivity Upgrade

After reviewing the vanilla JS prototype, the user identified key gaps:

> *"I don't think vanilla JavaScript is enough to achieve what I want. I don't just want to define the activation and the layers in some text box, I want an actual (playful) 'rendering' of the neural net, where I can hover over the parameters and see their values, where there are +/− buttons to add nodes/layers. Also, the window showing the trained boundary is too big and doesn't allow for space for other important elements."*

| Question | Answer |
|---|---|
| Delete alternatives B and C? | Yes |
| Framework? | *"nano-react-app"* (= React + Vite) |
| Build tooling? | Vite |
| Net diagram hover behaviour? | *"Hover tooltip on weights — also, add explicitly the bias parameter in the figure"* |
| Node/layer editing UX? | + / − buttons at the top/bottom of each column of nodes |
| Canvas vs diagram size balance? | Decide after seeing it live |
| Where to source shapes? | User browses open-source SVG sites themselves |

---

### Round 8 — React + Diagram Details

| Question | Answer |
|---|---|
| Confirm React + Vite scaffold? | Yes |
| Net diagram renderer? | HTML/CSS + SVG (no external graph library) |
| Node visual encoding? | All nodes same colour; **size scales with magnitude of total incoming weights** |
| SVG shape type to look for? | Filled silhouettes (black/solid, no internal detail — "sticker-style") |
| Keep placeholder shapes? | No — remove them; user will supply their own SVGs |
| How many shapes in v1? | 6–8 |
| Shape picker UI? | Small thumbnail cards showing a preview of the sampled target points |
| Bias node in diagram? | Explicit separate node per layer (constant value 1), with edges to every neuron in the next layer |
| Weight animation during training? | Edge **thickness pulses** as weights update |
| Mobile support? | Desktop-first for now |

---

### Round 9 — Late Additions

| Item | Decision |
|---|---|
| SVGs supplied by user | Placed in `/svg` folder — 5 shapes: alien, bluetooth, flame, light bulb, teddy bear |
| Pause button | Should **toggle between Play ▶ and Pause ⏸** depending on whether training is actively running |
| Canvas heatmap | Remains in left pane as agreed |

---

## 3. UI Alternatives Explored

### Alternative A — Split Lab ✅ SELECTED

```
┌──────────────────────────────────────────────────────┐
│  Neural Network Playground           [intro tagline] │
├──────────────────────┬───────────────────────────────┤
│                      │  [Shape picker cards]         │
│                      │  ─────────────────────────── │
│   HEATMAP CANVAS     │  [Interactive SVG Net Diagram]│
│   (left pane)        │   • bias nodes                │
│                      │   • +/− node/layer buttons    │
│                      │   • hover tooltips            │
│                      │  ─────────────────────────── │
│                      │  [Controls: speed, density,   │
│                      │   epochs, train/play/reset]   │
│                      │  [Loss chart + epoch counter] │
└──────────────────────┴───────────────────────────────┘
```

- Background: bright white with subtle coloured radial gradients
- Controls panel: soft rounded cards with coral/teal/yellow accents
- Feels like a lab notebook split open

### Alternative B — Mission Deck (archived, not used)

Canvas hero at top, three pastel-coloured cards (mint/lavender/peach) below in a row. Bold colorful-infographic-poster feel.

### Alternative C — Story Scroll (archived, not used)

Single-column narrative: intro text → full-width canvas → horizontal control ribbon → two-column insight row (loss left, network right). Warm off-white / light lavender background.

---

## 4. Final Technical Decisions

| Dimension | Decision |
|---|---|
| Framework | **React + Vite** (nano-react-app style) |
| Training engine | **TensorFlow.js** (fully in-browser, no server) |
| Net diagram | **SVG rendered in React** — no external graph library |
| Classifier type | **Binary**: inside-or-boundary = 1, outside = 0 |
| Labelling rule | Interior + boundary → positive; exterior → negative; 50/50 balance |
| Layout | **Alternative A (Split Lab)**: canvas left, diagram+controls right |
| Heatmap | HTML5 Canvas `putImageData` for pixel-level speed |
| Scatter overlay | Togglable, default hidden |
| Shape source | User-supplied **filled-silhouette SVGs** in `/svg` folder |
| Shape picker | Thumbnail card grid (6–8 shapes), each card previews the sampled scatter |
| SVG parser | Extract all `<path>` elements → sample boundary + interior via ray-casting point-in-polygon |
| Normalisation | All shapes normalised to [-1, 1] bounding box |
| Point density | User slider (200–2400 points) |
| Activations | Per-layer dropdown: ReLU · Tanh · Sigmoid · Sin |
| Advanced settings | Learning rate · Batch size · Optimizer (Adam / SGD / RMSProp) — behind a toggle |
| Max epochs | User-configurable (default 700); auto-stops |
| Play/Pause | Single toggle button: shows **▶ Play** when paused, **⏸ Pause** when training |
| Completion | Mild glow/confetti animation, then prompt to try another shape |
| Tooltips | Hover on any technical term → plain-English explanation |
| Mobile | Desktop-first for now |
| Deployment | TBD (static, GitHub Pages, or Vercel all viable) |
| 3D | Out of scope for v1 |
| Sharing/export | Out of scope for v1 |

---

## 5. What Will Be Built — Exact Spec

### File Structure

```
neural-network-playground/
├── index.html              ← Vite entry
├── vite.config.js
├── package.json
├── svg/                    ← user-supplied filled-silhouette SVGs
│   ├── alien-svgrepo-com.svg
│   ├── bluetooth-svgrepo-com.svg
│   ├── flame-fire-svgrepo-com.svg
│   ├── miniature-light-bulb-svgrepo-com.svg
│   └── teddy-bear-svgrepo-com.svg
└── src/
    ├── main.jsx            ← React root mount
    ├── App.jsx             ← Split Lab layout shell
    ├── App.css
    ├── components/
    │   ├── HeatmapCanvas.jsx   ← left pane: heatmap + scatter canvases
    │   ├── ShapePicker.jsx     ← thumbnail card grid
    │   ├── NetDiagram.jsx      ← interactive SVG network diagram
    │   ├── Controls.jsx        ← sliders, train/play/reset, loss chart
    │   └── Tooltip.jsx         ← reusable hover tooltip
    └── lib/
        ├── model.js            ← TF.js model builder + training loop
        └── shapes.js           ← SVG parser + point samplers
```

---

### Component Responsibilities

#### `App.jsx`
- Holds top-level state: `netSpec` (layers/activations), `shape`, `trainingState`, `weights`
- Renders the Split Lab two-pane layout
- Passes state down; training dispatch goes up via callbacks

#### `HeatmapCanvas.jsx`
- Stacked canvases: heatmap (bottom) + scatter (top)
- Every N training steps: evaluate network on 100×100 grid → `putImageData` with a cool (outside) → warm (inside) colour gradient
- Scatter overlay: shows positive (inside) and negative (outside) training points; togglable via a button; hidden by default

#### `ShapePicker.jsx`
- Card grid of all shapes loaded from `/svg`
- Each card renders a tiny canvas showing the sampled target scatter preview
- Selected shape is highlighted; clicking a card resets and rebuilds training data

#### `NetDiagram.jsx`
- SVG component; re-renders whenever `netSpec` or `weights` change
- **Layers rendered left → right**: input (x, y) · hidden layers · output
- **Bias node per layer**: distinct colour (e.g. amber), constant label "1", edges drawn to every neuron in the next layer
- **Edges**: `strokeWidth` ∝ `|weight|`; brief CSS glow animation fires when a weight changes significantly during training
- **Node circles**: uniform colour; `r` (radius) scales with magnitude of total incoming weights
- **Hover tooltip**: on any edge → weight value; on any neuron → bias value for that unit
- **+/− node buttons**: small `+` below the bottom node of each hidden layer column; `−` above the top node; min 1 neuron
- **+ Layer / − Layer**: buttons flanking the diagram to add/remove an entire hidden layer; min 1 hidden layer
- **Per-layer activation dropdown**: sits above each hidden layer column

#### `Controls.jsx`
- Point density slider (200–2400)
- Training speed slider (1–20 steps/frame)
- Max epochs input (default 700)
- **Train** button: starts training from scratch
- **Play ▶ / Pause ⏸** toggle button: label/icon switches based on `isTraining` state
- **Reset** button: resets model weights + epoch counter + loss history
- Loss chart: hand-drawn canvas line chart updating live
- Epoch counter + progress bar
- Advanced settings `<details>`: learning rate, batch size, optimizer
- Done banner (hidden until training completes): mild glow animation, "Try another shape?"

#### `Tooltip.jsx`
- Reusable component wrapping any term with a hover plain-English explanation
- Implemented as CSS-positioned `<span>` — no external library

---

### `lib/shapes.js`

```
loadSVG(url)                   → parses SVG, extracts all <path> d attributes
sampleBoundary(paths, n)       → n points along the outline
sampleInterior(paths, bbox, n) → n points inside using rejection + ray-casting
normalize(points)              → scales all points to [-1, 1] bounding box
buildDataset(shape, density)   → returns { xs: Float32Array, ys: Float32Array }
                                  (50% interior+boundary = 1, 50% exterior = 0)
```

Point-in-polygon uses the ray-casting algorithm (cast a horizontal ray from the point, count crossings with path segments; odd = inside).

---

### `lib/model.js`

```
buildModel(layerSpec, activations)  → tf.Sequential
                                       input(2) → [hidden layers] → dense(1, sigmoid)
trainStep(model, xs, ys, config)    → runs one epoch, returns loss
getWeights(model)                   → returns { layers: [{ weights, biases }] }
```

- Sin activation: implemented as a custom Lambda layer (`tf.sin`)
- Optimizer factory: Adam / SGD / RMSProp selected at build time

---

### Training Loop (in `App.jsx` or a custom hook)

```javascript
requestAnimationFrame loop:
  for (let i = 0; i < speed; i++) {
    loss = await trainStep(...)
    epoch++
  }
  update weights → NetDiagram re-renders
  update heatmap → HeatmapCanvas re-renders
  if (epoch >= maxEpochs) → stop, celebrate
```

---

### Heatmap Colour Mapping

| Predicted probability | Colour |
|---|---|
| 0.0 (outside) | `rgb(69, 198, 231)` — cyan/teal |
| 0.5 (uncertain) | interpolated |
| 1.0 (inside) | `rgb(259, 111, 120)` — coral/red |

---

### Interactive Network Diagram — Visual Rules

| Element | Visual encoding |
|---|---|
| Input nodes | Teal, labelled `x` and `y` |
| Hidden nodes | Lavender; `r` ∝ Σ\|incoming weights\| |
| Output node | Coral, labelled `ŷ` |
| Bias nodes | Amber, labelled `1`; one per layer |
| Edges | Gray; `strokeWidth` ∝ \|weight\|; glow pulse on significant change |
| Hover (edge) | Tooltip: `w = 0.342` |
| Hover (node) | Tooltip: `bias = -0.11` |
| + / − buttons | Small circle buttons at column top/bottom |
| + Layer / − Layer | Pill buttons flanking the diagram |
| Activation select | Small `<select>` above each hidden column |

---

## 6. SVG Shapes Supplied

All shapes sourced from SVGRepo (svgrepo.com).

| File | Shape | Path structure |
|---|---|---|
| `alien-svgrepo-com.svg` | Alien head | Single compound path, `fill-rule="evenodd"` |
| `bluetooth-svgrepo-com.svg` | Bluetooth symbol | Single compound path, `fill-rule="evenodd"` |
| `flame-fire-svgrepo-com.svg` | Flame | Single path with group transform |
| `miniature-light-bulb-svgrepo-com.svg` | Light bulb | Multiple paths in `<g>` |
| `teddy-bear-svgrepo-com.svg` | Teddy bear | Multiple paths in `<g>` |

The SVG parser will handle both single-path and multi-path shapes; for multi-path shapes it unions all paths for the point-in-polygon check and samples the combined interior.

---

## 7. Where to Find More Shapes

When looking for additional shapes, target **filled silhouettes** — solid single-colour shapes that look like stamps or stickers. They convert cleanly to trainable point sets.

| Site | Licence | Search tip |
|---|---|---|
| **svgrepo.com** | CC0 / free | Search `[object] silhouette`; filter by mono/flat |
| **openclipart.org** | CC0 (public domain) | Search `[object] silhouette` |
| **wikimedia.org/commons** | CC0 / CC-BY | SVG category, search `[object] outline SVG` |
| **thenounproject.com** | Free with attribution | Very clean minimal icons; good sticker shapes |
| **flaticon.com** | Free with attribution | Choose "solid" or "filled" icon style |

**What to avoid:** stroke-only outlines (hollow paths), multi-colour illustrations, icons with interior cutouts (unless you are happy with the evenodd fill rule, the parser handles those too).

---

*End of specification.*
