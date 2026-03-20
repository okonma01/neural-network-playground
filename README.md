# Neural Network Playground

Everyone knows by now that neural networks approximate functions. That idea has taken the world by storm, and for good reason. But mathematics is bigger than functions. Not every relation maps a single input to a single output. Not every object we care about can be described as a curve on a graph. A circle, a logo, a letter - these are sets of points, regions of the plane, shapes. And shapes are relations.

The question this project asks is: can a neural network learn any shape? Not a curve it can draw, but a region it can recognize - answering, for a given point in 2D space, whether that point forms part of the shape. The answer, in theory, is yes. Watching it happen in real time is even more fascinating!

## The Setup

To make a shape learnable, the problem is framed as **binary classification**. Each training example is a 2D coordinate labeled 1 (inside) or 0 (outside). The network learns a decision boundary separating the two classes - and if training succeeds, that boundary traces the shape.

Shapes are `.svg` files sourced from [SVGREPO](https://www.svgrepo.com). The browser's own rendering engine determines the ground truth: for any point, `isPointInPath` says whether it's inside. Hence, there is no manual labeling.

Points are sampled randomly across the plane, tested for containment, and shuffled into a balanced dataset. By default about 1500 points are used, split evenly between classes, with some samples drawn specifically near the boundary to sharpen the network's sense of the edge.

The network itself is small by design - a few hidden layers, configurable activations (ReLU, sigmoid, tanh, sin), trained with Adam and binary cross-entropy. Its hyperparameters are adjustable so you can see how they affect learning. The training loop runs in the browser, with no external dependencies - just vanilla JavaScript.

## What It Shows

Most introductions to neural networks show function fitting - a curve drawn through data. That framing is useful but limiting. A network is not drawing anything. It is partitioning space. Every neuron applies a linear cut to its inputs; the composition of all those cuts, layer by layer, produces a region. That region, after training, approximates the target shape.

What makes any shape learnable - not just simple ones - is that the Universal Approximation Theorem extends beyond function graphs. It applies to indicator functions too: any measurable region can be approximated arbitrarily well by a sufficiently expressive network. The yin-yang symbol, a brand logo, the contour of a continent - all of these are, in principle, within reach. The playground is an invitation to test that claim.

---

## Running Locally

**Prerequisites:** [Node.js](https://nodejs.org) v18 or later.

Clone the repository and install dependencies:

```bash
git clone https://github.com/okonma01/neural-network-playground.git
cd neural-network-playground
npm install
```

Start the development server:

```bash
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

If you want to contribute, fork the repo on GitHub first, then clone your fork and open a pull request from a feature branch when you're ready.

## Build

To produce a production build:

```bash
npm run build
```

The output goes to `dist/` and can be served with any static file host.

## Adding Shapes

Drop any `.svg` file into the `svg/` directory. In dev mode, refresh the page. In production, rebuild. The shape will appear in the picker automatically - no code changes needed. The SVG should use `<path>` elements with explicit `d` attributes; compound paths and `fill-rule="evenodd"` are both supported.

## Stack

- React + Vite
- Custom neural network engine written in plain JavaScript (no ML library dependencies)
- SVG geometry via the browser's native Canvas 2D and SVG DOM APIs
