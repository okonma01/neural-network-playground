# Neural Network Playground

**Live demo:** [Neural Playground](https://okonma01.github.io/neural-network-playground/)

Everyone knows by now that neural networks approximate functions. That idea has taken the world by storm, and for good reason. But not everything we care about is naturally expressed as a function. A circle, a logo, a letter — these are not curves on a graph, but regions of space: sets of points that belong together.

And shapes can be described as *relations over space* — specifically, whether a point belongs to a set.

So instead of asking whether a network can fit a function, we can ask a different question: can it learn such membership relations?

In other words, can a neural network learn a shape?

## The Setup

To make a shape learnable, we recast it as **binary classification** via an indicator function. Each point in the plane is assigned 1 (inside) or 0 (outside). The network learns an approximation to this function, and its decision boundary traces the resulting shape.

Shapes are `.svg` files sourced from [SVG Repo](https://www.svgrepo.com). The browser's own rendering engine determines the ground truth: for any point, `isPointInPath` says whether it's inside. Hence, there is no manual labeling.

Points are sampled randomly across the plane, tested for containment, and shuffled into a balanced dataset. By default about 1500 points are used, split evenly between classes, with some samples drawn specifically near the boundary to sharpen the network's sense of the edge.

The network itself is small by design - a few hidden layers, configurable activations (ReLU, sigmoid, tanh, sin), trained with the Adam optimizer and binary cross-entropy loss. Its hyperparameters are adjustable so you can see how they affect learning. The training loop runs in the browser without any external dependencies.

## What It Shows

Most introductions to neural networks show function fitting - a curve drawn through data. That framing is useful but limiting. A network is not drawing anything; it is partitioning space. Each neuron applies a linear cut to its inputs; the composition of these cuts, layer by layer, defines increasingly complex regions. After training, these regions align with the target shape.

What makes complex shapes learnable is that the Universal Approximation Theorem can be applied to indicator functions. A region can be represented by a function that is 1 inside and 0 outside, and neural networks can approximate such functions arbitrarily well (in an almost-everywhere sense) given sufficient capacity. The yin-yang symbol, a brand logo, the contour of a continent—all are, in principle, within reach. The playground is an invitation to test that claim.

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
