// Vite resolves this glob at build time — drop any .svg into /svg/ and it appears automatically.
// In dev, a page refresh after adding/removing a file picks up the change.
// In production the list is baked in from whatever files exist at build time.
const svgGlob = import.meta.glob("/svg/*.svg", { query: "?url", eager: true });

const SHAPE_FILES = Object.entries(svgGlob).map(([path, mod]) => {
  const stem = path.split("/").pop().replace(/\.svg$/, "").replace(/-svgrepo(-com)?$/, "");
  const label = stem.split("-").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  return { id: stem, label, file: mod.default };
});

function getBounds(points) {
  return points.reduce(
    (current, point) => ({
      minX: Math.min(current.minX, point.x),
      minY: Math.min(current.minY, point.y),
      maxX: Math.max(current.maxX, point.x),
      maxY: Math.max(current.maxY, point.y)
    }),
    { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity }
  );
}

function createTransform(bounds, targetSpan = 1.76) {
  const width = Math.max(1e-6, bounds.maxX - bounds.minX);
  const height = Math.max(1e-6, bounds.maxY - bounds.minY);
  const scale = Math.max(width, height);
  const midX = (bounds.minX + bounds.maxX) * 0.5;
  const midY = (bounds.minY + bounds.maxY) * 0.5;

  return {
    bounds,
    targetSpan,
    normalize(point) {
      return {
        x: ((point.x - midX) / scale) * targetSpan,
        y: ((midY - point.y) / scale) * targetSpan
      };
    },
    denormalize(point) {
      return {
        x: (point.x / targetSpan) * scale + midX,
        y: midY - (point.y / targetSpan) * scale
      };
    }
  };
}

function multiplyPoint(matrix, point) {
  const svgPoint = new DOMPoint(point.x, point.y);
  const transformed = svgPoint.matrixTransform(matrix);
  return { x: transformed.x, y: transformed.y };
}

function normalizePoints(points, transform) {
  return points.map((point) => transform.normalize(point));
}

function createHiddenSvg(doc) {
  const host = document.createElement("div");
  host.style.position = "absolute";
  host.style.width = "0";
  host.style.height = "0";
  host.style.overflow = "hidden";
  host.setAttribute("aria-hidden", "true");
  host.appendChild(doc.documentElement);
  document.body.appendChild(host);
  return host;
}

function sampleBoundary(pathElements, totalSamples) {
  const perPath = Math.max(24, Math.ceil(totalSamples / Math.max(1, pathElements.length)));
  const points = [];

  pathElements.forEach((pathElement) => {
    const length = pathElement.getTotalLength();
    const matrix = pathElement.getCTM();

    for (let index = 0; index < perPath; index += 1) {
      const sample = pathElement.getPointAtLength((index / perPath) * length);
      const point = matrix ? multiplyPoint(matrix, sample) : { x: sample.x, y: sample.y };
      points.push(point);
    }
  });

  return points;
}

function createPathTester(pathElements) {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 1024;
  const context = canvas.getContext("2d");
  const pathRecords = pathElements.map((pathElement) => ({
    path: new Path2D(pathElement.getAttribute("d") || ""),
    matrix: pathElement.getCTM(),
    fillRule: pathElement.getAttribute("fill-rule") || pathElement.style.fillRule || "nonzero"
  }));

  return {
    contains(point) {
      return pathRecords.some(({ path, matrix, fillRule }) => {
        if (!matrix) {
          return context.isPointInPath(path, point.x, point.y, fillRule);
        }
        context.save();
        context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f);
        const hit = context.isPointInPath(path, point.x, point.y, fillRule);
        context.restore();
        return hit;
      });
    }
  };
}

function randomBetween(min, max) {
  return Math.random() * (max - min) + min;
}

async function parseShape(file) {
  const response = await fetch(file);
  if (!response.ok) {
    throw new Error(`Unable to load SVG: ${file}`);
  }
  const text = await response.text();
  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  const host = createHiddenSvg(doc);
  const pathElements = Array.from(host.querySelectorAll("path"));
  if (pathElements.length === 0) {
    host.remove();
    throw new Error(`SVG has no path elements: ${file}`);
  }
  const boundaryPoints = sampleBoundary(pathElements, 900);
  const svgTransform = createTransform(getBounds(boundaryPoints));
  const normalizedBoundary = normalizePoints(boundaryPoints, svgTransform);
  const tester = createPathTester(pathElements);
  host.remove();

  return {
    normalizedBoundary,
    normalizedBounds: getBounds(normalizedBoundary),
    svgTransform,
    tester,
    bounds: svgTransform.bounds
  };
}

function sampleByContainment(shape, sampleCount, isInside, maxAttemptsFactor) {
  const points = [];
  let attempts = 0;

  while (points.length < sampleCount && attempts < sampleCount * maxAttemptsFactor) {
    attempts += 1;
    const normalized = {
      x: randomBetween(-1, 1),
      y: randomBetween(-1, 1)
    };
    const sourcePoint = shape.svgTransform.denormalize(normalized);
    if (shape.tester.contains(sourcePoint) === isInside) {
      points.push(normalized);
    }
  }

  return points;
}

function buildDisplayPoints(shape) {
  const points = [];
  const step = 0.08;
  const normalizedBounds = shape.normalizedBounds;

  for (let y = normalizedBounds.minY; y <= normalizedBounds.maxY; y += step) {
    for (let x = normalizedBounds.minX; x <= normalizedBounds.maxX; x += step) {
      const sourcePoint = shape.svgTransform.denormalize({ x, y });
      if (shape.tester.contains(sourcePoint)) {
        points.push({ x, y });
      }
    }
  }

  return [...points, ...shape.normalizedBoundary.filter((_, index) => index % 3 === 0)];
}

function shuffle(array) {
  const copy = [...array];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}

function buildDataset(points) {
  return {
    points,
    xs: points.map((point) => [point.x, point.y]),
    ys: points.map((point) => point.label)
  };
}

export async function loadShapeCatalog() {
  const loaded = await Promise.all(
    SHAPE_FILES.map(async (record) => {
      const parsed = await parseShape(record.file);
      const targetPoints = buildDisplayPoints(parsed);
      return {
        ...record,
        preview: parsed.normalizedBoundary.filter((_, index) => index % 4 === 0).map((point) => [point.x, point.y]),
        buildDataset: async (density) => {
          const positiveInterior = sampleByContainment(parsed, Math.floor(density * 0.38), true, 50);
          const boundary = parsed.normalizedBoundary.slice(0, Math.max(80, Math.floor(density * 0.12)));
          const positive = [...positiveInterior, ...boundary].map((point) => ({ ...point, label: 1 }));
          const negative = sampleByContainment(parsed, positive.length, false, 70).map((point) => ({ ...point, label: 0 }));
          const points = shuffle([...positive, ...negative]);

          return {
            ...buildDataset(points),
            targetPoints
          };
        }
      };
    })
  );

  return loaded;
}