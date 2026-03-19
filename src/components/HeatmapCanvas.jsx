import { useEffect, useRef, useState } from "react";

const AXIS_TICKS = [-1, 0, 1];
const GRID_LINES = [-1, -0.5, 0, 0.5, 1];

function toPixelX(value, width) {
  return ((value + 1) * 0.5) * width;
}

function toPixelY(value, height) {
  return (1 - (value + 1) * 0.5) * height;
}

function toColor(probability) {
  const value = Math.max(0, Math.min(1, probability));
  const red = Math.round(69 + (255 - 69) * value);
  const green = Math.round(198 + (111 - 198) * value);
  const blue = Math.round(231 + (120 - 231) * value);
  return [red, green, blue];
}

function drawGrid(context, width, height, clear = true) {
  if (clear) {
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fffdf8";
    context.fillRect(0, 0, width, height);
  }

  context.strokeStyle = "rgba(78, 101, 122, 0.08)";
  context.lineWidth = 1;
  GRID_LINES.forEach((value) => {
    const x = toPixelX(value, width);
    const y = toPixelY(value, height);
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  });

  context.strokeStyle = "rgba(35, 53, 69, 0.22)";
  context.beginPath();
  context.moveTo(toPixelX(0, width), 0);
  context.lineTo(toPixelX(0, width), height);
  context.stroke();
  context.beginPath();
  context.moveTo(0, toPixelY(0, height));
  context.lineTo(width, toPixelY(0, height));
  context.stroke();

  context.fillStyle = "rgba(35, 53, 69, 0.62)";
  context.font = '12px Manrope';
  AXIS_TICKS.forEach((tick) => {
    const x = toPixelX(tick, width);
    const y = toPixelY(tick, height);
    context.fillText(String(tick), Math.max(4, Math.min(width - 18, x - 7)), height - 8);
    if (tick !== 0) {
      context.fillText(String(tick), 8, Math.max(14, Math.min(height - 8, y + 4)));
    }
  });
}

function drawContour(context, probabilities, resolution, width, height) {
  const threshold = 0.5;
  const lineSegments = [];

  function pointAt(x, y, edge) {
    const p00 = probabilities[y * resolution + x];
    const p10 = probabilities[y * resolution + x + 1];
    const p01 = probabilities[(y + 1) * resolution + x];
    const p11 = probabilities[(y + 1) * resolution + x + 1];
    const nx = x / (resolution - 1);
    const ny = y / (resolution - 1);
    const step = 1 / (resolution - 1);

    if (edge === "top") {
      const t = (threshold - p00) / ((p10 - p00) || 1);
      return [nx + step * t, ny];
    }
    if (edge === "right") {
      const t = (threshold - p10) / ((p11 - p10) || 1);
      return [nx + step, ny + step * t];
    }
    if (edge === "bottom") {
      const t = (threshold - p01) / ((p11 - p01) || 1);
      return [nx + step * t, ny + step];
    }
    const t = (threshold - p00) / ((p01 - p00) || 1);
    return [nx, ny + step * t];
  }

  const lookup = {
    1: [["left", "top"]],
    2: [["top", "right"]],
    3: [["left", "right"]],
    4: [["right", "bottom"]],
    5: [["left", "bottom"], ["top", "right"]],
    6: [["top", "bottom"]],
    7: [["left", "bottom"]],
    8: [["left", "bottom"]],
    9: [["top", "bottom"]],
    10: [["left", "top"], ["right", "bottom"]],
    11: [["right", "bottom"]],
    12: [["left", "right"]],
    13: [["top", "right"]],
    14: [["left", "top"]]
  };

  for (let y = 0; y < resolution - 1; y += 1) {
    for (let x = 0; x < resolution - 1; x += 1) {
      const p00 = probabilities[y * resolution + x] >= threshold ? 1 : 0;
      const p10 = probabilities[y * resolution + x + 1] >= threshold ? 1 : 0;
      const p11 = probabilities[(y + 1) * resolution + x + 1] >= threshold ? 1 : 0;
      const p01 = probabilities[(y + 1) * resolution + x] >= threshold ? 1 : 0;
      const mask = p00 | (p10 << 1) | (p11 << 2) | (p01 << 3);
      (lookup[mask] || []).forEach(([edgeA, edgeB]) => {
        lineSegments.push([pointAt(x, y, edgeA), pointAt(x, y, edgeB)]);
      });
    }
  }

  context.strokeStyle = "rgba(251, 111, 120, 0.92)";
  context.lineWidth = 1.5;
  context.beginPath();
  lineSegments.forEach(([[ax, ay], [bx, by]]) => {
    context.moveTo(ax * width, ay * height);
    context.lineTo(bx * width, by * height);
  });
  context.stroke();
}

function drawHeatmapField(context, probabilities, width, height) {
  const resolution = Math.round(Math.sqrt(probabilities.length));
  const offscreen = document.createElement("canvas");
  offscreen.width = resolution;
  offscreen.height = resolution;
  const offscreenContext = offscreen.getContext("2d");
  const image = offscreenContext.createImageData(resolution, resolution);

  for (let y = 0; y < resolution; y += 1) {
    for (let x = 0; x < resolution; x += 1) {
      const probability = probabilities[y * resolution + x] ?? 0;
      const [red, green, blue] = toColor(probability);
      const offset = (y * resolution + x) * 4;
      image.data[offset] = red;
      image.data[offset + 1] = green;
      image.data[offset + 2] = blue;
      image.data[offset + 3] = 255;
    }
  }

  offscreenContext.putImageData(image, 0, 0);
  context.save();
  context.globalAlpha = 0.2;
  context.imageSmoothingEnabled = true;
  context.drawImage(offscreen, 0, 0, width, height);
  context.restore();

  return resolution;
}

function HeatmapCanvas({ heatmapData, points, showLearnedField, onToggleLearnedField, isCompleted }) {
  const heatmapRef = useRef(null);
  const pointsRef = useRef(null);
  const [size, setSize] = useState({ width: 720, height: 520 });

  useEffect(() => {
    const element = heatmapRef.current;
    if (!element) {
      return undefined;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      setSize({
        width: Math.max(320, Math.floor(entry.contentRect.width)),
        height: Math.max(320, Math.floor(entry.contentRect.height))
      });
    });

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function drawHeatmap() {
      const canvas = heatmapRef.current;
      if (!canvas) {
        return;
      }
      const context = canvas.getContext("2d", { willReadFrequently: true });
      drawGrid(context, size.width, size.height);

      if (!showLearnedField || !heatmapData?.length) {
        return;
      }

      if (cancelled) {
        return;
      }

      const resolution = drawHeatmapField(context, heatmapData, size.width, size.height);
      drawGrid(context, size.width, size.height, false);
      drawContour(context, heatmapData, resolution, size.width, size.height);
    }

    drawHeatmap();

    return () => {
      cancelled = true;
    };
  }, [heatmapData, showLearnedField, size]);

  useEffect(() => {
    const canvas = pointsRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, size.width, size.height);

    context.globalAlpha = 0.76;
    context.fillStyle = "#fb6f77";
    points.forEach(({ x, y }) => {
      const pixelX = toPixelX(x, size.width);
      const pixelY = toPixelY(y, size.height);
      context.fillRect(pixelX, pixelY, 2, 2);
    });

    if (isCompleted) {
      context.fillStyle = "rgba(255, 214, 102, 0.18)";
      context.fillRect(0, 0, size.width, size.height);
    }
  }, [isCompleted, points, size]);

  return (
    <div className={`heatmap-shell ${isCompleted ? "celebrate" : ""}`}>
      <div className="pane-head">
        <p className="section-kicker">Live prediction map</p>
        <button type="button" className="toggle-chip canvas-toggle" onClick={onToggleLearnedField}>
          {showLearnedField ? "Hide Boundary" : "Show Boundary"}
        </button>
      </div>
      <div className="heatmap-stage">
        <canvas ref={heatmapRef} width={size.width} height={size.height} />
        <canvas ref={pointsRef} width={size.width} height={size.height} />
      </div>
    </div>
  );
}

export default HeatmapCanvas;