import { useMemo, useState } from "react";

const ACTIVATIONS = ["relu", "tanh", "sigmoid", "sin"];
const MAX_HIDDEN_LAYERS = 3;
const MAX_NODES_PER_LAYER = 8;
const DIAGRAM_VIEW = "34 6 652 348";
const DIAGRAM_TOP = 44;
const DIAGRAM_BOTTOM = 316;
const DIAGRAM_LEFT = 68;
const DIAGRAM_RIGHT = 652;
const LAYER_CONTROL_TOP = 26;
const LAYER_CONTROL_BOTTOM = 334;

function formatValue(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function NetDiagram({ netSpec, weights, onChange }) {
  const [hover, setHover] = useState(null);

  const columns = useMemo(() => [2, ...netSpec.hiddenLayers, 1], [netSpec.hiddenLayers]);
  const positions = useMemo(() => {
    const gapX = (DIAGRAM_RIGHT - DIAGRAM_LEFT) / Math.max(1, columns.length - 1);
    return columns.map((count, columnIndex) => {
      const visibleCount = count;
      return Array.from({ length: visibleCount }, (_, nodeIndex) => ({
        x: DIAGRAM_LEFT + columnIndex * gapX,
        y: DIAGRAM_TOP + ((nodeIndex + 1) / (visibleCount + 1)) * (DIAGRAM_BOTTOM - DIAGRAM_TOP),
        nodeIndex,
        visibleCount
      }));
    });
  }, [columns]);

  function patchLayer(layerIndex, units) {
    const hiddenLayers = [...netSpec.hiddenLayers];
    hiddenLayers[layerIndex] = Math.max(1, Math.min(MAX_NODES_PER_LAYER, units));
    onChange({ ...netSpec, hiddenLayers });
  }

  function patchActivation(layerIndex, activation) {
    const activations = [...netSpec.activations];
    activations[layerIndex] = activation;
    onChange({ ...netSpec, activations });
  }

  function addLayer() {
    if (netSpec.hiddenLayers.length >= MAX_HIDDEN_LAYERS) {
      return;
    }

    onChange({
      hiddenLayers: [...netSpec.hiddenLayers, 4],
      activations: [...netSpec.activations, "relu"]
    });
  }

  function removeLayer() {
    if (netSpec.hiddenLayers.length <= 1) {
      return;
    }
    onChange({
      hiddenLayers: netSpec.hiddenLayers.slice(0, -1),
      activations: netSpec.activations.slice(0, -1)
    });
  }

  const edgeMarkup = [];
  const nodeMarkup = [];

  positions.forEach((column, columnIndex) => {
    if (columnIndex === positions.length - 1) {
      return;
    }
    const nextColumn = positions[columnIndex + 1];
    const layerWeights = weights[columnIndex]?.weights ?? [];

    column.forEach((source) => {
      nextColumn.forEach((target) => {
        const weight = layerWeights[source.nodeIndex]?.[target.nodeIndex] ?? 0;
        const thickness = 1 + Math.min(4, Math.abs(weight) * 3.5);
        edgeMarkup.push(
          <line
            key={`edge-${columnIndex}-${source.nodeIndex}-${target.nodeIndex}`}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            stroke="rgba(76, 97, 116, 0.38)"
            strokeWidth={thickness}
            className="net-edge"
            onMouseEnter={() => setHover({ x: (source.x + target.x) * 0.5, y: (source.y + target.y) * 0.5, label: `w = ${formatValue(weight)}` })}
            onMouseLeave={() => setHover(null)}
          />
        );
      });
    });

  });

  positions.forEach((column, columnIndex) => {
    const layerBiases = weights[columnIndex - 1]?.biases ?? [];

    if (columnIndex > 0 && columnIndex < positions.length - 1) {
      const hiddenIndex = columnIndex - 1;
      const currentUnits = netSpec.hiddenLayers[hiddenIndex];
      const canDecrement = currentUnits > 1;
      const canIncrement = currentUnits < MAX_NODES_PER_LAYER;
      nodeMarkup.push(
        <g key={`controls-${columnIndex}`}>
          <g
            style={{ cursor: canDecrement ? "pointer" : "default", opacity: canDecrement ? 1 : 0.3 }}
            onClick={canDecrement ? () => patchLayer(hiddenIndex, currentUnits - 1) : undefined}
          >
            <circle className="layer-chip" cx={column[0].x} cy={LAYER_CONTROL_TOP} r={12} />
            <text x={column[0].x} y={LAYER_CONTROL_TOP + 4} textAnchor="middle" style={{ userSelect: "none", pointerEvents: "none" }}>−</text>
          </g>
          <g
            style={{ cursor: canIncrement ? "pointer" : "default", opacity: canIncrement ? 1 : 0.3 }}
            onClick={canIncrement ? () => patchLayer(hiddenIndex, currentUnits + 1) : undefined}
          >
            <circle className="layer-chip" cx={column[0].x} cy={LAYER_CONTROL_BOTTOM} r={12} />
            <text x={column[0].x} y={LAYER_CONTROL_BOTTOM + 4} textAnchor="middle" style={{ userSelect: "none", pointerEvents: "none" }}>+</text>
          </g>
        </g>
      );
    }

    column.forEach((node, nodeIndex) => {
      const incoming = columnIndex === 0 ? 0.35 : Math.abs((weights[columnIndex - 1]?.weights ?? []).reduce((sum, row) => sum + Math.abs(row[nodeIndex] ?? 0), 0));
      const radius = 18 + Math.min(8, incoming * 2.8);
      const fill = columnIndex === 0 ? "#22c5d8" : columnIndex === positions.length - 1 ? "#fb6f77" : "#bca9ff";
      const label = columnIndex === 0 ? ["x", "y"][nodeIndex] : columnIndex === positions.length - 1 ? "ŷ" : `${nodeIndex + 1}`;
      nodeMarkup.push(
        <g
          key={`node-${columnIndex}-${nodeIndex}`}
          onMouseEnter={() => setHover({ x: node.x, y: node.y - 24, label: `bias = ${formatValue(layerBiases[nodeIndex] ?? 0)}` })}
          onMouseLeave={() => setHover(null)}
        >
          <circle cx={node.x} cy={node.y} r={radius} fill={fill} stroke="#ffffff" strokeWidth="3" />
          <text x={node.x} y={node.y + 4} textAnchor="middle" className="node-label">{label}</text>
        </g>
      );
    });
  });

  return (
    <section className="control-card diagram-card">
      <div className="section-head">
        <div>
          <p className="section-kicker">Network layout</p>
          <h2>Network editor</h2>
        </div>
        <div className="layer-actions">
          <button type="button" className="mini-pill" onClick={removeLayer} disabled={netSpec.hiddenLayers.length <= 1}>− Layer</button>
          <button type="button" className="mini-pill" onClick={addLayer} disabled={netSpec.hiddenLayers.length >= MAX_HIDDEN_LAYERS}>+ Layer</button>
        </div>
      </div>

      <div className="activation-row" aria-label="Hidden layer activations">
        {netSpec.hiddenLayers.map((units, index) => (
          <label key={`activation-control-${index}`} className="activation-control">
            <span>{`Layer ${index + 1} · ${units} nodes`}</span>
            <select className="activation-select" value={netSpec.activations[index]} onChange={(event) => patchActivation(index, event.target.value)}>
              {ACTIVATIONS.map((activation) => (
                <option key={activation} value={activation}>{activation}</option>
              ))}
            </select>
          </label>
        ))}
      </div>

      <svg viewBox={DIAGRAM_VIEW} className="network-svg" aria-label="Neural network diagram">
        {edgeMarkup}
        {nodeMarkup}
        {hover ? (
          <g className="net-tooltip" transform={`translate(${hover.x}, ${hover.y})`}>
            <rect x="-42" y="-28" width="84" height="22" rx="10" />
            <text x="0" y="-13" textAnchor="middle">{hover.label}</text>
          </g>
        ) : null}
      </svg>
    </section>
  );
}

export default NetDiagram;