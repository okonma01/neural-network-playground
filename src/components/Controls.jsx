import { useEffect, useRef } from "react";
import Tooltip from "./Tooltip";

function LossChart({ values, maxEpochs }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    const { width, height } = canvas;

    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#c6d7e6";
    context.strokeRect(0.5, 0.5, width - 1, height - 1);

    if (values.length < 2) {
      return;
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = Math.max(1e-6, max - min);

    context.beginPath();
    values.forEach((value, index) => {
      const x = 12 + ((index + 1) / maxEpochs) * (width - 24);
      const y = height - 12 - ((value - min) / range) * (height - 24);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.lineWidth = 2.5;
    context.strokeStyle = "#ff6f78";
    context.stroke();
  }, [values, maxEpochs]);

  return <canvas ref={canvasRef} width={440} height={80} className="loss-canvas" />;
}

function Controls({
  config,
  epoch,
  loss,
  lossHistory,
  isTraining,
  isCompleted,
  onConfigChange,
  onTrain,
  onTogglePlayback,
  onReset
}) {
  const progress = Math.min(100, (epoch / config.maxEpochs) * 100);

  return (
    <section className="control-card controls-card">
      <div className="section-head">
        <div>
          <p className="section-kicker">Training</p>
          <h2>Train the network</h2>
        </div>
      </div>

      <div className="control-grid">
        <label>
          <span><Tooltip label="Point density" text="How many example points the network trains on. More points = more detail, but slightly slower per step." /></span>
          <input type="range" min="200" max="2400" step="100" value={config.density} onChange={(event) => onConfigChange({ density: Number(event.target.value) })} />
          <strong>{config.density}</strong>
        </label>

        <label>
          <span><Tooltip label="Training speed" text="How many training steps run per animation frame. Higher = trains faster but the display updates less smoothly." /></span>
          <input type="range" min="1" max="20" step="1" value={config.speed} onChange={(event) => onConfigChange({ speed: Number(event.target.value) })} />
          <strong>{config.speed} steps/frame</strong>
        </label>

        <label>
          <span><Tooltip label="Max epochs" text="An epoch is one full pass through all training points. Training stops automatically when this number is reached." /></span>
          <input type="number" min="100" max="100000" step="100" value={config.maxEpochs} onChange={(event) => onConfigChange({ maxEpochs: Math.min(100000, Math.max(50, Number(event.target.value))) })} />
        </label>
      </div>

      <div className="button-row">
        <button type="button" className="primary-btn" onClick={onTrain}>Train</button>
        <button type="button" className="secondary-btn" onClick={onTogglePlayback}>
          {isTraining ? "⏸ Pause" : "▶ Play"}
        </button>
        <button type="button" className="ghost-btn" onClick={onReset}>Reset</button>
      </div>

      <div className="stats-grid">
        <div>
          <span>Epoch</span>
          <strong>{epoch}</strong>
        </div>
        <div>
          <span>Loss</span>
          <strong>{loss === null ? "-" : loss.toFixed(4)}</strong>
        </div>
        <div>
          <span>Progress</span>
          <strong>{progress.toFixed(0)}%</strong>
        </div>
      </div>

      <div className="progress-rail" aria-hidden="true">
        <span style={{ width: `${progress}%` }} />
      </div>

      <LossChart values={lossHistory} maxEpochs={config.maxEpochs} />

      <details className="advanced-panel">
        <summary>Advanced settings</summary>
        <div className="advanced-grid">
          <label>
            <span><Tooltip label="Learning rate" text="How large each optimization step is." /></span>
            <input type="number" min="0.0001" max="0.2" step="0.0005" value={config.learningRate} onChange={(event) => onConfigChange({ learningRate: Number(event.target.value) })} />
          </label>

          <label>
            <span><Tooltip label="Batch size" text="How many training examples each optimizer step sees." /></span>
            <input type="number" min="16" max="512" step="16" value={config.batchSize} onChange={(event) => onConfigChange({ batchSize: Number(event.target.value) })} />
          </label>

        </div>
      </details>

      {isCompleted ? <p className="done-banner">Done! Try another shape, or change the network and train again.</p> : null}
    </section>
  );
}

export default Controls;