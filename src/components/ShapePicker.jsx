import { useEffect, useRef } from "react";

function PreviewCanvas({ preview }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !preview?.length) {
      return;
    }
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fff8ef";
    context.fillRect(0, 0, width, height);
    context.fillStyle = "#ff6f78";

    preview.forEach(([x, y]) => {
      const pixelX = ((x + 1) * 0.5) * width;
      const pixelY = (1 - (y + 1) * 0.5) * height;
      context.fillRect(pixelX, pixelY, 2, 2);
    });
  }, [preview]);

  return <canvas ref={canvasRef} width={66} height={48} className="shape-preview" />;
}

function ShapePicker({ shapes, selectedShapeId, onSelect, isLoading }) {
  return (
    <section className="control-card shape-card">
      <div className="section-head">
        <div>
          <p className="section-kicker">Pick a shape</p>
          <h2>Shapes</h2>
        </div>
      </div>

      <div className="shape-grid">
        {isLoading ? <p className="shape-loading">Loading shapes...</p> : null}
        {shapes.map((shape) => (
          <button
            key={shape.id}
            type="button"
            aria-label={shape.label}
            className={`shape-tile ${shape.id === selectedShapeId ? "selected" : ""}`}
            onClick={() => onSelect(shape.id)}
          >
            <PreviewCanvas preview={shape.preview} />
          </button>
        ))}
      </div>
    </section>
  );
}

export default ShapePicker;