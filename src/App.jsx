import { useEffect, useMemo, useRef, useState } from "react";
import HeatmapCanvas from "./components/HeatmapCanvas";
import ShapePicker from "./components/ShapePicker";
import NetDiagram from "./components/NetDiagram";
import Controls from "./components/Controls";
import { buildModel, getWeights, predictGrid, trainStep } from "./lib/model";
import { loadShapeCatalog } from "./lib/shapes";

const DEFAULT_NET_SPEC = {
  hiddenLayers: [6, 5],
  activations: ["sigmoid", "sin"]
};

const DEFAULT_TRAINING = {
  density: 1500,
  speed: 10,
  maxEpochs: 10000,
  learningRate: 0.01,
  batchSize: 128
};

const HEATMAP_RESOLUTION = 100;
const HEATMAP_SYNC_INTERVAL = 8;
const WEIGHT_SYNC_INTERVAL = 12;

function App() {
  const [shapes, setShapes] = useState([]);
  const [selectedShapeId, setSelectedShapeId] = useState(null);
  const [shapeState, setShapeState] = useState(null);
  const [netSpec, setNetSpec] = useState(DEFAULT_NET_SPEC);
  const [trainingConfig, setTrainingConfig] = useState(DEFAULT_TRAINING);
  const [showLearnedField, setShowLearnedField] = useState(true);
  const [heatmapData, setHeatmapData] = useState([]);
  const [weights, setWeights] = useState([]);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [isLoadingShapes, setIsLoadingShapes] = useState(true);
  const [loadError, setLoadError] = useState("");

  const modelRef = useRef(null);
  const rafRef = useRef(0);
  const trainingRef = useRef(false);
  const shapeStateRef = useRef(null);
  const configRef = useRef(trainingConfig);
  const epochRef = useRef(0);
  const lossRef = useRef(null);
  const heatmapPendingRef = useRef(false);
  const modelVersionRef = useRef(0);

  function resetTrainingMetrics() {
    setEpoch(0);
    epochRef.current = 0;
    setLoss(null);
    lossRef.current = null;
    setLossHistory([]);
    setIsCompleted(false);
  }

  async function refreshHeatmap(force = false) {
    if (!modelRef.current) {
      setHeatmapData([]);
      return;
    }
    if (heatmapPendingRef.current && !force) {
      return;
    }

    heatmapPendingRef.current = true;
    const version = modelVersionRef.current;
    const nextHeatmap = await predictGrid(modelRef.current, HEATMAP_RESOLUTION);
    if (version === modelVersionRef.current) {
      setHeatmapData(nextHeatmap);
    }
    heatmapPendingRef.current = false;
  }

  function stopTraining() {
    trainingRef.current = false;
    setIsTraining(false);
    cancelAnimationFrame(rafRef.current);
  }

  function rebuildModel() {
    modelRef.current?.dispose();
    modelVersionRef.current += 1;
    modelRef.current = buildModel(netSpec.hiddenLayers, netSpec.activations, {
      learningRate: trainingConfig.learningRate
    });
    setWeights(getWeights(modelRef.current));
    setHeatmapData([]);
  }

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        setIsLoadingShapes(true);
        const catalog = await loadShapeCatalog();
        if (cancelled) {
          return;
        }
        setShapes(catalog);
        setSelectedShapeId(catalog[0]?.id ?? null);
        setLoadError("");
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : "Failed to load shapes.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingShapes(false);
        }
      }
    }

    boot();

    return () => {
      cancelled = true;
    };
  }, []);

  const selectedShape = useMemo(
    () => shapes.find((shape) => shape.id === selectedShapeId) ?? null,
    [selectedShapeId, shapes]
  );

  useEffect(() => {
    configRef.current = trainingConfig;
  }, [trainingConfig]);

  useEffect(() => {
    epochRef.current = epoch;
  }, [epoch]);

  useEffect(() => {
    lossRef.current = loss;
  }, [loss]);

  useEffect(() => {
    shapeStateRef.current = shapeState;
  }, [shapeState]);

  useEffect(() => {
    return () => {
      stopTraining();
      modelRef.current?.dispose();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function prepareShape() {
      if (!selectedShape) {
        return;
      }

      stopTraining();
      resetTrainingMetrics();

      const nextState = await selectedShape.buildDataset(trainingConfig.density);
      if (cancelled) {
        return;
      }

      setShapeState(nextState);
    }

    prepareShape();

    return () => {
      cancelled = true;
    };
  }, [selectedShape, trainingConfig.density]);

  useEffect(() => {
    if (!shapeState) {
      return;
    }

    rebuildModel();
    resetTrainingMetrics();
    refreshHeatmap(true);
  }, [netSpec, trainingConfig.learningRate, shapeState]);

  useEffect(() => {
    if (!isTraining || !shapeState || !modelRef.current) {
      return undefined;
    }

    trainingRef.current = true;

    const tick = async () => {
      if (!trainingRef.current || !shapeStateRef.current || !modelRef.current) {
        return;
      }

      let latestLoss = lossRef.current;
      let latestEpoch = epochRef.current;
      const nextLosses = [];

      for (let stepIndex = 0; stepIndex < configRef.current.speed; stepIndex += 1) {
        if (latestEpoch >= configRef.current.maxEpochs) {
          break;
        }
        latestLoss = await trainStep(modelRef.current, shapeStateRef.current.xs, shapeStateRef.current.ys, {
          batchSize: configRef.current.batchSize
        });
        latestEpoch += 1;
        nextLosses.push(latestLoss);
      }

      if (nextLosses.length > 0) {
        epochRef.current = latestEpoch;
        lossRef.current = latestLoss;
        setEpoch(latestEpoch);
        setLoss(latestLoss);
        setLossHistory((current) => [...current, ...nextLosses]);
        if (latestEpoch % WEIGHT_SYNC_INTERVAL === 0 || latestEpoch >= configRef.current.maxEpochs) {
          setWeights(getWeights(modelRef.current));
        }
        if (latestEpoch % HEATMAP_SYNC_INTERVAL === 0 || latestEpoch >= configRef.current.maxEpochs) {
          refreshHeatmap();
        }
      }

      if (latestEpoch >= configRef.current.maxEpochs) {
        stopTraining();
        setIsCompleted(true);
        return;
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      trainingRef.current = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [isTraining, shapeState]);

  function handleNetChange(nextSpec) {
    setNetSpec(nextSpec);
  }

  function handleConfigChange(patch) {
    setTrainingConfig((current) => ({ ...current, ...patch }));
  }

  function handleTrain() {
    if (!shapeState || !modelRef.current) {
      return;
    }

    rebuildModel();
    resetTrainingMetrics();
    setIsTraining(true);
  }

  function handleTogglePlayback() {
    if (!shapeState || !modelRef.current) {
      return;
    }
    setIsTraining((current) => !current);
  }

  function handleReset() {
    if (!shapeState) {
      return;
    }

    stopTraining();
    rebuildModel();
    resetTrainingMetrics();
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">See a neural net learn in real time</p>
          <h1>Neural Playground</h1>
          <p className="intro-copy">
            Pick a shape and hit Train. Watch the network learn which points form the shape, and which points don't. Adjust the hyperparameters on the right to change how it learns.
          </p>
        </div>
        <div className="intro-note">
          <span>Inside = 1 (yes)</span>
          <span>Outside = 0 (no)</span>
        </div>
      </header>

      {loadError ? <p className="error-banner">{loadError}</p> : null}

      <main className="split-lab">
        <div className="controls-col">
          <ShapePicker
            shapes={shapes}
            selectedShapeId={selectedShapeId}
            onSelect={setSelectedShapeId}
            isLoading={isLoadingShapes}
          />

          <Controls
            config={trainingConfig}
            epoch={epoch}
            loss={loss}
            lossHistory={lossHistory}
            isTraining={isTraining}
            isCompleted={isCompleted}
            onConfigChange={handleConfigChange}
            onTrain={handleTrain}
            onTogglePlayback={handleTogglePlayback}
            onReset={handleReset}
          />
        </div>

        <section className="center-pane panel-frame">
          <HeatmapCanvas
            heatmapData={heatmapData}
            points={shapeState?.targetPoints ?? []}
            showLearnedField={showLearnedField}
            onToggleLearnedField={() => setShowLearnedField((current) => !current)}
            isCompleted={isCompleted}
          />
        </section>

        <NetDiagram netSpec={netSpec} weights={weights} onChange={handleNetChange} />
      </main>

      <footer className="app-footer">
        <span>Built by <a href="https://okonma01.github.io" target="_blank" rel="noreferrer">okonma01</a></span>
        <span className="footer-sep">·</span>
        <a href="https://github.com/okonma01/neural-network-playground" target="_blank" rel="noreferrer">GitHub</a>
      </footer>
    </div>
  );
}

export default App;