import { useLayoutEffect, useRef, useState } from "react";

import { createMlpScene, MLP_SCENE_COUNTS, type MlpSceneHandle } from "../three/createMlpScene";

const layerLabels = [
  { name: "Entrada", count: 18, tone: "purple" },
  { name: "Oculta I", count: 16, tone: "lavender" },
  { name: "Oculta II", count: 8, tone: "violet" },
  { name: "Saída", count: 1, tone: "cyan" },
] as const;

interface MlpArchitectureSceneProps {
  readonly onOpenLaboratory?: () => void;
}

export function MlpArchitectureScene({ onOpenLaboratory }: MlpArchitectureSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<MlpSceneHandle | null>(null);
  const [ready, setReady] = useState(false);

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handle = createMlpScene(container);
    sceneRef.current = handle;
    setReady(true);

    return () => {
      sceneRef.current = null;
      handle.dispose();
    };
  }, []);

  return (
    <div className="mlp-scene-module">
      <div className="mlp-layer-legend" aria-label="Camadas da arquitetura MLP">
        {layerLabels.map((layer) => (
          <div className={`layer-label layer-label-${layer.tone}`} key={layer.name}>
            <span>{layer.name}</span>
            <strong>{layer.count}</strong>
          </div>
        ))}
      </div>

      <div className="mlp-scene-frame">
        <div
          ref={containerRef}
          className="mlp-scene-container"
          role="img"
          aria-label={`Visualização estrutural interativa da MLP com ${MLP_SCENE_COUNTS.neurons} neurônios nas camadas 18, 16, 8 e 1.`}
        />
        <div className="scene-corner scene-corner-left" aria-hidden="true" />
        <div className="scene-corner scene-corner-right" aria-hidden="true" />
        <div className="scene-controls-hint">
          <span className={ready ? "scene-ready" : ""} />
          Arraste para rotacionar · scroll para zoom
        </div>
        <button
          className="scene-reset"
          type="button"
          onClick={() => sceneRef.current?.resetView()}
          disabled={!ready}
        >
          Recentrar
        </button>
      </div>

      <div className="mlp-scene-meta">
        <span>{MLP_SCENE_COUNTS.neurons} neurônios</span>
        <span>{MLP_SCENE_COUNTS.connections} conexões estruturais</span>
        <span>Renderização sob demanda</span>
      </div>

      <p className="structural-disclaimer">
        <span aria-hidden="true">◇</span>
        <strong>Visualização estrutural educativa. Não representa pesos, ativações ou resultados de uma execução real.</strong>
      </p>

      {onOpenLaboratory && (
        <div className="laboratory-launch-row">
          <button className="laboratory-launch" type="button" onClick={onOpenLaboratory}>
            <span aria-hidden="true">↗</span>
            Abrir laboratório neural
          </button>
        </div>
      )}
    </div>
  );
}
