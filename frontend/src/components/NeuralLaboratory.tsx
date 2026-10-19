import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import {
  createMlpScene,
  type MlpSceneHandle,
  type MlpSignalState,
} from "../three/createMlpScene";

const FEATURES = [
  ["LOAN", "Valor do empréstimo"],
  ["MORTDUE", "Saldo hipotecário"],
  ["VALUE", "Valor do imóvel"],
  ["REASON=DebtCon", "Motivo: consolidação"],
  ["REASON=HomeImp", "Motivo: melhoria"],
  ["JOB=Mgr", "Ocupação: gestão"],
  ["JOB=Office", "Ocupação: escritório"],
  ["JOB=Other", "Ocupação: outras"],
  ["JOB=ProfExe", "Ocupação: profissional"],
  ["JOB=Sales", "Ocupação: vendas"],
  ["JOB=Self", "Ocupação: autônomo"],
  ["YOJ", "Anos no emprego"],
  ["DEROG", "Registros depreciativos"],
  ["DELINQ", "Linhas em atraso"],
  ["CLAGE", "Idade do crédito"],
  ["NINQ", "Consultas recentes"],
  ["CLNO", "Linhas de crédito"],
  ["DEBTINC", "Relação dívida/renda"],
] as const;

const INITIAL_VALUES = [
  0.68, 0.43, 0.57, 1, 0, 0, 0, 1, 0, 0, 0, 0.36, 0.12, 0.19, 0.62, 0.27, 0.51, 0.74,
] as const;

type LaboratoryMode = "idle" | "running" | "paused" | "auto" | "completed";

interface NeuralLaboratoryProps {
  readonly onClose: () => void;
}

function illustrativeValues(): number[] {
  const source = new Uint32Array(12);
  globalThis.crypto.getRandomValues(source);
  const values = new Array<number>(FEATURES.length).fill(0);
  const numericIndices = [0, 1, 2, 11, 12, 13, 14, 15, 16, 17] as const;
  numericIndices.forEach((featureIndex, sourceIndex) => {
    values[featureIndex] = (source[sourceIndex]! % 1001) / 1000;
  });
  values[3 + (source[10]! % 2)] = 1;
  values[5 + (source[11]! % 6)] = 1;
  return values;
}

function modeLabel(mode: LaboratoryMode): string {
  const labels: Record<LaboratoryMode, string> = {
    idle: "Idle",
    running: "Forward Pass",
    paused: "Pausado",
    auto: "Auto",
    completed: "Concluído",
  };
  return labels[mode];
}

function outputMessage(mode: LaboratoryMode): string {
  if (mode === "running" || mode === "auto") return "Sinal didático em propagação";
  if (mode === "paused") return "Simulação visual pausada";
  if (mode === "completed") return "Percurso visual concluído";
  return "Sem inferência carregada";
}

export function NeuralLaboratory({ onClose }: NeuralLaboratoryProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<MlpSceneHandle | null>(null);
  const autoRef = useRef(false);
  const autoTimerRef = useRef<number | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const [mode, setMode] = useState<LaboratoryMode>("idle");
  const [values, setValues] = useState<readonly number[]>(INITIAL_VALUES);
  const [ready, setReady] = useState(false);

  const clearAutoTimer = () => {
    if (autoTimerRef.current === null) return;
    window.clearTimeout(autoTimerRef.current);
    autoTimerRef.current = null;
  };

  useLayoutEffect(() => {
    const container = canvasRef.current;
    if (!container) return;

    const onSignalStateChange = (state: MlpSignalState) => {
      if (state === "running") setMode(autoRef.current ? "auto" : "running");
      if (state === "paused") setMode("paused");
      if (state === "idle") setMode("idle");
    };

    const handle = createMlpScene(container, {
      signals: true,
      onSignalStateChange,
      onForwardComplete: () => {
        if (!autoRef.current) {
          setMode("completed");
          return;
        }
        setMode("auto");
        autoTimerRef.current = window.setTimeout(() => {
          if (!autoRef.current) return;
          setValues(illustrativeValues());
          handle.resetSignals();
          handle.playForwardPass();
        }, 650);
      },
    });
    sceneRef.current = handle;
    setReady(true);

    return () => {
      autoRef.current = false;
      clearAutoTimer();
      sceneRef.current = null;
      handle.dispose();
    };
  }, []);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [onClose]);

  const disableAuto = () => {
    autoRef.current = false;
    clearAutoTimer();
  };

  const startForwardPass = () => {
    disableAuto();
    sceneRef.current?.playForwardPass();
  };

  const startAuto = () => {
    clearAutoTimer();
    autoRef.current = true;
    sceneRef.current?.playForwardPass();
  };

  const pause = () => {
    disableAuto();
    sceneRef.current?.pauseForwardPass();
    setMode("paused");
  };

  const reset = () => {
    disableAuto();
    sceneRef.current?.resetSignals();
    setMode("idle");
  };

  const newExample = () => {
    reset();
    setValues(illustrativeValues());
  };

  return createPortal(
    <div className="neural-lab-overlay" role="dialog" aria-modal="true" aria-labelledby="neural-lab-title">
      <header className="neural-lab-header">
        <div className="neural-lab-title-block">
          <span className="panel-label">Obsidian Research · Laboratório didático</span>
          <h2 id="neural-lab-title">MLP Credit Risk Laboratory</h2>
          <p>Arquitetura corrigida · 18 → 16 → 8 → 1 · 449 parâmetros</p>
        </div>
        <div className="neural-lab-header-actions">
          <span className={`lab-mode lab-mode-${mode}`}><i /> {modeLabel(mode)}</span>
          <button ref={closeButtonRef} className="lab-close" type="button" onClick={onClose}>
            Fechar <kbd>Esc</kbd>
          </button>
        </div>
      </header>

      <div className="neural-lab-integrity" role="note">
        <span>◇</span>
        <strong>Simulação didática de propagação de sinais.</strong>
        Não representa pesos, ativações, inferências ou resultados reais do benchmark.
      </div>

      <main className="neural-lab-workspace">
        <aside className="lab-panel lab-features-panel">
          <div className="lab-panel-heading">
            <span className="panel-label">Entrada ilustrativa</span>
            <strong>18 features transformadas</strong>
          </div>
          <div className="lab-feature-list">
            {FEATURES.map(([name, description], index) => (
              <div className="lab-feature" key={name} title={description}>
                <span>{name}</span>
                <div><i style={{ width: `${Math.round(values[index]! * 100)}%` }} /></div>
                <b>{values[index]!.toFixed(2)}</b>
              </div>
            ))}
          </div>
          <p>Valores normalizados apenas para a animação visual; não são dados de uma pessoa real.</p>
        </aside>

        <section className="lab-canvas-panel" aria-label="Rede neural tridimensional">
          <div ref={canvasRef} className="lab-canvas" />
          <div className="lab-layer-labels" aria-hidden="true">
            <span>Entrada <b>18</b></span>
            <span>Oculta I <b>16</b></span>
            <span>Oculta II <b>8</b></span>
            <span>Saída <b>1</b></span>
          </div>
          <div className="lab-camera-hint">
            <i className={ready ? "ready" : ""} /> Arraste para rotacionar · scroll para zoom
          </div>
          <button className="lab-recenter" type="button" onClick={() => sceneRef.current?.resetView()} disabled={!ready}>
            Recentrar câmera
          </button>
        </section>

        <aside className="lab-panel lab-architecture-panel">
          <div className="lab-output-state" aria-live="polite">
            <span>Estado da saída</span>
            <strong>{outputMessage(mode)}</strong>
            <small>Probabilidade indisponível · nenhum checkpoint carregado</small>
          </div>

          <div className="lab-panel-heading">
            <span className="panel-label">Arquitetura</span>
            <strong>Configuração congelada</strong>
          </div>
          <dl className="lab-architecture-list">
            <div><dt>Parâmetros</dt><dd>449</dd></div>
            <div><dt>Camadas</dt><dd>4</dd></div>
            <div><dt>Ocultas</dt><dd>ReLU</dd></div>
            <div><dt>Treino</dt><dd>Logits</dd></div>
            <div><dt>Loss</dt><dd>BCEWithLogitsLoss</dd></div>
            <div><dt>Optimizer</dt><dd>Adam · 0.01</dd></div>
          </dl>

          <div className="lab-layer-stack">
            <div><i className="layer-purple" /><span>Entrada</span><b>18</b></div>
            <div><i className="layer-lavender" /><span>Dense + ReLU</span><b>16</b></div>
            <div><i className="layer-violet" /><span>Dense + ReLU</span><b>8</b></div>
            <div><i className="layer-cyan" /><span>Logit</span><b>1</b></div>
          </div>
        </aside>
      </main>

      <footer className="neural-lab-controls" aria-label="Controles da simulação">
        <span className="lab-control-status"><i /> {modeLabel(mode)}</span>
        <button type="button" onClick={startForwardPass} disabled={!ready || mode === "running" || mode === "auto"}>
          ▶ Forward Pass
        </button>
        <button className={mode === "auto" ? "active" : ""} type="button" onClick={startAuto} disabled={!ready || mode === "auto"}>
          ⟳ Auto
        </button>
        <button type="button" onClick={pause} disabled={mode !== "running" && mode !== "auto"}>
          Ⅱ Pausar
        </button>
        <button type="button" onClick={reset} disabled={!ready || mode === "idle"}>
          ↺ Reiniciar
        </button>
        <button className="new-example" type="button" onClick={newExample} disabled={!ready}>
          Novo exemplo
        </button>
      </footer>
    </div>,
    document.body,
  );
}
