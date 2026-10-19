import {
  AdditiveBlending,
  BufferGeometry,
  Color,
  DynamicDrawUsage,
  Float32BufferAttribute,
  InstancedMesh,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  MeshBasicMaterial,
  PerspectiveCamera,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const LAYERS = [
  { id: "input", count: 18, x: -6, spacing: 0.47, color: "#8b5cf6", radius: 0.15 },
  { id: "hidden-1", count: 16, x: -2, spacing: 0.51, color: "#c4b5fd", radius: 0.17 },
  { id: "hidden-2", count: 8, x: 2, spacing: 0.78, color: "#a78bfa", radius: 0.19 },
  { id: "output", count: 1, x: 6, spacing: 0, color: "#22d3ee", radius: 0.25 },
] as const;

const INITIAL_CAMERA_POSITION = new Vector3(0, 0, 20);

export const MLP_SCENE_COUNTS = Object.freeze({
  neurons: LAYERS.reduce((total, layer) => total + layer.count, 0),
  connections: LAYERS.slice(0, -1).reduce(
    (total, layer, index) => total + layer.count * LAYERS[index + 1]!.count,
    0,
  ),
});

export interface MlpSceneHandle {
  resetView(): void;
  playForwardPass(): void;
  pauseForwardPass(): void;
  resetSignals(): void;
  dispose(): void;
}

export type MlpSignalState = "idle" | "running" | "paused" | "completed";

export interface MlpSceneOptions {
  readonly signals?: boolean;
  readonly onSignalStateChange?: (state: MlpSignalState) => void;
  readonly onForwardComplete?: () => void;
}

const SIGNAL_COUNT = 5;
const SIGNAL_PHASE_DURATION_MS = 1800;
const SIGNAL_PHASES_DURATION_MS = SIGNAL_PHASE_DURATION_MS * (LAYERS.length - 1);
const OUTPUT_HOLD_DURATION_MS = 600;
const SIGNAL_TOTAL_DURATION_MS = SIGNAL_PHASES_DURATION_MS + OUTPUT_HOLD_DURATION_MS;
const LAB_IDLE_NEURON_OPACITY = 0.16;
const LAB_IDLE_CONNECTION_OPACITY = 0.025;

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function smoothstep(edgeStart: number, edgeEnd: number, value: number): number {
  const progress = clamp01((value - edgeStart) / (edgeEnd - edgeStart));
  return progress * progress * (3 - 2 * progress);
}

function layerPositions(count: number, x: number, spacing: number): Vector3[] {
  const center = (count - 1) / 2;
  return Array.from(
    { length: count },
    (_, index) => new Vector3(x, (center - index) * spacing, 0),
  );
}

function createConnectionGeometry(positions: readonly Vector3[][]): BufferGeometry {
  const vertices: number[] = [];
  const colors: number[] = [];

  for (let layerIndex = 0; layerIndex < positions.length - 1; layerIndex += 1) {
    const sourceLayer = positions[layerIndex]!;
    const targetLayer = positions[layerIndex + 1]!;
    const sourceColor = new Color(LAYERS[layerIndex]!.color);
    const targetColor = new Color(LAYERS[layerIndex + 1]!.color);

    for (const source of sourceLayer) {
      for (const target of targetLayer) {
        vertices.push(source.x, source.y, source.z, target.x, target.y, target.z);
        colors.push(
          sourceColor.r, sourceColor.g, sourceColor.b,
          targetColor.r, targetColor.g, targetColor.b,
        );
      }
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(vertices, 3));
  geometry.setAttribute("color", new Float32BufferAttribute(colors, 3));
  return geometry;
}

/**
 * Monta a visualização estrutural da MLP sem loop contínuo de animação.
 * A cena renderiza somente em montagem, resize, reset ou interação do usuário.
 */
export function createMlpScene(
  container: HTMLElement,
  options: MlpSceneOptions = {},
): MlpSceneHandle {
  const scene = new Scene();
  const camera = new PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.copy(INITIAL_CAMERA_POSITION);

  const renderer = new WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.domElement.className = "mlp-webgl-canvas";
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.enablePan = false;
  controls.minDistance = 13;
  controls.maxDistance = 34;
  controls.minPolarAngle = Math.PI * 0.2;
  controls.maxPolarAngle = Math.PI * 0.8;
  controls.target.set(0, 0, 0);
  controls.update();
  controls.saveState();

  const positions = LAYERS.map((layer) =>
    layerPositions(layer.count, layer.x, layer.spacing),
  );
  const layerBaseColors = LAYERS.map((layer) => new Color(layer.color));
  const neuronHighlight = new Color("#f8f5ff");
  const neuronGeometry = new SphereGeometry(1, 18, 12);
  const neuronMaterials: MeshBasicMaterial[] = [];
  const neuronMeshes: InstancedMesh[] = [];
  const matrix = new Matrix4();

  LAYERS.forEach((layer, layerIndex) => {
    const material = new MeshBasicMaterial({
      color: layer.color,
      transparent: true,
      opacity: options.signals
        ? LAB_IDLE_NEURON_OPACITY
        : layer.id === "input" ? 0.82 : 0.94,
    });
    const mesh = new InstancedMesh(neuronGeometry, material, layer.count);
    mesh.name = `mlp-layer-${layer.id}`;

    positions[layerIndex]!.forEach((position, neuronIndex) => {
      matrix.makeScale(layer.radius, layer.radius, layer.radius);
      matrix.setPosition(position);
      mesh.setMatrixAt(neuronIndex, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    neuronMaterials.push(material);
    neuronMeshes.push(mesh);
    scene.add(mesh);
  });

  const connectionGeometry = createConnectionGeometry(positions);
  const connectionMaterial = new LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: options.signals ? LAB_IDLE_CONNECTION_OPACITY : 0.105,
    depthWrite: false,
  });
  const connections = new LineSegments(connectionGeometry, connectionMaterial);
  connections.name = "mlp-structural-connections";
  connections.renderOrder = -1;
  scene.add(connections);

  const activeRoutePositions = new Float32Array(SIGNAL_COUNT * 2 * 3);
  const activeRouteGeometry = new BufferGeometry();
  const activeRoutePositionAttribute = new Float32BufferAttribute(activeRoutePositions, 3);
  activeRoutePositionAttribute.setUsage(DynamicDrawUsage);
  activeRouteGeometry.setAttribute("position", activeRoutePositionAttribute);
  activeRouteGeometry.setDrawRange(0, 0);
  const activeRouteMaterial = new LineBasicMaterial({
    color: "#67e8f9",
    transparent: true,
    opacity: 0.68,
    depthTest: false,
    depthWrite: false,
    blending: AdditiveBlending,
  });
  const activeRoutes = new LineSegments(activeRouteGeometry, activeRouteMaterial);
  activeRoutes.name = "mlp-didactic-active-routes";
  activeRoutes.renderOrder = 1;
  activeRoutes.visible = options.signals === true;
  activeRoutes.frustumCulled = false;
  scene.add(activeRoutes);

  const signalCoreMaterial = new MeshBasicMaterial({
    color: "#ffffff",
    transparent: true,
    opacity: 1,
    depthTest: false,
    depthWrite: false,
    blending: AdditiveBlending,
  });
  const signalGlowMaterial = new MeshBasicMaterial({
    color: "#22d3ee",
    transparent: true,
    opacity: 0.38,
    depthTest: false,
    depthWrite: false,
    blending: AdditiveBlending,
  });
  const signalCores = new InstancedMesh(neuronGeometry, signalCoreMaterial, SIGNAL_COUNT);
  const signalGlows = new InstancedMesh(neuronGeometry, signalGlowMaterial, SIGNAL_COUNT);
  signalCores.name = "mlp-didactic-signal-cores";
  signalGlows.name = "mlp-didactic-signal-glows";
  signalCores.renderOrder = 3;
  signalGlows.renderOrder = 2;
  signalCores.visible = options.signals === true;
  signalGlows.visible = options.signals === true;
  signalCores.frustumCulled = false;
  signalGlows.frustumCulled = false;
  for (let index = 0; index < SIGNAL_COUNT; index += 1) {
    matrix.makeScale(0, 0, 0);
    signalCores.setMatrixAt(index, matrix);
    signalGlows.setMatrixAt(index, matrix);
  }
  signalCores.instanceMatrix.setUsage(DynamicDrawUsage);
  signalGlows.instanceMatrix.setUsage(DynamicDrawUsage);
  signalCores.instanceMatrix.needsUpdate = true;
  signalGlows.instanceMatrix.needsUpdate = true;
  scene.add(signalGlows, signalCores);

  const render = () => renderer.render(scene, camera);
  const onControlsChange = () => render();
  controls.addEventListener("change", onControlsChange);

  let previousWidth = 0;
  let previousHeight = 0;
  const resize = () => {
    const { width, height } = container.getBoundingClientRect();
    const nextWidth = Math.max(1, Math.round(width));
    const nextHeight = Math.max(1, Math.round(height));
    if (nextWidth === previousWidth && nextHeight === previousHeight) return;
    previousWidth = nextWidth;
    previousHeight = nextHeight;
    camera.aspect = nextWidth / nextHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(nextWidth, nextHeight, false);
    render();
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  let disposed = false;
  let signalState: MlpSignalState = "idle";
  let animationFrame: number | null = null;
  let animationStartedAt = 0;
  let pausedElapsed = 0;

  const emitSignalState = (state: MlpSignalState) => {
    signalState = state;
    options.onSignalStateChange?.(state);
  };

  const restoreLayerAppearance = () => {
    neuronMaterials.forEach((material, index) => {
      material.opacity = options.signals
        ? LAB_IDLE_NEURON_OPACITY
        : LAYERS[index]!.id === "input" ? 0.82 : 0.94;
      material.color.copy(layerBaseColors[index]!);
    });
    connectionMaterial.opacity = options.signals ? LAB_IDLE_CONNECTION_OPACITY : 0.105;
  };

  const setLayerActivity = (layerIndex: number, intensity: number) => {
    const activity = clamp01(intensity);
    const material = neuronMaterials[layerIndex]!;
    material.opacity = LAB_IDLE_NEURON_OPACITY + (1 - LAB_IDLE_NEURON_OPACITY) * activity;
    material.color
      .copy(layerBaseColors[layerIndex]!)
      .lerp(neuronHighlight, activity * 0.82);
  };

  const hideSignals = () => {
    activeRouteGeometry.setDrawRange(0, 0);
    for (let index = 0; index < SIGNAL_COUNT; index += 1) {
      matrix.makeScale(0, 0, 0);
      signalCores.setMatrixAt(index, matrix);
      signalGlows.setMatrixAt(index, matrix);
    }
    signalCores.instanceMatrix.needsUpdate = true;
    signalGlows.instanceMatrix.needsUpdate = true;
  };

  const updateRoutesAndPulses = (phase: number, phaseProgress: number) => {
    const sourceLayer = positions[phase]!;
    const targetLayer = positions[phase + 1]!;
    const travelProgress = clamp01((phaseProgress - 0.12) / 0.7);

    for (let index = 0; index < SIGNAL_COUNT; index += 1) {
      const normalizedIndex = index / (SIGNAL_COUNT - 1);
      const source = sourceLayer[Math.round(normalizedIndex * (sourceLayer.length - 1))]!;
      const target = targetLayer[Math.round(normalizedIndex * (targetLayer.length - 1))]!;
      const routeOffset = index * 6;
      activeRoutePositions[routeOffset] = source.x;
      activeRoutePositions[routeOffset + 1] = source.y;
      activeRoutePositions[routeOffset + 2] = 0.1;
      activeRoutePositions[routeOffset + 3] = target.x;
      activeRoutePositions[routeOffset + 4] = target.y;
      activeRoutePositions[routeOffset + 5] = 0.1;

      const stagger = index * 0.025;
      const pulseProgress = clamp01((travelProgress - stagger) / (1 - stagger));
      const easedProgress = smoothstep(0, 1, pulseProgress);
      const visibility = smoothstep(0, 0.06, pulseProgress)
        * (1 - smoothstep(0.9, 1, pulseProgress));
      const pulseX = source.x + (target.x - source.x) * easedProgress;
      const pulseY = source.y + (target.y - source.y) * easedProgress;
      const pulseScale = visibility * (1 + Math.sin(phaseProgress * Math.PI * 8 + index) * 0.08);

      matrix.makeScale(0.14 * pulseScale, 0.14 * pulseScale, 0.14 * pulseScale);
      matrix.setPosition(pulseX, pulseY, 0.32);
      signalCores.setMatrixAt(index, matrix);
      matrix.makeScale(0.31 * pulseScale, 0.31 * pulseScale, 0.31 * pulseScale);
      matrix.setPosition(pulseX, pulseY, 0.3);
      signalGlows.setMatrixAt(index, matrix);
    }

    activeRouteMaterial.opacity = 0.22
      + smoothstep(0.08, 0.22, phaseProgress)
      * (1 - smoothstep(0.82, 1, phaseProgress)) * 0.58;
    activeRoutePositionAttribute.needsUpdate = true;
    activeRouteGeometry.setDrawRange(0, SIGNAL_COUNT * 2);
    signalCores.instanceMatrix.needsUpdate = true;
    signalGlows.instanceMatrix.needsUpdate = true;
  };

  const updateSignals = (elapsed: number) => {
    restoreLayerAppearance();

    if (elapsed >= SIGNAL_PHASES_DURATION_MS) {
      hideSignals();
      const holdProgress = clamp01(
        (elapsed - SIGNAL_PHASES_DURATION_MS) / OUTPUT_HOLD_DURATION_MS,
      );
      const outputIntensity = 1 - smoothstep(0.35, 1, holdProgress);
      setLayerActivity(LAYERS.length - 1, outputIntensity);
      return;
    }

    const phase = Math.min(
      LAYERS.length - 2,
      Math.floor(elapsed / SIGNAL_PHASE_DURATION_MS),
    );
    const phaseElapsed = elapsed - phase * SIGNAL_PHASE_DURATION_MS;
    const phaseProgress = Math.min(1, phaseElapsed / SIGNAL_PHASE_DURATION_MS);
    const sourceFade = 1 - smoothstep(0.7, 1, phaseProgress);
    const sourceActivation = phase === 0
      ? smoothstep(0, 0.12, phaseProgress)
      : 1;
    const targetActivation = smoothstep(0.62, 0.9, phaseProgress);

    setLayerActivity(phase, sourceActivation * sourceFade);
    setLayerActivity(phase + 1, targetActivation);
    updateRoutesAndPulses(phase, phaseProgress);
  };

  const stopAnimationFrame = () => {
    if (animationFrame === null) return;
    window.cancelAnimationFrame(animationFrame);
    animationFrame = null;
  };

  const finishForwardPass = () => {
    stopAnimationFrame();
    pausedElapsed = 0;
    hideSignals();
    restoreLayerAppearance();
    emitSignalState("completed");
    render();
    options.onForwardComplete?.();
  };

  const animateSignals = (now: number) => {
    if (disposed || signalState !== "running") return;
    const elapsed = Math.max(0, now - animationStartedAt);
    if (elapsed >= SIGNAL_TOTAL_DURATION_MS) {
      finishForwardPass();
      return;
    }
    updateSignals(elapsed);
    render();
    animationFrame = window.requestAnimationFrame(animateSignals);
  };

  const playForwardPass = () => {
    if (disposed || options.signals !== true || signalState === "running") return;
    if (signalState === "completed" || signalState === "idle") pausedElapsed = 0;
    activeRoutes.visible = true;
    signalCores.visible = true;
    signalGlows.visible = true;
    animationStartedAt = performance.now() - pausedElapsed;
    emitSignalState("running");
    animationFrame = window.requestAnimationFrame(animateSignals);
  };

  const pauseForwardPass = () => {
    if (disposed || signalState !== "running") return;
    pausedElapsed = Math.min(
      SIGNAL_TOTAL_DURATION_MS,
      performance.now() - animationStartedAt,
    );
    stopAnimationFrame();
    emitSignalState("paused");
    render();
  };

  const resetSignals = () => {
    if (disposed) return;
    stopAnimationFrame();
    pausedElapsed = 0;
    hideSignals();
    restoreLayerAppearance();
    emitSignalState("idle");
    render();
  };

  return {
    resetView() {
      if (disposed) return;
      controls.reset();
      render();
    },
    playForwardPass,
    pauseForwardPass,
    resetSignals,
    dispose() {
      if (disposed) return;
      disposed = true;
      stopAnimationFrame();
      resizeObserver.disconnect();
      controls.removeEventListener("change", onControlsChange);
      controls.dispose();
      scene.remove(connections, activeRoutes, signalCores, signalGlows, ...neuronMeshes);
      connectionGeometry.dispose();
      connectionMaterial.dispose();
      activeRouteGeometry.dispose();
      activeRouteMaterial.dispose();
      signalCoreMaterial.dispose();
      signalGlowMaterial.dispose();
      neuronGeometry.dispose();
      neuronMaterials.forEach((material) => material.dispose());
      renderer.renderLists.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      scene.clear();
    },
  };
}
