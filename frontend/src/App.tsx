import { useEffect, useLayoutEffect, useState } from "react";

import { AppHeader } from "./components/AppHeader";
import { DatasetOverview } from "./components/DatasetOverview";
import { FutureModules } from "./components/FutureModules";
import { MethodologyPipeline } from "./components/MethodologyPipeline";
import { ProjectOverview } from "./components/ProjectOverview";
import { ProvenanceOverview } from "./components/ProvenanceOverview";
import type { ArtifactDescriptor } from "./data/artifactTypes";
import { artifactRepository } from "./data/staticArtifactRepository";

export function App() {
  const [artifacts, setArtifacts] = useState<readonly ArtifactDescriptor[]>([]);

  useEffect(() => {
    let active = true;
    void artifactRepository.list().then((catalog) => {
      if (active) setArtifacts(catalog);
    });
    return () => {
      active = false;
    };
  }, []);

  useLayoutEffect(() => {
    const sectionId = window.location.hash.slice(1);
    if (!sectionId) return;
    document.getElementById(sectionId)?.scrollIntoView();
  }, []);

  return (
    <div className="app">
      <AppHeader />
      <main>
        <ProjectOverview />
        <DatasetOverview />
        <MethodologyPipeline />
        <ProvenanceOverview artifacts={artifacts} />
        <FutureModules />
      </main>
      <footer className="app-footer section-shell">
        <div className="footer-brand">
          <span className="brand-mark small" aria-hidden="true"><span /><span /><span /></span>
          <div><strong>Obsidian Research</strong><small>Federated Learning Credit</small></div>
        </div>
        <p>
          Interface de apresentação científica. Fontes autoritativas permanecem nos
          protocolos, contratos e artefatos versionados do projeto.
        </p>
        <a href="#visao-geral" aria-label="Voltar ao início">↑</a>
      </footer>
    </div>
  );
}
