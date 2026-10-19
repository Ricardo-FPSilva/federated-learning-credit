import { useState } from "react";

import { MlpArchitectureScene } from "./MlpArchitectureScene";
import { NeuralLaboratory } from "./NeuralLaboratory";
import { SectionHeading } from "./SectionHeading";

export function FutureModules() {
  const [laboratoryOpen, setLaboratoryOpen] = useState(false);

  return (
    <>
      <section className="content-section section-shell" id="laboratorio">
        <SectionHeading
          eyebrow="04 · Laboratório visual"
          title="Arquitetura explorável, evidência preservada."
          description="A estrutura da MLP pode ser inspecionada em três dimensões sem atribuir significado científico a posições, conexões ou cores ilustrativas."
        />

        <div className="future-grid">
          <article className="future-panel architecture-visualization">
            <div className="future-panel-header">
              <div>
                <span className="panel-label">Arquitetura interativa</span>
                <h3>MLP 18 → 16 → 8 → 1</h3>
              </div>
              <span className="interactive-badge"><i /> Interativo</span>
            </div>
            <MlpArchitectureScene onOpenLaboratory={() => setLaboratoryOpen(true)} />
          </article>

          <article className="future-panel results-placeholder">
            <div className="future-panel-header">
              <div>
                <span className="panel-label">Comparação de resultados</span>
                <h3>Visão multi-modelo</h3>
              </div>
              <span className="coming-soon">Entrega 2</span>
            </div>
            <div className="placeholder-chart" aria-hidden="true">
              <span className="axis-y" />
              <span className="axis-x" />
              <div className="ghost-line ghost-one"><i /><i /><i /><i /></div>
              <div className="ghost-line ghost-two"><i /><i /><i /><i /></div>
            </div>
            <p>
              Históricos serão exibidos com fonte e limitações explícitas. O benchmark
              corrigido permanecerá vazio até uma execução validada.
            </p>
          </article>
        </div>
      </section>
      {laboratoryOpen && (
        <NeuralLaboratory onClose={() => setLaboratoryOpen(false)} />
      )}
    </>
  );
}
