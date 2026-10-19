import { methodologyTracks } from "../data/presentationModel";
import { SectionHeading } from "./SectionHeading";

export function MethodologyPipeline() {
  return (
    <section className="content-section methodology-section section-shell" id="metodologia">
      <SectionHeading
        eyebrow="02 · Desenho experimental"
        title="Mesma base. Três perspectivas."
        description="A comparação mantém dados, arquitetura e orçamento observáveis. O que muda é onde o aprendizado acontece e como os estados são combinados."
      />

      <div className="methodology-origin">
        <div className="origin-core"><span>HMEQ</span><strong>Treino congelado</strong></div>
        <div className="origin-line" aria-hidden="true"><span /><span /><span /></div>
      </div>

      <div className="methodology-grid">
        {methodologyTracks.map((track) => (
          <article className={`method-card method-${track.tone}`} key={track.id}>
            <div className="method-topline">
              <span>{track.index}</span>
              <small>{track.status}</small>
            </div>
            <div className="method-icon" aria-hidden="true">
              {track.id === "central" && <><i /><i /><i /></>}
              {track.id === "local" && <><i /><i /><i className="separated" /></>}
              {track.id === "federated" && <><i /><i className="hub" /><i /></>}
            </div>
            <span className="method-subtitle">{track.subtitle}</span>
            <h3>{track.title}</h3>
            <p>{track.description}</p>
          </article>
        ))}
      </div>

      <div className="protocol-guardrail">
        <span className="guardrail-icon" aria-hidden="true">◇</span>
        <div>
          <strong>Guardrail científico</strong>
          <p>
            Threshold fixo em <code>&gt; 0.5</code>, checkpoint final e teste reservado.
            Sem tuning no teste, seleção da melhor seed ou ensemble oculto.
          </p>
        </div>
      </div>
    </section>
  );
}
