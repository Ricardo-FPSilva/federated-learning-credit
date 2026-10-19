import { datasetSummary } from "../data/presentationModel";
import { SectionHeading } from "./SectionHeading";

export function DatasetOverview() {
  return (
    <section className="content-section section-shell" id="dataset">
      <SectionHeading
        eyebrow="01 · Base experimental"
        title="O dataset HMEQ, sem atalhos."
        description="Solicitações de empréstimos com garantia imobiliária, organizadas em um pipeline reproduzível e sem ajuste sobre validação ou teste."
      />

      <div className="metric-grid">
        <article className="metric-card metric-primary">
          <span className="metric-index">Amostras</span>
          <strong>{datasetSummary.samples.toLocaleString("pt-BR")}</strong>
          <small>observações no arquivo congelado</small>
        </article>
        <article className="metric-card">
          <span className="metric-index">Entrada bruta</span>
          <strong>{datasetSummary.rawFeatures}</strong>
          <small>features originais, sem BAD</small>
        </article>
        <article className="metric-card">
          <span className="metric-index">Entrada da MLP</span>
          <strong>{datasetSummary.transformedFeatures}</strong>
          <small>features após one-hot encoding</small>
        </article>
        <article className="metric-card">
          <span className="metric-index">Target</span>
          <strong className="metric-text">{datasetSummary.target}</strong>
          <small>1 representa inadimplência</small>
        </article>
      </div>

      <div className="dataset-detail-grid">
        <article className="data-panel split-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-label">Split estratificado</span>
              <h3>Uma divisão, todos os modelos</h3>
            </div>
            <span className="verified-mark">✓ índices congelados</span>
          </div>

          <div className="split-bar" aria-label="60% treino, 20% validação e 20% teste">
            {datasetSummary.splits.map((split) => (
              <span
                className={`split-segment split-${split.id}`}
                key={split.id}
                style={{ width: `${split.fraction}%` }}
              />
            ))}
          </div>

          <div className="split-legend">
            {datasetSummary.splits.map((split) => (
              <div key={split.id}>
                <span className={`legend-dot dot-${split.id}`} />
                <div>
                  <strong>{split.label}</strong>
                  <small>{split.count.toLocaleString("pt-BR")} · {split.fraction}%</small>
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="data-panel client-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-label">Cenário distribuído</span>
              <h3>Três clientes simulados</h3>
            </div>
          </div>
          <div className="client-list">
            {datasetSummary.clients.map((client, index) => (
              <div className="client-row" key={client.id}>
                <span className="client-node">0{index + 1}</span>
                <div>
                  <strong>{client.id}</strong>
                  <small>partição IID de treino</small>
                </div>
                <b>{client.count.toLocaleString("pt-BR")}</b>
              </div>
            ))}
          </div>
          <p className="scientific-note">
            Clientes são partições experimentais do mesmo dataset — não representam bancos,
            regiões ou instituições reais.
          </p>
        </article>
      </div>

      <div className="transformation-strip">
        <span>Dados brutos</span><i>→</i>
        <span>Imputação no treino</span><i>→</i>
        <span>One-hot encoding</span><i>→</i>
        <span>Padronização</span><i>→</i>
        <strong>18 entradas</strong>
      </div>
    </section>
  );
}
