export function ProjectOverview() {
  return (
    <section className="hero section-shell" id="visao-geral">
      <div className="hero-grid" aria-hidden="true" />
      <div className="hero-content">
        <div className="hero-kicker">
          <span>Pesquisa aplicada</span>
          <span className="separator">/</span>
          <span>Risco de crédito</span>
          <span className="separator">/</span>
          <span>Aprendizado federado</span>
        </div>
        <h1>
          Decisões de crédito sob uma
          <span> lente federada.</span>
        </h1>
        <p className="hero-lead">
          Uma análise controlada sobre o HMEQ para comparar modelos clássicos,
          centralizados, locais e federados — preservando a diferença entre evidência
          histórica e experimentos corrigidos ainda não executados.
        </p>
        <div className="hero-actions">
          <a className="primary-action" href="#dataset">
            Explorar metodologia
            <span aria-hidden="true">↘</span>
          </a>
          <a className="secondary-action" href="#proveniencia">
            Estado científico
          </a>
        </div>
      </div>

      <aside className="research-console" aria-label="Estado científico do projeto">
        <div className="console-header">
          <span>research://benchmark-v1</span>
          <span className="console-lights" aria-hidden="true"><i /><i /><i /></span>
        </div>
        <div className="console-body">
          <div className="console-row">
            <span>DATASET</span>
            <strong>HMEQ · congelado</strong>
          </div>
          <div className="console-row">
            <span>PROTOCOLO</span>
            <strong>benchmark-v1</strong>
          </div>
          <div className="console-row">
            <span>CLIENTES</span>
            <strong>3 · simulados</strong>
          </div>
          <div className="console-divider" />
          <div className="console-signal">
            <div className="signal-orbit"><span /><span /><span /></div>
            <div>
              <small>STATUS DO BENCHMARK</small>
              <strong>Preparado, não executado</strong>
            </div>
          </div>
          <p>
            Nenhuma métrica nova é exibida até que uma execução validada produza
            artefatos científicos completos.
          </p>
        </div>
      </aside>
    </section>
  );
}
