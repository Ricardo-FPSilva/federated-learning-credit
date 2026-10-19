const navigation = [
  ["visao-geral", "Visão geral"],
  ["dataset", "Dataset"],
  ["metodologia", "Metodologia"],
  ["proveniencia", "Proveniência"],
  ["laboratorio", "Laboratório"],
] as const;

export function AppHeader() {
  return (
    <header className="app-header">
      <a className="brand" href="#visao-geral" aria-label="Obsidian Research — início">
        <span className="brand-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="brand-copy">
          <strong>OBSIDIAN</strong>
          <small>RESEARCH</small>
        </span>
      </a>

      <nav className="main-navigation" aria-label="Navegação principal">
        {navigation.map(([id, label]) => (
          <a href={`#${id}`} key={id}>
            {label}
          </a>
        ))}
      </nav>

      <div className="header-status" title="O benchmark corrigido ainda não foi executado">
        <span className="status-dot" />
        Pesquisa em progresso
      </div>
    </header>
  );
}
