import type { ArtifactDescriptor } from "../data/artifactTypes";
import { EvidenceBadge } from "./EvidenceBadge";
import { SectionHeading } from "./SectionHeading";

interface ProvenanceOverviewProps {
  readonly artifacts: readonly ArtifactDescriptor[];
}

export function ProvenanceOverview({ artifacts }: ProvenanceOverviewProps) {
  return (
    <section className="content-section section-shell" id="proveniencia">
      <SectionHeading
        eyebrow="03 · Proveniência"
        title="Evidências não são intercambiáveis."
        description="A interface mantém cada fonte em sua categoria científica. Um valor publicado, uma reexecução histórica e um benchmark corrigido possuem significados distintos."
      />

      <div className="provenance-grid">
        {artifacts.map((artifact, index) => (
          <article className="provenance-card" key={artifact.id}>
            <div className="provenance-index">0{index + 1}</div>
            <EvidenceBadge
              evidenceKind={artifact.evidenceKind}
              availability={artifact.availability}
            />
            <h3>{artifact.title}</h3>
            <p>{artifact.summary}</p>
            <div className="caveat-box">
              <span>Limite de interpretação</span>
              <p>{artifact.caveat}</p>
            </div>
            <footer>
              {artifact.sourcePath ? (
                <code>{artifact.sourcePath}</code>
              ) : (
                <span>Sem artefato de resultado</span>
              )}
            </footer>
          </article>
        ))}
      </div>
    </section>
  );
}
