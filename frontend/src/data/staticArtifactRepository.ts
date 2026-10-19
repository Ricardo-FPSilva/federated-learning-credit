import type { ArtifactRepository } from "./artifactRepository";
import type { ArtifactDescriptor, ValidatedArtifactEnvelope } from "./artifactTypes";

const artifactCatalog = [
  {
    id: "historical-readme-results",
    title: "Resultados publicados no README",
    evidenceKind: "historical_documented",
    availability: "catalogued",
    sourcePath: "results/historical/readme-results.json",
    summary: "Transcrição preservada dos valores publicados originalmente.",
    caveat: "Não comprova execução, ambiente, seed de treinamento ou origem numérica completa.",
  },
  {
    id: "historical-LqJ1xZ-manifest",
    title: "Reexecução do protocolo histórico",
    evidenceKind: "historical_rerun",
    availability: "completed",
    sourcePath: "results/runs/historical-LqJ1xZ/manifest.json",
    summary: "Execução preservada do comportamento histórico com ambiente registrado.",
    caveat: "Mantém deliberadamente leakage e agregações históricas; não é o benchmark corrigido.",
  },
  {
    id: "benchmark-v1-planned",
    title: "Benchmark corrigido",
    evidenceKind: "new_experiment",
    availability: "not_executed",
    sourcePath: null,
    summary: "Protocolo, dados e componentes de modelos preparados para execução futura.",
    caveat: "Nenhuma métrica do benchmark corrigido foi produzida ou publicada.",
  },
] as const satisfies readonly ArtifactDescriptor[];

class StaticArtifactRepository implements ArtifactRepository {
  async list(): Promise<readonly ArtifactDescriptor[]> {
    return artifactCatalog;
  }

  async getById(_artifactId: string): Promise<ValidatedArtifactEnvelope | null> {
    // Entrega 1 cataloga as fontes sem duplicar a validação científica Python.
    // A implementação HTTP futura substituirá este adapter.
    return null;
  }
}

export const artifactRepository: ArtifactRepository = new StaticArtifactRepository();
