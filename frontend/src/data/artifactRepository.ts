import type { ArtifactDescriptor, ValidatedArtifactEnvelope } from "./artifactTypes";

/**
 * Fronteira de dados do frontend.
 *
 * A implementação HTTP futura deverá entregar apenas artefatos já validados
 * pelos contratos Python. Componentes visuais não conhecem filesystem,
 * PyTorch, Flower ou detalhes do transporte.
 */
export interface ArtifactRepository {
  list(): Promise<readonly ArtifactDescriptor[]>;
  getById(artifactId: string): Promise<ValidatedArtifactEnvelope | null>;
}
