export type EvidenceKind =
  | "historical_documented"
  | "historical_rerun"
  | "new_experiment";

export type ArtifactType = "manifest" | "metrics" | "history" | "model" | "trace";

export type ArtifactAvailability = "catalogued" | "completed" | "not_executed";

/** Envelope mínimo esperado após validação autoritativa pelo backend. */
export interface ValidatedArtifactEnvelope {
  readonly schema_version: "1.0";
  readonly artifact_type: ArtifactType;
  readonly artifact_id: string;
  readonly evidence_kind: EvidenceKind;
  readonly source_refs: readonly unknown[];
}

/** Metadados seguros para navegação antes do carregamento do JSON completo. */
export interface ArtifactDescriptor {
  readonly id: string;
  readonly title: string;
  readonly evidenceKind: EvidenceKind;
  readonly availability: ArtifactAvailability;
  readonly sourcePath: string | null;
  readonly summary: string;
  readonly caveat: string;
}
