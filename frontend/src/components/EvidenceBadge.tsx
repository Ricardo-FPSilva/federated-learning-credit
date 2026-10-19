import type { ArtifactAvailability, EvidenceKind } from "../data/artifactTypes";

const labels: Record<EvidenceKind, string> = {
  historical_documented: "Histórico documentado",
  historical_rerun: "Reexecução histórica",
  new_experiment: "Benchmark corrigido",
};

interface EvidenceBadgeProps {
  readonly evidenceKind: EvidenceKind;
  readonly availability: ArtifactAvailability;
}

export function EvidenceBadge({ evidenceKind, availability }: EvidenceBadgeProps) {
  const state = availability === "not_executed" ? "Não executado" : labels[evidenceKind];

  return (
    <span className={`evidence-badge evidence-${evidenceKind}`}>
      <span aria-hidden="true" />
      {state}
    </span>
  );
}
