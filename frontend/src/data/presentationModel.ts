export const datasetSummary = {
  name: "HMEQ",
  domain: "Home Equity",
  samples: 5960,
  rawFeatures: 12,
  transformedFeatures: 18,
  target: "BAD",
  splits: [
    { id: "train", label: "Treino", count: 3576, fraction: 60 },
    { id: "validation", label: "Validação", count: 1192, fraction: 20 },
    { id: "test", label: "Teste", count: 1192, fraction: 20 },
  ],
  clients: ["client-0", "client-1", "client-2"].map((id) => ({ id, count: 1192 })),
  source: "results/data/benchmark-v1-split.json",
} as const;

export const methodologyTracks = [
  {
    id: "central",
    index: "01",
    title: "Centralizado",
    subtitle: "Referência global",
    description:
      "Uma MLP aprende sobre o conjunto global de treino congelado. Serve como comparador controlado, não como cenário privado.",
    status: "Componentes preparados",
    tone: "purple",
  },
  {
    id: "local",
    index: "02",
    title: "Local",
    subtitle: "Três modelos independentes",
    description:
      "Cada cliente simulado recebe somente sua partição de treino. Não há ensemble implícito nem troca de estados entre clientes.",
    status: "Componentes preparados",
    tone: "lavender",
  },
  {
    id: "federated",
    index: "03",
    title: "Federado",
    subtitle: "FedAvg com participação 3/3",
    description:
      "Atualizações locais serão agregadas por número de exemplos. O benchmark corrigido ainda não foi executado.",
    status: "Planejado · não executado",
    tone: "cyan",
  },
] as const;
