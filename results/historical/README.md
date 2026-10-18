# Evidências históricas documentadas

Estes JSONs são transcrições rastreáveis, **não logs, checkpoints, resultados de
treinamento executado nesta entrega ou comprovação de reprodução**.

| Registro | Fonte | Commit |
|---|---|---|
| `readme-results.json` | `README.md:132–140` | `725f203d0bdcfec1a2c6e5fcb95a258562d7597a` |
| `dashboard-results.json` | `analysis_dashboard/script.js:2–5` | `da6612a6af626c51b91061752218655817b671d1` |

Cada fonte inclui também o hash de seu blob Git. O dashboard está na árvore de
`origin/feature` disponível no clone; não foi feito checkout nem incorporada a
interface à `main`.

## O que foi preservado

- README: somente rounds 1, 5 e 10, com accuracy e F1.
- Dashboard: dez rounds, com accuracy, F1 e loss conforme constantes JavaScript.
- Literais originais, incluindo precisão decimal e percentuais do README.
- Accuracy do round 1: `83.97%` no README e `0.8398` no dashboard.
- F1 do round 5: `0.589` no README e `0.5895` no dashboard.

Não foi presumida uma regra comum de arredondamento, nem que as duas fontes
vieram da mesma execução. Percentuais do README foram convertidos para fração
apenas para uniformizar unidades; o literal original continua presente.

O nome canônico `bce_loss` para a constante `LOSS` reflete o contexto do código
PyTorch, não uma verificação da origem experimental. A definição é `as_reported`;
não deve ser comparada silenciosamente com BCE corrigida por exemplo.

## O que não foi preenchido

- `run_id`: `null`, sem vínculo comprovado com execução preservada;
- `sample_count`: `null`, sem presumir que a execução publicada usou exatamente
  o mesmo split que o código atual descreve;
- `evaluation_scope`: `unknown`;
- ambiente, seeds de treinamento, checkpoint e predições: não disponíveis;
- rounds e loss ausentes do README: não interpolados ou copiados do dashboard.

Os testes verificam schema, conversão de unidades e valores contra os blobs e
intervalos de linhas citados. Também verificam que os quatro arquivos históricos
permanecem byte a byte iguais ao snapshot original. Testes que consultam Git
podem ser ignorados se este diretório for distribuído sem `.git`; isso não equivale
a comprovar a proveniência naquela distribuição.

Novos experimentos devem gerar artefatos automaticamente por exportadores
próprios em entregas posteriores. Não atualizar estes registros para substituir
resultados antigos por valores obtidos depois.
