# Contratos científicos — E0 / versão 1.0

Implementação com **biblioteca padrão do Python**, sem dependência de FastAPI,
Pydantic, jsonschema, pandas, PyTorch ou Flower. Os schemas são dataclasses Python
com validação explícita; **não são uma implementação do padrão JSON Schema**.

E0 valida estrutura, valores e relações declaradas. Não executa experimentos,
preprocessing, treinamento, carregamento de checkpoint ou um servidor web. A
checagem aritmética de traces é pequena e independente de ML: verifica somente
se tensores fornecidos são coerentes com pesos fornecidos, sem inferência sobre
inputs brutos ou integração com Three.js.

## Uso

A partir da raiz do repositório:

```python
from contracts import load_artifact, parse_artifact, validate_bundle

document = load_artifact("results/historical/readme-results.json")
payload = document.to_dict()
assert parse_artifact(payload) == document
validate_bundle(document)
```

- `load_artifact(path)`: lê exclusivamente JSON e valida o schema.
- `parse_artifact(dict)`: seleciona o schema pelo discriminador `artifact_type`.
- `Manifest`, `Metrics`, `History`, `Model`, `Trace`: construtores validados,
  com `from_dict()` e `to_dict()`.
- `MetricValue`: schema de um valor de métrica.
- `validate_bundle(*artifacts)`: valida ligações entre documentos fornecidos.
- `validate_protocol(dict)`: valida o envelope documental de um protocolo.
- Falhas de contrato geram `ContractError`, subclasse de `ValueError`.
- Erros de filesystem em `load_artifact` continuam sendo erros de I/O; uma API
  futura deverá traduzi-los, sem apresentá-los como resultados científicos.

Todos os campos de cada documento são **obrigatórios**, mesmo quando aceitam
`null`. Não há preenchimento de valores científicos por default. Campos
desconhecidos são rejeitados. Números em strings não são convertidos;
booleanos não são aceitos como números. NaN e Infinity são proibidos.

O leitor JSON rejeita também chaves duplicadas, incluindo em objetos aninhados.
Para escrever, o consumidor deverá usar `json.dumps(..., allow_nan=False)`.
`from_dict()` copia os dados recebidos. As dataclasses são `frozen`, mas listas
e objetos internos não são profundamente imutáveis; `to_dict()` os revalida e
devolve outra cópia. Não usar `dataclasses.asdict()` como substituto da validação.

## Campos comuns

| Campo | Regra |
|---|---|
| `schema_version` | Exatamente `"1.0"` |
| `artifact_type` | `manifest`, `metrics`, `history`, `model` ou `trace` |
| `artifact_id` | ID não vazio: letras/números, `_`, `-`, `.`, até 128 caracteres |
| `evidence_kind` | Uma das três categorias abaixo |
| `source_refs` | Lista de fontes; obrigatoriamente não vazia no histórico documentado |

Categorias:

- `historical_documented`: transcrição de evidência preservada, não execução
  comprovada. `run_id` deve ser `null`. Modelo/trace não aceitam essa categoria.
- `historical_rerun`: reexecução do protocolo histórico, com ID próprio.
  Seeds historicamente desconhecidas podem continuar `null`.
- `new_experiment`: execução do protocolo novo, com ID próprio. Quando concluída,
  exige seeds explícitas e preprocessing não ajustado no dataset completo.

Não existe categoria que promova uma animação educativa a resultado científico.
O modo educativo da futura interface ficará fora do catálogo de experimentos.

Cada `source_ref` contém exatamente:

```text
commit      hash Git completo, 40 caracteres hexadecimais minúsculos
blob        hash completo do conteúdo Git, 40 caracteres
path        caminho relativo ao repositório, sem escape de diretórios
line_start  linha inicial, inclusiva, a partir de 1
line_end    linha final, inclusiva, >= line_start
```

Os hashes de 40 caracteres correspondem ao formato do repositório atual, não a
suporte universal a repositórios Git SHA-256. Hashes de artefatos científicos
externos usam SHA-256 de 64 caracteres.

## Manifesto — `manifest`

Além dos campos comuns:

| Campo | Tipo / significado |
|---|---|
| `run_id` | ID da execução ou `null` para documentação histórica |
| `protocol_id` | Identificador do protocolo |
| `comparison_group_id` | ID do grupo comparável, ou `null` quando não estabelecido |
| `status` | `documented`, `planned`, `running`, `completed`, `failed` |
| `source_commit` | Hash Git completo do código executado, ou `null` quando desconhecido |
| `source_dirty` | Estado do working tree observado: booleano ou `null` quando desconhecido |
| `source_files` | Mapa de caminhos relativos do código executado para SHA-256; vazio se não registrado |
| `dataset` | `{id, sha256, split_id, partition_id, sizes}` |
| `model` | `{family, scope, client_id}` |
| `seeds` | `{split, validation, partition, training}`, inteiros não negativos ou `null` |
| `environment` | Objeto de ambiente ou `null` |
| `preprocessing` | Identidade do pré-processador ou `null` |
| `decision` | Regra de classificação ou `null` |
| `configuration` | Objeto JSON com configuração integral; não executável |
| `artifacts` | Mapa de nome para `{path, sha256}` |
| `notes` | Lista de observações não vazias; a lista pode ser vazia |

`dataset.sizes` permite `train`, `validation`, `test`, com contagens inteiras
positivas; desconhecidos não são inventados. Os IDs de split/partição e hash do
dataset podem ser `null` quando desconhecidos.

`model.family`: `majority`, `logistic_regression`, `random_forest`,
`mlp_centralized`, `mlp_federated`, `mlp_local`. Escopo `local` é reservado a
`mlp_local` nesta versão e exige `client_id`; escopo `global` exige `null`.

`environment`: `{python, packages, platform, hardware, threads}`; `packages`
mapeia nomes para versões textuais, e `threads` é inteiro positivo. A declaração
de versão não comprova que um pacote estava instalado: isso depende do produtor
do artefato e das evidências da execução.

`preprocessing`: `{id, sha256, fit_scope}`. Escopos: `full_dataset`,
`global_train`, `local_train`. Este objeto identifica o transformador, não contém
ainda o schema de suas estatísticas internas.

`decision`: `{positive_label: 1, threshold: número em [0,1], operator: ">"}`.
A decisão usa o score integral, nunca o percentual arredondado para exibição.

Status `documented` só pode ser usado com `historical_documented`. Status
`completed` exige identificação do código/dataset, ambiente com versões,
e referências a artefatos, ao menos logs. A reexecução histórica pode terminar
sem preprocessing ou checkpoint salvo, porque o script original não os exporta;
isso não autoriza um trace real sem modelo correspondente. Um novo experimento
concluído também exige preprocessing, decisão, grupo comparável, split, quatro
seeds, tamanhos de treino/validação/teste e fit fora do dataset completo.

Novos experimentos concluídos exigem `source_dirty` e `source_files` preenchidos:
um commit isolado não identifica código não commitado. O contrato aceita working
tree com alterações, desde que registrado; cabe ao produtor incluir todos os
arquivos relevantes e verificar seus hashes. E0 não cria commits para satisfazer
esse requisito.

`configuration` é uma área de extensão JSON, não um schema de runner. Hiperparâmetros
específicos e a execução do protocolo serão validados na entrega do runner; E0
não cria um benchmark implicitamente.

## Valores de métricas

Cada valor tem exatamente:

```text
name              nome canônico
value             número finito ou null
unit              fraction ou loss
definition        definição/método não vazio
undefined_reason  motivo quando value=null; null quando há valor
reported_literal  literal documental ou null
```

Nomes e unidades:

- `accuracy`, `precision`, `recall`, `f1`, `average_precision`,
  `pr_auc_trapezoid`, `roc_auc`: `fraction`, intervalo `[0,1]`.
- `bce_loss`, `log_loss`: `loss`, número não negativo, sem limite superior de 1.

Não há nome ambíguo `pr_auc`: Average Precision e área trapezoidal têm nomes
distintos. A definição deve registrar a implementação e a convenção adotadas.

Histórico com valor exige o literal publicado e pode usar `as_reported` como
definição. Resultados executados não aceitam `as_reported` nem literal documental.
Precisão e métricas ausentes não são reconstruídas a partir de suposições.

## Métricas finais — `metrics`

Campos adicionais:

```text
run_id
evaluation_scope   train | validation | test | client_test | unknown
client_id          ID para client_test; null nos demais escopos
sample_count       inteiro positivo ou null apenas no histórico documentado
values             lista não vazia de MetricValue, sem nomes repetidos
confusion_matrix   objeto ou null
resources          objeto ou null
```

Uma avaliação executada exige escopo conhecido e contagem. `test` designa o
teste global; `client_test` designa um subconjunto identificado pelo cliente.

Matriz: `{labels: [0,1], orientation: "true_rows_predicted_columns",
counts: [[TN,FP],[FN,TP]]}`. Contagens inteiras não negativas; soma igual a
`sample_count`. Em resultados executados, accuracy/precision/recall/F1 presentes
são conferidos contra a matriz. No histórico, a comparação numérica é omitida
para não rejeitar valores publicados arredondados.

Convenções de indefinição:

- precision sem predições positivas: `null`, com motivo;
- recall sem positivos reais: `null`, com motivo;
- F1 com denominador zero: `null`, com motivo;
- F1 com positivos reais e nenhum verdadeiro positivo: zero é válido.

AP/AUC não são recalculadas na E0; exigem predições na futura avaliação.

`resources`: `{training_seconds, total_seconds, peak_memory_bytes,
measurement_scope}`. Números desconhecidos são `null`; valores existentes são
não negativos, memória é inteira e tempo total não pode ser menor que treino.
`measurement_scope` descreve o que foi medido; não presumir inclusão de workers.

## Histórico — `history`

Campos: `run_id`, `evaluation_scope`, `client_id`, `sample_count`, `step_type`,
`records`, `notes`. As regras de avaliação são as mesmas de `metrics`.

- `step_type`: `round` ou `epoch`.
- `records`: lista de `{step, metrics}`.
- `step`: inteiro não negativo, único e crescente; zero pode representar
  avaliação inicial, quando realmente registrada.
- Lacunas são permitidas e preservadas, sem interpolação ou preenchimento.
- `metrics`: lista de MetricValue.
- `notes`: lista de textos não vazios, podendo ser vazia.

## Modelo — `model`

O schema de modelo desta entrega descreve **pesos de MLP binária**, não árvores
ou coeficientes de todos os estimadores. Os manifestos contemplam todas as
famílias; exportadores específicos de RF/LR ficam para as entregas de modelos.

Campos: `run_id`, `model_sha256`, `preprocessing`, `dtype`, `weight_layout`,
`feature_order`, `layers`, `parameter_count`, `decision`.

- `dtype`: `float32` ou `float64`.
- `weight_layout`: exclusivamente `out_in`.
- `feature_order`: nomes únicos, sem `BAD`, na ordem real de entrada.
- Camada: `{name, input_dim, output_dim, activation, weights, bias}`.
- `weights`: matriz `[output_dim][input_dim]`; `bias`: vetor `[output_dim]`.
- Dimensões devem encadear, com ReLU nas ocultas e Sigmoid na saída de dimensão 1.
- `parameter_count` deve igualar a soma de pesos e biases; não é fixado em 353.
- `model_sha256` identifica o checkpoint externo; não é um hash autorreferente
  deste JSON. O produtor deverá calcular o hash real do arquivo.

## Trace — `trace`

Campos: `run_id`, `sample_id`, `model_id`, `model_sha256`, `preprocessing`,
`input_vector`, `layers`, `probability`, `predicted_label`, `decision`.

- `input_vector`: entrada já transformada, na ordem do modelo; aceita negativos.
- Camada: `{name, pre_activation, activation}`, com vetores finitos e de mesmo
  tamanho. Ativações ReLU acima de 1 são permitidas.
- Saída final escalar, igual à probabilidade em `[0,1]` dentro da tolerância.
- Classe coerente com `probability > threshold`; empate em 0.5 dá classe 0.
- Inputs brutos e estatísticas do transformador não são reconstruídos pela E0.

`validate_bundle` exige o manifesto para artefatos executados e o modelo para
traces. Confere IDs, categoria, hashes, preprocessing, decisão, tamanho do split
global quando disponível, shapes e operações `z=W×x+b`, ReLU/Sigmoid. Tolerância
das operações: absoluta `1e-6`, relativa `1e-5`. A consistência entre saída do
trace e probabilidade usa absoluta `1e-7`, relativa `1e-6`.

## Limites de validação

- Validar um documento **não comprova que o experimento aconteceu**.
- Hash bem formado não comprova que o arquivo correspondente existe ou possui
  aquele conteúdo. Os testes históricos verificam os blobs Git e seus literais;
  os futuros exportadores devem verificar artefatos de execução.
- Sem predições brutas, não se recalcula AP/AUC nem se prova a definição textual.
- Sem índices do dataset, não se prova ausência de sobreposição entre splits.
- Sem transformador completo, não se prova que input bruto virou aquele vetor.
- `validate_bundle` não completa artefatos ausentes e não acessa arquivos externos.
- Resultados parciais não são promovidos a completos pelo validador.
- Ainda não há endpoints, frontend, treino, exportação de checkpoints ou 3D real.

## Testes

```bash
python3 -B -m unittest discover -s tests -v
```

`-B` evita criação de bytecode. Fixtures sintéticas ficam apenas em
`tests/test_contracts.py`, identificadas como `synthetic`/`test-only`. Nenhuma
fixture é copiada para `results/`. Os testes históricos usam somente leituras
do filesystem e comandos Git de consulta; não importam os scripts de treino/EDA.

Mudanças incompatíveis de campos ou significado devem alterar a versão do
contrato. A integração futura com FastAPI deverá respeitar estas regras e testes,
sem introduzir coerção silenciosa por um segundo schema divergente.
