# Protocolos — fronteira entre documentação e execução

## Documentos da E0

| Arquivo | Categoria | Status |
|---|---|---|
| `historical-v1.json` | Comportamento da implementação histórica | `documented` |
| `benchmark-v1.json` | Proposta metodológica corrigida | `proposed` |

Esses arquivos são **especificações documentais**, não configurações de um runner
já existente. O histórico foi descrito com referências verificáveis; o benchmark
não foi executado e ainda não está congelado como protocolo experimental.

Cada arquivo contém: `schema_version`, `protocol_id`, `status`, `kind`,
`description`, `source_refs`, `rules`, `limitations`. O envelope é validado por
`contracts.validate_protocol`. Regras e limitações são listas de textos; E0 não
interpreta esses textos como comandos nem valida execução de hiperparâmetros.

## Preservação

Não modificar os arquivos históricos para acomodar contratos ou dependências:

- `README.md`;
- `hmeq.csv`;
- `data_analysis.py`;
- `machine_learning.py`.

Os testes comparam seus hashes de blob Git aos valores do snapshot original.
O framework/API e o código científico novo serão separados em entregas futuras.

## Categorias de evidência

- **Histórico documentado:** valores do README/dashboard, sem comprovação da
  execução, do ambiente ou das seeds que os produziram.
- **Reexecução histórica:** nova execução identificada do protocolo original;
  ambiente reconstruído não é automaticamente ambiente original recuperado.
- **Novo experimento:** protocolo corrigido, sem promover seus resultados a
  correção retroativa dos números históricos.

As diferenças entre histórico e benchmark incluem split, representação
categórica e controle de aleatoriedade, além das correções de leakage/loss.
Portanto, não se pode atribuir toda diferença de desempenho a uma única correção.

## Condições antes de congelar o benchmark

1. Aprovar configurações científicas detalhadas, separadamente da arquitetura.
2. Auditar duplicatas/grupos e congelar índices de split e partição.
3. Definir explicitamente tratamento de categorias desconhecidas e colunas vazias.
4. Registrar versões, recursos, política de falhas e métrica primária.
5. Executar testes de dados/avaliação sem consultar resultados do teste final.
6. Só então marcar a especificação como `frozen` para a campanha autorizada.

O preprocessing global de treino é uma simplificação controlada. Mesmo sem
leakage de holdout, ele compartilha estatísticas entre clientes e não demonstra
privacidade ponta a ponta. FedAvg não precisa superar os baselines para que o
resultado seja cientificamente válido.

## E1

A recuperação do ambiente exige autorização própria para criar ambientes,
instalar/baixar pacotes ou executar scripts. E0 não instala dependências e não
altera a recomendação histórica de instalação no README; apenas a preserva como
documento. A próxima etapa deve evitar `--break-system-packages`, isolar o runtime
legado e não migrar silenciosamente a API Flower no original.
