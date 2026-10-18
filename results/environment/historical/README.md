# E1 — ambiente histórico reconstruído com uv

Este diretório registra uma **reconstrução instalada e validada por smoke tests**, não a
recuperação comprovada do ambiente que produziu os números do README/dashboard.
Os scripts originais e o CSV devem permanecer intactos.

## Separação de ambientes

- `.venv-historical/`: reservado ao código histórico e à sua validação.
- O ambiente da aplicação web será separado, por exemplo `.venv-web/`.
- FastAPI, ferramentas frontend e dependências da aplicação não são adicionados
  intencionalmente a este ambiente. Dependências transitivas do Flower podem
  incluir bibliotecas de infraestrutura, sem configurar uma aplicação web.
- Não usar `--system`, `--system-site-packages` ou `--break-system-packages`.
- Não baixar outro Python automaticamente: usar o Python 3.12 já disponível.

## Especificação autorizada

`requirements.in` fixa Flower 1.17.0 com extra `simulation`, NumPy 1.26.4,
pandas 2.2.3, scikit-learn 1.5.2, Protobuf 4.25.6 e PyTorch 2.5.1 CPU.
O sufixo `+cpu` identifica a distribuição CPU dessa versão do PyTorch.

As dependências foram resolvidas conjuntamente e instaladas sem conflito ou
alteração dos pins autorizados. Em futuras resoluções, se houver conflito ou
ausência de wheel compatível, interromper sem alterar os pins automaticamente.

## Instalação realizada em 18/10/2026

| Item | Resultado observado |
|---|---|
| Gerenciador | uv 0.12.10 (x86_64-unknown-linux-gnu) |
| Interpretador | CPython 3.12.14 já disponível; nenhum Python baixado |
| Plataforma | Linux x86_64, glibc 2.41 |
| Ambiente | `.venv-historical`, sem system-site-packages |
| Resolução e instalação | 55 distribuições; resolução conjunta |
| Flower / Ray | 1.17.0 / 2.31.0 |
| NumPy / pandas | 1.26.4 / 2.2.3 |
| scikit-learn / Protobuf | 1.5.2 / 4.25.6 |
| PyTorch | 2.5.1+cpu; `torch.version.cuda` é `None` |
| Hashes de distribuição | Exigidos por `uv pip sync --require-hashes` |
| Dependências compartilhadas | `uv pip check`: 55 pacotes compatíveis |
| Reaplicação do lock em dry-run | `Would make no changes` |
| Testes E0 neste ambiente | 49 testes aprovados, sem skips ou falhas |
| Treinamento e runtime Ray | Não executados/inicializados |

Não houve mudança do Python do sistema, dos pins diretos, dos scripts históricos,
do dataset ou do ambiente web. Não foram feitos commit ou push.

## Registros preservados

- `requirements.in`: seis dependências diretas autorizadas.
- `requirements.lock.txt`: resolução completa com versões e hashes; preservar
  para reinstalar a mesma combinação sem escolher novas transitivas.
- `verify_environment.py`: verificador repetível, sem treinamento ou escrita de
  arquivos, que produz um relatório JSON em stdout.
- `validation.json`: registro da saída da verificação realizada, com todas as
  55 versões efetivas, hashes dos registros e dos quatro arquivos históricos.

As transitivas foram resolvidas na data desta reconstrução e incluem versões
mais recentes que os commits do projeto. Isso é registrado, não tratado como
prova de recuperação das versões usadas originalmente. `validation.json` é um
registro de ambiente, não um manifesto de experimento ou conjunto de métricas.

## Comandos de preparação

Executar na raiz do repositório, com autorização para criar/instalar o ambiente.
Não executar `uv venv` sobre um diretório existente sem antes inspecioná-lo.

```bash
uv venv --python python3.12 --no-python-downloads --no-project \
  --cache-dir .uv-cache .venv-historical

uv pip compile results/environment/historical/requirements.in \
  --python .venv-historical/bin/python \
  --python-platform x86_64-unknown-linux-gnu \
  --torch-backend cpu --default-index https://pypi.org/simple \
  --only-binary :all: --generate-hashes --no-strip-extras \
  --emit-index-annotation --no-python-downloads --cache-dir .uv-cache \
  --output-file results/environment/historical/requirements.lock.txt

uv pip sync results/environment/historical/requirements.lock.txt \
  --python .venv-historical/bin/python \
  --torch-backend cpu --default-index https://pypi.org/simple \
  --require-hashes --only-binary :all: --strict \
  --no-python-downloads --cache-dir .uv-cache

uv pip check --python .venv-historical/bin/python
```

O lock é específico da resolução Linux x86_64/Python 3.12. Não é um lock
universal nem uma garantia de execução em outros sistemas. O uso de wheels
evita compilar distribuições-fonte durante esta recuperação.

Para repetir uma instalação já resolvida, usar o lock existente com `uv pip sync`;
não recompilar `requirements.in` silenciosamente, pois transitivas podem mudar.
O sync é destinado somente ao ambiente dedicado: ele pode remover pacotes que
não constem do lock. Não apontá-lo para o ambiente web ou para o Python do sistema.

## Limites desta etapa

- Validar imports e metadados não equivale a reproduzir as métricas históricas.
- Não iniciar `ray.init`, `start_simulation` ou `main()` durante os checks.
- Nenhum treinamento está autorizado nesta etapa.
- Versões legadas são usadas para reprodução local; isto não é certificação de
  ausência de vulnerabilidades nem recomendação para deployment público.
- Resultados da instalação e dos checks estão registrados acima e em
  `validation.json`. Eles não incluem resultados de treinamento.

## Repetir verificações sem treinamento

```bash
.venv-historical/bin/python -B results/environment/historical/verify_environment.py
.venv-historical/bin/python -B -m unittest discover -s tests -v
uv pip check --python .venv-historical/bin/python
```

O verificador desabilita telemetria Flower e estatísticas de uso Ray somente no
próprio processo. Verifica isolamento, imports, versões, lock, build CPU,
interoperabilidade NumPy/PyTorch, assinatura de `start_simulation`, adaptação de
`client_fn(cid)`, classe NumPyClient e os hashes originais.

Também realiza um forward em modo de inferência com duas entradas sintéticas
zero, sem `backward`, otimizador, `fit`, treino ou leitura do dataset para avaliação.
Os valores aleatórios da inicialização não são publicados como métricas.

O adaptador legado do Flower emite um aviso de depreciação esperado. Ele não foi
suprimido nem corrigido alterando o código original. Compatibilidade de imports
e assinatura não comprova que o runtime distribuído concluirá a simulação.

## Comando de reexecução histórica — NÃO executado nesta etapa

Executar **somente após autorização de treinamento**, a partir da raiz do
repositório, pois o script usa o caminho relativo `hmeq.csv`:

```bash
FLWR_TELEMETRY_ENABLED=0 RAY_USAGE_STATS_ENABLED=0 \
  .venv-historical/bin/python -B -u machine_learning.py
```

Esse comando mantém o script e seus hiperparâmetros intactos. `-B` evita bytecode;
`-u` disponibiliza logs sem buffering. As duas variáveis desabilitam telemetria,
não ajustam seeds, rounds, preprocessing ou treinamento.

A execução iniciará processos Ray, consumirá CPU/memória e poderá criar arquivos
temporários do runtime. Antes dela, preparar captura de stdout/stderr e manifesto
próprio, sem sobrescrever os registros históricos. Ainda não existe resultado
reproduzido de treinamento nem garantia de recuperar exatamente 88%/F1 0.632.
