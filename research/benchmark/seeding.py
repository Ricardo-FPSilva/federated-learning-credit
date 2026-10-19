"""Derivação SHA-256 e configuração de streams determinísticos.

Configurações que precisam ocorrer antes da importação de bibliotecas numéricas:

- ``PYTHONHASHSEED``: deve ser definida antes do início do processo Python;
  a randomização de hash não é controlável em runtime.
- ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
  ``NUMEXPR_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS`` e ``BLIS_NUM_THREADS``:
  devem estar definidas antes do import de NumPy/PyTorch; seus valores são
  registrados como ambiente, nunca como threads efetivamente verificadas.

Seeds de Python, NumPy e PyTorch e ``torch.use_deterministic_algorithms(true)``
são aplicados por ``configure_determinism``, que distingue explicitamente
configurações solicitadas das efetivamente verificadas: ``threads_effective``
só é preenchido por leitura imediata (``get_num_threads``/
``get_num_interop_threads``) e a ausência de verificação é registrada como
``None`` com limitação documentada — nunca como single-thread afirmado.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

EXECUTION_SPEC_ID = "benchmark-v1-execution-1"
TRAINING_SEEDS = (42, 43, 44, 45, 46)
MAX_TORCH_SEED = 2**63 - 1
CANONICAL_THREAD_LIMIT = 1
BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class SeedingError(ValueError):
    """Seed ou namespace incompatível com a especificação congelada."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SeedingError(message)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialização canônica definida por benchmark-v1-execution-1."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def derive_seed(training_seed: int, stream: str, model: str,
                client_id: str | None = None, step: int | None = None,
                epoch: int | None = None) -> int:
    _require(type(training_seed) is int and training_seed in TRAINING_SEEDS,
             "training_seed deve ser uma das seeds 42–46")
    _require(type(stream) is str and bool(stream), "stream inválido")
    _require(type(model) is str and bool(model), "model inválido")
    _require(client_id is None or (type(client_id) is str and bool(client_id)),
             "client_id inválido")
    _require(step is None or (type(step) is int and step >= 0), "step inválido")
    _require(epoch is None or (type(epoch) is int and epoch >= 0), "epoch inválido")
    payload = {
        "execution_spec_id": EXECUTION_SPEC_ID,
        "training_seed": training_seed,
        "stream": stream,
        "model": model,
        "client_id": client_id,
        "step": step,
        "epoch": epoch,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") % MAX_TORCH_SEED


def paired_initialization_seed(training_seed: int) -> int:
    return derive_seed(training_seed, "mlp_initialization", "paired_mlp")


def shuffle_seed(training_seed: int, namespace: str, *, client_id: str | None,
                 step: int, epoch: int) -> int:
    _require(namespace in {"central", "federated", "local"}, "namespace de shuffle inválido")
    if namespace == "central":
        _require(client_id is None, "shuffle central não possui client_id")
    else:
        _require(client_id in {"client-0", "client-1", "client-2"},
                 "shuffle federated/local exige client_id congelado")
    return derive_seed(training_seed, "shuffle", namespace, client_id, step, epoch)


def configure_determinism(seed: int, *, numpy_module: Any | None = None,
                          torch_module: Any | None = None,
                          environ: dict[str, str] | None = None) -> dict:
    """Aplica seeds e limita threads; configura módulos fornecidos sem importá-los.

    O retorno separa solicitado de verificado: ``threads_requested`` é a política
    canônica (uma thread), ``threads_effective`` só vem de leitura imediata do
    PyTorch e ``environment_variables`` reflete o ambiente (BLAS/OpenMP e
    PYTHONHASHSEED), cujo efeito não é verificável dentro do processo.
    """
    _require(type(seed) is int and 0 <= seed < MAX_TORCH_SEED, "seed derivada inválida")
    env = dict(os.environ) if environ is None else dict(environ)
    random.seed(seed)
    limitations = [
        "PYTHONHASHSEED e as variáveis BLAS/OpenMP só têm efeito quando definidas "
        "antes da importação das bibliotecas numéricas; os valores registrados "
        "refletem o ambiente e não constituem verificação de threads efetivas.",
    ]
    numpy_seeded = False
    if numpy_module is not None:
        numpy_module.random.seed(seed % 2**32)
        numpy_seeded = True
    torch_report = {
        "configured": False,
        "manual_seed": False,
        "deterministic_algorithms": False,
        "threads_requested": None,
        "threads_effective": None,
        "interop_threads_requested": None,
        "interop_threads_effective": None,
        "interop_error": None,
    }
    if torch_module is None:
        limitations.append(
            "torch não fornecido: threads e determinismo PyTorch não configurados "
            "nem verificados"
        )
    else:
        torch_module.manual_seed(seed)
        torch_module.use_deterministic_algorithms(True)
        torch_report["configured"] = True
        torch_report["manual_seed"] = True
        torch_report["deterministic_algorithms"] = True
        torch_report["threads_requested"] = CANONICAL_THREAD_LIMIT
        torch_module.set_num_threads(CANONICAL_THREAD_LIMIT)
        threads_effective = int(torch_module.get_num_threads())
        torch_report["threads_effective"] = threads_effective
        if threads_effective != CANONICAL_THREAD_LIMIT:
            limitations.append(
                f"torch num_threads efetivo {threads_effective} difere do "
                f"solicitado {CANONICAL_THREAD_LIMIT}"
            )
        torch_report["interop_threads_requested"] = CANONICAL_THREAD_LIMIT
        try:
            torch_module.set_num_interop_threads(CANONICAL_THREAD_LIMIT)
        except RuntimeError as exc:
            torch_report["interop_error"] = str(exc)
            limitations.append(f"set_num_interop_threads não aplicado: {exc}")
        try:
            interop_effective = int(torch_module.get_num_interop_threads())
            torch_report["interop_threads_effective"] = interop_effective
            if interop_effective != CANONICAL_THREAD_LIMIT:
                limitations.append(
                    f"torch interop threads efetivas {interop_effective} difere do "
                    f"solicitado {CANONICAL_THREAD_LIMIT}"
                )
        except RuntimeError as exc:
            limitations.append(f"não foi possível verificar interop threads: {exc}")
    environment_variables = {"PYTHONHASHSEED": env.get("PYTHONHASHSEED")}
    for name in BLAS_THREAD_ENV_VARS:
        environment_variables[name] = env.get(name)
    if environment_variables["PYTHONHASHSEED"] is None:
        limitations.append(
            "PYTHONHASHSEED ausente: a randomização de hash do Python não pode ser "
            "controlada após o início do processo"
        )
    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": numpy_seeded,
        "torch": torch_report,
        "environment_variables": environment_variables,
        "limitations": limitations,
    }
