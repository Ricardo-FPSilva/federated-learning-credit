"""Checks E1 sem treino ou inicialização do Ray; relatório JSON em stdout.

Executar com .venv-historical/bin/python -B a partir da raiz do projeto.
Não instala pacotes e não escreve arquivos. Tensores sintéticos servem somente
para smoke tests de interoperabilidade, nunca como resultados experimentais.
"""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
RECORDS = Path(__file__).resolve().parent
EXPECTED_VENV = ROOT / ".venv-historical"
EXPECTED_PACKAGES = {
    "flwr": "1.17.0",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scikit-learn": "1.5.2",
    "protobuf": "4.25.6",
    "torch": "2.5.1+cpu",
    "ray": "2.31.0",
}
ORIGINAL_BLOBS = {
    "README.md": "b21af9fc1c310d2f9879ee2260c30c86371c60b4",
    "data_analysis.py": "36505e80533fb58cc7d92cc103341c1d68172440",
    "machine_learning.py": "3fe7d3fe44ddd2fa281fc5037a07432af87b9701",
    "hmeq.csv": "669687b31b616ff4030ef015c4d275d6b446db97",
}


def ensure(condition, description):
    if not condition:
        raise RuntimeError(description)


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    ensure(sys.prefix != sys.base_prefix, "Executar somente em virtualenv")
    ensure(Path(sys.prefix).resolve() == EXPECTED_VENV.resolve(),
           "Interpretador não pertence à .venv-historical deste repositório")
    ensure(sys.version_info[:2] == (3, 12), "Python 3.12 obrigatório")

    # Controles locais ao processo; não alteram configuração do sistema.
    os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    sys.path.insert(0, str(ROOT))

    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import torch
    import ray
    import flwr
    import google.protobuf
    from flwr.client.client_app import _inspect_maybe_adapt_client_fn_signature
    import machine_learning as historical

    purelib = Path(sysconfig.get_path("purelib"))
    packages = {}
    for distribution in metadata.distributions(path=[str(purelib)]):
        name = canonical_name(distribution.metadata["Name"])
        ensure(name not in packages, f"Distribuição duplicada: {name}")
        packages[name] = distribution.version
    packages = dict(sorted(packages.items()))
    for name, version in EXPECTED_PACKAGES.items():
        ensure(packages.get(name) == version, f"Versão inesperada: {name}")

    lock_text = (RECORDS / "requirements.lock.txt").read_text(encoding="utf-8")
    locked = {
        canonical_name(name): version
        for name, version in re.findall(
            r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^\s\\]+)", lock_text, re.MULTILINE
        )
    }
    ensure(locked == packages, "Inventário instalado diverge do lock")
    ensure(not any(name.startswith("nvidia-") for name in packages),
           "Pacotes CUDA inesperados no ambiente CPU")
    ensure("fastapi" not in packages, "Não misturar aplicação web no ambiente histórico")

    for module in (np, pd, scipy, sklearn, torch, ray, flwr, google.protobuf):
        ensure(Path(module.__file__).resolve().is_relative_to(EXPECTED_VENV.resolve()),
               f"Import fora do virtualenv: {module.__name__}")
    ensure(torch.version.cuda is None, "Build do PyTorch não é CPU-only")
    ensure(not ray.is_initialized(), "Ray não deveria ter sido inicializado")

    # Nenhum otimizador, backward, fit ou main é executado.
    sample = np.zeros((2, 12), dtype=np.float32)
    tensor = torch.from_numpy(sample)
    ensure(np.array_equal(tensor.numpy(), sample), "Round-trip NumPy/PyTorch falhou")
    neural_network = historical.RedeNeural(input_dim=12)
    neural_network.eval()
    with torch.inference_mode():
        output = neural_network(tensor)
    ensure(tuple(output.shape) == (2, 1), "Shape de saída inesperado")
    ensure(bool(torch.isfinite(output).all()) and bool(((output >= 0) & (output <= 1)).all()),
           "Saída inválida no smoke test de forward")
    parameter_count = sum(parameter.numel() for parameter in neural_network.parameters())
    ensure(parameter_count == 353, "Arquitetura histórica divergente")
    ensure(issubclass(historical.ClienteHMEQ, flwr.client.NumPyClient),
           "Classe de cliente incompatível com Flower")
    ensure(callable(historical.ClienteHMEQ.to_client), "Conversão to_client ausente")

    simulation_signature = inspect.signature(flwr.simulation.start_simulation)
    for name in ("client_fn", "num_clients", "config", "strategy", "client_resources"):
        ensure(name in simulation_signature.parameters, f"Parâmetro legado ausente: {name}")

    def legacy_factory(cid: str):
        return cid

    # API privada usada apenas para verificar o adaptador da versão fixada.
    # O Flower emite depreciação esperada; nenhum cliente ou servidor é iniciado.
    adapted_factory = _inspect_maybe_adapt_client_fn_signature(legacy_factory)
    ensure(adapted_factory(SimpleNamespace(node_config={"partition-id": 2}, node_id=99)) == "2",
           "Adaptador legado não preserva o partition-id")

    original_hashes = {}
    for name, expected_blob in ORIGINAL_BLOBS.items():
        data = (ROOT / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        ensure(blob == expected_blob, f"Arquivo histórico modificado: {name}")
        original_hashes[name] = {"git_blob": blob, "sha256": hashlib.sha256(data).hexdigest()}

    uv_version = subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    return {
        "record_type": "reconstructed_environment_validation",
        "record_version": "1.0",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "original_environment_proven": False,
        "training_executed": False,
        "ray_runtime_started": False,
        "uv_version": uv_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "environment_path": ".venv-historical",
        "package_count": len(packages),
        "packages": packages,
        "checks": {
            "virtualenv_isolated": "passed",
            "imports_from_virtualenv": "passed",
            "installed_versions_match_lock": "passed",
            "authorized_direct_versions": "passed",
            "torch_cpu_only": "passed",
            "numpy_torch_roundtrip": "passed",
            "historical_forward_without_training": "passed",
            "historical_parameter_count": parameter_count,
            "flower_numpy_client_interface": "passed",
            "flower_start_simulation_signature": "passed",
            "legacy_cid_adapter": "passed_with_expected_deprecation_warning",
            "historical_files_preserved": "passed",
        },
        "record_hashes": {
            name: sha256(RECORDS / name)
            for name in ("requirements.in", "requirements.lock.txt", "verify_environment.py")
        },
        "historical_files": original_hashes,
        "limitations": [
            "Sem ray.init, processos de simulação, treinamento ou avaliação do dataset.",
            "Forward usou tensores sintéticos apenas como smoke test; não gera métricas de pesquisa.",
            "Hashes de wheels foram exigidos na instalação; este check não revalida cada arquivo instalado.",
            "Não é uma auditoria de vulnerabilidades nem prova de reprodução dos resultados históricos.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2, allow_nan=False))
