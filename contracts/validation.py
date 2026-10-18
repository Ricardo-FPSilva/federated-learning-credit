"""Validação dos contratos E0. Não implementa o padrão JSON Schema."""

import json
import math
import re
from pathlib import Path, PurePosixPath


class ContractError(ValueError):
    """Documento inválido; a mensagem identifica o campo ou a relação."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def keys(value, expected, path):
    require(type(value) is dict, f"{path}: esperado objeto")
    require(all(type(key) is str for key in value), f"{path}: chaves devem ser textuais")
    missing = set(expected) - value.keys()
    extra = value.keys() - set(expected)
    require(not missing and not extra,
            f"{path}: campos ausentes={sorted(missing)}, desconhecidos={sorted(extra)}")


def text(value, path):
    require(type(value) is str and bool(value.strip()), f"{path}: texto não vazio")


def identifier(value, path):
    text(value, path)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is not None,
            f"{path}: identificador inválido")


def integer(value, path, minimum=0):
    require(type(value) is int and value >= minimum,
            f"{path}: inteiro >= {minimum} (booleanos não são inteiros válidos)")


def number(value, path, minimum=None, maximum=None):
    require(type(value) in (int, float), f"{path}: número esperado")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, f"{path}: número deve ser finito")
    require(minimum is None or value >= minimum, f"{path}: abaixo do mínimo")
    require(maximum is None or value <= maximum, f"{path}: acima do máximo")


def choice(value, values, path):
    require(type(value) is str and value in values, f"{path}: opção inválida")


def digest(value, path, length=64):
    require(type(value) is str and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None,
            f"{path}: hash hexadecimal de {length} caracteres esperado")


def sequence(value, path, nonempty=True):
    require(type(value) is list, f"{path}: lista esperada")
    require(not nonempty or bool(value), f"{path}: lista vazia")


def vector(value, path):
    sequence(value, path)
    for index, item in enumerate(value):
        number(item, f"{path}[{index}]")


def json_value(value, path="document"):
    """Recusa NaN/Infinity, chaves não textuais e objetos não JSON."""
    if value is None or type(value) in (str, bool):
        return
    if type(value) in (int, float):
        number(value, path)
    elif type(value) is list:
        for index, item in enumerate(value):
            json_value(item, f"{path}[{index}]")
    elif type(value) is dict:
        for key, item in value.items():
            text(key, path)
            json_value(item, f"{path}.{key}")
    else:
        raise ContractError(f"{path}: valor não serializável como JSON")


def source_refs(value):
    sequence(value, "source_refs", nonempty=False)
    for ref in value:
        keys(ref, {"commit", "blob", "path", "line_start", "line_end"}, "source_ref")
        digest(ref["commit"], "source_ref.commit", 40)
        digest(ref["blob"], "source_ref.blob", 40)
        relative_path(ref["path"], "source_ref.path")
        integer(ref["line_start"], "source_ref.line_start", 1)
        integer(ref["line_end"], "source_ref.line_end", ref["line_start"])


def relative_path(value, path):
    text(value, path)
    parts = value.split("/")
    require(not PurePosixPath(value).is_absolute()
            and all(p not in ("", ".", "..") for p in parts)
            and "\\" not in value and ":" not in value
            and all(ord(character) >= 32 and ord(character) != 127 for character in value),
            f"{path}: caminho relativo seguro esperado")


def decision(value):
    keys(value, {"positive_label", "threshold", "operator"}, "decision")
    integer(value["positive_label"], "decision.positive_label")
    require(value["positive_label"] == 1, "decision: classe positiva deve ser BAD=1")
    number(value["threshold"], "decision.threshold", 0, 1)
    choice(value["operator"], {">"}, "decision.operator")


def strict_json(text_value):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"JSON: chave duplicada {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise ContractError(f"JSON: constante não finita {value}")

    try:
        result = json.loads(text_value, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"JSON inválido: {exc}") from exc
    json_value(result)
    return result


def parse_artifact(value):
    from .schemas import History, Manifest, Metrics, Model, Trace

    require(type(value) is dict, "artifact: esperado objeto")
    registry = {cls.artifact_type: cls for cls in (Manifest, Metrics, History, Model, Trace)}
    kind = value.get("artifact_type")
    choice(kind, registry, "artifact_type")
    return registry[kind].from_dict(value)


def load_artifact(path):
    """Lê apenas JSON; não carrega pickle, checkpoint ou código Python."""
    return parse_artifact(strict_json(Path(path).read_text(encoding="utf-8")))


def validate_protocol(value):
    """Valida o envelope documental; regras são texto, não configuração executável."""
    keys(value, {"schema_version", "protocol_id", "status", "kind", "description",
                 "source_refs", "rules", "limitations"}, "protocol")
    require(value["schema_version"] == "1.0", "protocol: versão não suportada")
    identifier(value["protocol_id"], "protocol_id")
    choice(value["status"], {"documented", "proposed", "frozen"}, "protocol.status")
    choice(value["kind"], {"historical", "corrected"}, "protocol.kind")
    text(value["description"], "protocol.description")
    source_refs(value["source_refs"])
    for name in ("rules", "limitations"):
        sequence(value[name], name)
        for item in value[name]:
            text(item, name)
    if value["kind"] == "historical":
        require(value["status"] == "documented" and bool(value["source_refs"]),
                "protocol: histórico exige fontes e status documented")
    else:
        require(value["status"] != "documented", "protocol: corrigido não é histórico")
    return value


def validate_bundle(*artifacts):
    """Confere vínculos entre os artefatos fornecidos, sem acessar arquivos externos.

    Não certifica execução nem integridade criptográfica do conteúdo externo.
    """
    from .schemas import Manifest, Model, Trace

    require(bool(artifacts), "bundle: nenhum artefato")
    parsed = [parse_artifact(a.to_dict() if hasattr(a, "to_dict") else a) for a in artifacts]
    ids = [a.artifact_id for a in parsed]
    require(len(ids) == len(set(ids)), "bundle: artifact_id duplicado")
    run_manifests = [a for a in parsed if isinstance(a, Manifest) and a.run_id is not None]
    manifests = {a.run_id: a for a in run_manifests}
    require(len(manifests) == len(run_manifests),
            "bundle: manifestos duplicados para run_id")
    models = {a.artifact_id: a for a in parsed if isinstance(a, Model)}
    for artifact in parsed:
        if artifact.evidence_kind == "historical_documented":
            continue
        manifest = manifests.get(artifact.run_id)
        if not isinstance(artifact, Manifest):
            require(manifest is not None, "bundle: manifesto da execução ausente")
            require(manifest.evidence_kind == artifact.evidence_kind,
                    "bundle: categorias de evidência divergentes")
        if isinstance(artifact, Model):
            require(manifest.model["family"] in {"mlp_centralized", "mlp_federated", "mlp_local"},
                    "bundle: pesos MLP associados a outra família")
            require(manifest.preprocessing is not None, "bundle: preprocessing ausente")
            require(artifact.preprocessing == manifest.preprocessing,
                    "bundle: preprocessing do modelo difere do manifesto")
            require(artifact.decision == manifest.decision, "bundle: decisão divergente")
        if artifact.artifact_type in {"metrics", "history"}:
            sizes = manifest.dataset["sizes"]
            if artifact.evaluation_scope in sizes:
                require(artifact.sample_count == sizes[artifact.evaluation_scope],
                        "bundle: sample_count difere do split")
        if isinstance(artifact, Trace):
            model = models.get(artifact.model_id)
            require(model is not None, "bundle: modelo do trace ausente")
            require(model.run_id == artifact.run_id, "bundle: trace de outra execução")
            require(model.evidence_kind == artifact.evidence_kind, "bundle: evidência do trace divergente")
            require(model.model_sha256 == artifact.model_sha256, "bundle: hash de modelo divergente")
            require(model.preprocessing == artifact.preprocessing, "bundle: preprocessing do trace divergente")
            require(model.decision == artifact.decision, "bundle: decisão do trace divergente")
            require(len(artifact.input_vector) == len(model.feature_order), "bundle: dimensão de entrada incorreta")
            require(len(model.layers) == len(artifact.layers), "bundle: número de camadas incorreto")
            previous = artifact.input_vector
            for layer, observed in zip(model.layers, artifact.layers):
                require(layer["name"] == observed["name"], "bundle: nome de camada divergente")
                require(len(observed["activation"]) == layer["output_dim"], "bundle: shape de ativação incorreto")
                expected_z = [sum(w * x for w, x in zip(row, previous)) + bias
                              for row, bias in zip(layer["weights"], layer["bias"])]
                for expected, actual in zip(expected_z, observed["pre_activation"]):
                    require(math.isclose(expected, actual, abs_tol=1e-6, rel_tol=1e-5),
                            "bundle: pré-ativação não corresponde aos pesos e entrada")
                for z, actual in zip(observed["pre_activation"], observed["activation"]):
                    if layer["activation"] == "relu":
                        expected = max(0, z)
                    else:
                        expected = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                    require(math.isclose(expected, actual, abs_tol=1e-6, rel_tol=1e-5),
                            "bundle: ativação incompatível com a função declarada")
                previous = observed["activation"]
    return tuple(parsed)
