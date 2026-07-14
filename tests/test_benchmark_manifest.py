import ast
import json
import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    return json.loads((ROOT / "benchmark_manifest.json").read_text(encoding="utf-8"))


def _assignment(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    raise AssertionError(f"{name} not found in {path}")


def test_manifest_datasets_match_loader():
    datasets = list(ast.literal_eval(_assignment(ROOT / "util" / "data.py", "SUPPORTED_DATASETS")))
    assert _manifest()["datasets"] == datasets


def test_manifest_methods_match_registry_decorators():
    registered = set()
    for path in (ROOT / "model" / "methods").glob("*.py"):
        registered.update(re.findall(r"@register\(['\"]([^'\"]+)['\"]\)", path.read_text(encoding="utf-8")))
    assert {item["id"] for item in _manifest()["methods"]} == registered


def test_manifest_backbones_match_model_registry():
    manifest = _manifest()
    registry = _assignment(ROOT / "util" / "profiling.py", "model_registry")
    implemented = {key.value for key in registry.keys if isinstance(key, ast.Constant)}
    assert {item["id"] for item in manifest["backbones"]} == implemented


def test_manifest_noise_types_match_dispatcher():
    implemented = set(ast.literal_eval(_assignment(ROOT / "util" / "noise.py", "allowed_noise_types")))
    assert {item["id"] for item in _manifest()["noise_types"]} == implemented


def test_manifest_modified_gnn_variants_match_implementation():
    implemented = set(ast.literal_eval(
        _assignment(ROOT / "model" / "gnns.py", "SUPPORTED_GCN_MODIFIED_INNER_GNNS")
    ))
    modified = next(item for item in _manifest()["backbones"] if item["id"] == "gcn_modified")
    assert {item["id"] for item in modified["variants"]} == implemented


def test_manifest_has_schema_version():
    manifest = _manifest()
    assert manifest["schema_version"]


def test_method_parameter_defaults_match_release_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for method in _manifest()["methods"]:
        defaults = {
            name: specification["default"]
            for name, specification in method["parameters"].items()
        }
        config_key = f"{method['id']}_params"
        if defaults:
            assert defaults == config[config_key], (
                f"{method['id']} manifest defaults differ from {config_key} in config.yaml"
            )
        else:
            assert config_key not in config


def test_method_parameter_schema_is_well_formed():
    allowed_types = {
        "integer", "number", "boolean", "enum", "string",
        "nullable_integer", "nullable_number",
    }
    for method in _manifest()["methods"]:
        assert isinstance(method.get("parameters"), dict)
        for name, specification in method["parameters"].items():
            assert name
            assert specification["type"] in allowed_types
            assert "default" in specification
            assert specification.get("description")
            if specification["type"] == "enum":
                assert specification["default"] in specification["options"]
            if specification["default"] is not None:
                if "min" in specification:
                    assert specification["default"] >= specification["min"]
                if "max" in specification:
                    assert specification["default"] <= specification["max"]


def test_modified_gnn_rejects_unknown_variant():
    from model.gnns import GCN_modified

    with pytest.raises(ValueError, match="Unknown GCN_modified inner_gnn"):
        GCN_modified(8, 16, 3, inner_gnn="not-a-backbone")
