"""YAML configuration with inheritance.

A config file may contain ``base: other.yaml`` (path relative to the file); the base is loaded first and the
current file is merged into it recursively. Command-line overrides use dotted keys, e.g.
``--set experiment.time_limit=5``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    bases = cfg.pop("base", None)
    if bases:
        merged: dict = {}
        for b in [bases] if isinstance(bases, str) else bases:
            merged = deep_update(merged, load_config(path.parent / b))
        cfg = deep_update(merged, cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    cfg.setdefault("_source", str(path))
    return cfg
