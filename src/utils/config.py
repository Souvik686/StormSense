"""Configuration loading utilities.

Loads the YAML config, resolves relative paths against a project root, and
supports a lightweight override mechanism (`--set key.nested=value`).
Never hardcode paths elsewhere in the codebase -- go through Config.
"""
from __future__ import annotations

import os
import copy
from dataclasses import dataclass
from typing import Any

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_by_path(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    # best-effort type coercion so CLI overrides stay convenient
    if isinstance(value, str):
        low = value.lower()
        if low in ("true", "false"):
            value = low == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
    cur[keys[-1]] = value


class Config:
    """Dict-like config object with attribute access and path resolution."""

    def __init__(self, data: dict, root: str = PROJECT_ROOT):
        self._data = data
        self._root = root

    @classmethod
    def load(cls, path: str = None, overrides: dict | None = None,
             cli_overrides: list[str] | None = None) -> "Config":
        path = path or os.path.join(PROJECT_ROOT, "configs", "default.yaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if overrides:
            data = _deep_merge(data, overrides)
        if cli_overrides:
            for item in cli_overrides:
                key, _, val = item.partition("=")
                _set_by_path(data, key.strip(), val.strip())
        return cls(data)

    def __getitem__(self, key):
        return self._data[key]

    def get(self, *keys, default=None):
        cur = self._data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def path(self, *keys: str) -> str:
        """Resolve a config value that names a path.

        Paths under top-level sections other than "paths" (era5.*, dem.*,
        insat.*, boundaries.*) are relative to paths.data_root; paths.* keys
        themselves and any already-absolute path are relative to the
        project root.
        """
        rel = self.get(*keys)
        if rel is None:
            raise KeyError(f"Config path not found: {keys}")
        if os.path.isabs(rel):
            return rel
        if keys[0] == "paths":
            return os.path.join(self._root, rel)
        data_root = self.get("paths", "data_root")
        base = os.path.join(self._root, data_root) if not os.path.isabs(data_root) else data_root
        return os.path.join(base, rel)

    def raw(self) -> dict:
        return self._data

    def __repr__(self):
        return f"Config({self._data!r})"


def load_config(path: str = None, cli_overrides: list[str] | None = None) -> Config:
    return Config.load(path, cli_overrides=cli_overrides)
