"""설정 로딩 — `inherit:`로 다른 설정을 물려받고 바뀌는 값만 적는다 (연구 저장소와 같은 방식)."""

from __future__ import annotations

import copy
import os

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(os.path.join(os.path.dirname(path), parent)), cfg)
    return _merge(cfg, overrides or {})


def set_path(cfg: dict, dotted: str, value) -> None:
    """'reward.lambda_sw=0.2' 같은 명령줄 덮어쓰기."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    v = yaml.safe_load(str(value))
    if isinstance(v, str):                       # YAML 1.1은 '1e-4'를 문자열로 읽는다
        try:
            v = float(v)
        except ValueError:
            pass
    d[keys[-1]] = v
