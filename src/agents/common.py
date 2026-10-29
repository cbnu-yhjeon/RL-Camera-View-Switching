"""에이전트 공통 — 시드, 평가용 정책 래퍼, 학습 중 검증·최고 모델 보관."""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch

from src.encoders.state_encoder import to_torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_policy(score_fn, device, chunk: int = 512):
    """score_fn(obs_torch) → (B,6) 점수. argmax가 행동, 점수가 골드셋 순위."""

    @torch.no_grad()
    def policy(obs: dict):
        n = len(obs["pad"])
        outs = []
        for i in range(0, n, chunk):
            o = to_torch({k: v[i:i + chunk] for k, v in obs.items()}, device)
            outs.append(score_fn(o).float().cpu().numpy())
        s = np.concatenate(outs)
        return s.argmax(1), s

    return policy


class Validator:
    """주기적 검증 — 검증 분할 누적 보상이 가장 높은 가중치를 보관한다(조기 종료 대신)."""

    def __init__(self, evaluate_fn, every: int):
        self.evaluate_fn = evaluate_fn
        self.every = every
        self.next_at = every
        self.best = -np.inf
        self.best_state = None
        self.history: list[dict] = []
        self.t0 = time.time()

    def maybe(self, step: int, module: torch.nn.Module, extra: dict | None = None, force=False):
        if step < self.next_at and not force:
            return None
        self.next_at = step + self.every
        was = module.training
        module.eval()
        res = self.evaluate_fn()
        module.train(was)
        row = dict(step=int(step), time=round(time.time() - self.t0, 1), **(extra or {}), **res)
        self.history.append(row)
        if res["ret"] > self.best:
            self.best = res["ret"]
            self.best_state = copy.deepcopy(module.state_dict())
        print(f"  [{step:>8d}] val ret {res['ret']:7.3f}  vis {res['visibility_haz5']:5.1f}%  "
              f"flicker {res['flicker']:.3f}  sw/min {res['switch_per_min']:5.2f}  "
              f"aux {res['mean_aux']:.2f}  ({row['time']:.0f}s)", flush=True)
        return res
