"""결정적 평가 롤아웃 — 모든 에피소드를 한꺼번에 시간 정렬해 돌린다.

행동이 세계에 영향을 주지 않으므로, t번째 스텝에서 각 에피소드의 프레임은 start+t로 정해져 있다.
에피소드별로 따로 돌릴 필요 없이 (에피소드 수) 배치로 한 스텝씩 진행한다.

policy(obs: dict[np.ndarray]) → (actions (B,), scores (B,6) | None)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.data.log import NUM_CAMERAS, ViewLog
from src.env.view_switch_env import NONE_VIEW, EnvConfig, RewardConfig, _Core, displayed, episodes_of
from src.eval.metrics import evaluate_actions

Policy = Callable[[dict], tuple[np.ndarray, np.ndarray | None]]


def rollout(policy: Policy, log: ViewLog, cond: str, reward: RewardConfig,
            env_cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray | None, list]:
    core = _Core(log, [cond], reward, env_cfg)
    eps = episodes_of(log, env_cfg.concat_scenes)
    starts = np.array([s for s, _ in eps])
    lens = np.array([e - s for s, e in eps])
    actions = np.zeros(log.n_frames, dtype=np.int64)
    scores = None
    prev = np.full(len(eps), NONE_VIEW, dtype=np.int64)
    since = np.full(len(eps), env_cfg.since_cap, dtype=np.int64)
    for t in range(int(lens.max())):
        act = np.flatnonzero(lens > t)
        f = starts[act] + t
        obs = core.gather(f, prev[act], since[act], np.zeros(len(act), dtype=np.int64))
        a, sc = policy(obs)
        a = np.asarray(a, dtype=np.int64)
        actions[f] = a
        if sc is not None:
            if scores is None:
                scores = np.zeros((log.n_frames, NUM_CAMERAS), dtype=np.float32)
            scores[f] = sc
        d = displayed(a, log.primary[f])
        since[act] = np.where(d != prev[act], 0, since[act] + 1)
        prev[act] = d
    return actions, scores, eps


def evaluate_policy(policy: Policy, log: ViewLog, cond: str, reward: RewardConfig,
                    env_cfg: EnvConfig, gold: dict | None = None) -> dict:
    actions, scores, eps = rollout(policy, log, cond, reward, env_cfg)
    return evaluate_actions(log, actions, scores, reward, eps, gold)
