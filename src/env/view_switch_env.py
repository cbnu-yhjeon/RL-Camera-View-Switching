"""ViewSwitchEnv — 로그로 정확하게 재생되는 뷰 전환 MDP (README '핵심 아이디어').

    상태 sₜ = (xₜ, vₜ₋₁, 마지막 전환 후 경과)   xₜ: 과거 T 프레임의 영상 특징·위험도·CAN
    행동 aₜ ∈ {0..5}                              보조(attention) 뷰. aₜ = 진행 방향 채널이면 '보조 뷰 없음'
    전이   xₜ₊₁ ← 로그 (행동과 무관),  vₜ = aₜ
    보상   rₜ = w_risk·risk_gt[aₜ] − λ_sw·1[vₜ ≠ vₜ₋₁] − λ_miss·1[위험 채널이 화면 밖]

'화면' = 진행 방향(primary, 상시 표시) ∪ 보조 뷰. 진행 방향 채널은 연구 저장소처럼 고정이고
선택 대상이 아니므로, 그 채널을 고르는 행동을 '보조 뷰를 끈다'로 해석한다. 그래서 표시 상태
vₜ는 {없음, 채널 0..5} 7가지이고 전환 비용은 표시 상태가 바뀔 때만 든다.

세계의 전이가 행동과 무관하므로 시뮬레이터 오차가 없고, 같은 이유로 GT를 알면 최적 행동열을
동적 계획법으로 정확히 구할 수 있다(dp_oracle — 보상의 상한).

두 가지 인터페이스를 둔다.
  ViewSwitchEnv         Gymnasium 표준 단일 환경 (scene 하나 = 에피소드 하나)
  BatchedViewSwitchEnv  학습용 n개 병렬 환경. 같은 _Core로 관측·보상을 계산하므로 두 환경의
                        전이·보상은 동일하다(tests/test_env.py가 확인).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.data.log import MANEUVER_FIT, NUM_CAMERAS, ViewLog

NONE_VIEW = -1                 # 보조 뷰 없음
N_DISPLAY = NUM_CAMERAS + 1    # 표시 상태 수 (없음 + 6채널), 원-핫에서 '없음'은 마지막 칸


@dataclass
class RewardConfig:
    w_risk: float = 1.0
    lambda_sw: float = 0.1
    lambda_miss: float = 0.5
    hazard: str = "haz5"       # haz5: TTC<5s·DCA<3m (연구 저장소 정의) | haz3: TTC<3s·DCA<3m
    # 기동 적합도 항 w_m·m(c) — 연구 저장소 라벨 점수 s(c) = risk(c) + λ·m(c)의 기동 항(λ=0.4).
    # 0이면 README 보상 그대로. 라벨 기동(미래 궤적)은 보상에만 쓰고 관측에는 넣지 않는다.
    w_maneuver: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "RewardConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EnvConfig:
    history: int = 4           # 과거 프레임 수 (2초) — 연구 저장소 모델과 같은 입력 창
    since_cap: int = 10        # 경과 시간 특징 상한(스텝)
    concat_scenes: int = 1     # 연속 scene을 이어 붙인 긴 에피소드 (README 리스크 대응)

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def gain_matrix(log: ViewLog, cfg: RewardConfig) -> np.ndarray:
    """보조 뷰로 고른 채널의 이득 행렬 (N,6) — reward_terms의 risk_gt 자리에 들어간다."""
    if cfg.w_maneuver == 0:
        return log.risk_gt
    return (log.risk_gt + cfg.w_maneuver / cfg.w_risk * MANEUVER_FIT[log.maneuver]).astype(np.float32)


def displayed(action, primary):
    """행동 → 표시 상태 (진행 방향 채널을 고르면 '없음')."""
    action = np.asarray(action)
    return np.where(action == np.asarray(primary), NONE_VIEW, action)


def reward_terms(risk_gt: np.ndarray, haz: np.ndarray, primary, disp, prev_disp,
                 cfg: RewardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(위험 가시 보상, 전환 비용, 놓침 벌점) — 배치(B,6) 또는 단일(6,) 모두 받는다."""
    risk_gt = np.atleast_2d(risk_gt)
    haz = np.atleast_2d(haz)
    primary = np.atleast_1d(primary)
    disp = np.atleast_1d(disp)
    prev_disp = np.atleast_1d(prev_disp)
    rows = np.arange(len(disp))
    on = disp >= 0
    gain = np.where(on, risk_gt[rows, np.maximum(disp, 0)], 0.0) * cfg.w_risk
    switch = (disp != prev_disp).astype(np.float64) * cfg.lambda_sw
    visible = np.zeros_like(haz, dtype=bool)
    visible[rows, primary] = True
    visible[rows[on], disp[on]] = True
    miss = (haz & ~visible).any(1).astype(np.float64) * cfg.lambda_miss
    return gain, switch, miss


def step_reward(risk_gt, haz, primary, disp, prev_disp, cfg: RewardConfig) -> np.ndarray:
    g, s, m = reward_terms(risk_gt, haz, primary, disp, prev_disp, cfg)
    return g - s - m


def episodes_of(log: ViewLog, concat: int = 1) -> list[tuple[int, int]]:
    """에피소드 = 연속 scene concat개의 프레임 구간 [start, end)."""
    off = log.scene_offsets
    out = []
    for k in range(0, log.n_scenes, concat):
        out.append((int(off[k]), int(off[min(k + concat, log.n_scenes)])))
    return out


class _Core:
    """관측 조립과 보상 — 단일·병렬 환경이 공유한다."""

    def __init__(self, log: ViewLog, conds: list[str], reward: RewardConfig, cfg: EnvConfig):
        self.log = log
        self.conds = list(conds)
        self.reward_cfg = reward
        self.cfg = cfg
        self.risk = np.stack([log.risk(c) for c in self.conds])          # (C,N,6)
        self.blackout = np.stack([log.blackout(c) for c in self.conds])  # (C,N,6)
        self.haz = log.haz[reward.hazard]
        self.gain = gain_matrix(log, reward)
        self.T = cfg.history

    def observation_space(self) -> spaces.Dict:
        T, D = self.T, self.log.feat_dim
        f = np.float32
        return spaces.Dict({
            "img": spaces.Box(-np.inf, np.inf, (T, NUM_CAMERAS, D), f),
            "risk": spaces.Box(0.0, np.inf, (T, NUM_CAMERAS), f),
            "can": spaces.Box(-np.inf, np.inf, (T, self.log.can.shape[1]), f),
            "pad": spaces.Box(0.0, 1.0, (T,), f),
            "prev": spaces.Box(0.0, 1.0, (N_DISPLAY,), f),
            "primary": spaces.Box(0.0, 1.0, (NUM_CAMERAS,), f),
            "since": spaces.Box(0.0, 1.0, (1,), f),
        })

    def gather(self, frame, prev_disp, since, cond) -> dict[str, np.ndarray]:
        """배치 관측. frame·prev_disp·since·cond 모두 (B,) 정수 배열.

        과거 창은 scene 시작에서 잘린다(패딩 + pad=0) — 모델은 과거만 본다.
        """
        frame = np.asarray(frame, dtype=np.int64)
        cond = np.asarray(cond, dtype=np.int64)
        B, T = len(frame), self.T
        ks = frame[:, None] - np.arange(T - 1, -1, -1)[None, :]          # (B,T) 오래된 → 현재
        valid = ks >= self.log.scene_start[frame][:, None]
        ks = np.where(valid, ks, frame[:, None])
        c = cond[:, None]
        img = self.log.img[ks].astype(np.float32)                          # (B,T,6,D)
        img[self.blackout[c, ks]] = 0.0                                    # 영상 결손 채널
        risk = self.risk[c, ks]
        can = self.log.can[ks].copy()
        img[~valid] = 0.0
        risk = np.where(valid[..., None], risk, 0.0).astype(np.float32)
        can[~valid] = 0.0
        prev = np.zeros((B, N_DISPLAY), dtype=np.float32)
        prev[np.arange(B), np.asarray(prev_disp)] = 1.0                    # -1 → 마지막 칸
        prim = np.zeros((B, NUM_CAMERAS), dtype=np.float32)
        prim[np.arange(B), self.log.primary[frame]] = 1.0
        s = np.minimum(np.asarray(since), self.cfg.since_cap) / self.cfg.since_cap
        return {"img": img, "risk": risk, "can": can, "pad": valid.astype(np.float32),
                "prev": prev, "primary": prim, "since": s.astype(np.float32)[:, None]}

    def transition(self, frame, action, prev_disp, since):
        """(보상, 새 표시 상태, 새 경과 시간, 보상 항 dict)."""
        frame = np.asarray(frame)
        prim = self.log.primary[frame]
        disp = displayed(action, prim)
        g, s, m = reward_terms(self.gain[frame], self.haz[frame], prim, disp,
                               prev_disp, self.reward_cfg)
        changed = disp != np.asarray(prev_disp)
        new_since = np.where(changed, 0, np.asarray(since) + 1)
        return g - s - m, disp, new_since, dict(gain=g, switch=s, miss=m)


class ViewSwitchEnv(gym.Env):
    """Gymnasium 단일 환경. reset(options={"episode": k})로 특정 에피소드를 고를 수 있다."""

    metadata = {"render_modes": []}

    def __init__(self, log: ViewLog, conds: list[str] | str = "gt",
                 reward: RewardConfig | None = None, cfg: EnvConfig | None = None):
        super().__init__()
        self.core = _Core(log, [conds] if isinstance(conds, str) else conds,
                          reward or RewardConfig(), cfg or EnvConfig())
        self.episodes = episodes_of(log, self.core.cfg.concat_scenes)
        self.observation_space = self.core.observation_space()
        self.action_space = spaces.Discrete(NUM_CAMERAS)
        self._frame = self._end = self._cond = 0
        self._prev = NONE_VIEW
        self._since = self.core.cfg.since_cap

    def _obs(self):
        o = self.core.gather([self._frame], [self._prev], [self._since], [self._cond])
        return {k: v[0] for k, v in o.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        ep = options.get("episode")
        if ep is None:
            ep = int(self.np_random.integers(len(self.episodes)))
        cond = options.get("cond")
        self._cond = (int(self.np_random.integers(len(self.core.conds))) if cond is None
                      else self.core.conds.index(cond))
        self._frame, self._end = self.episodes[ep]
        self._prev = NONE_VIEW
        self._since = self.core.cfg.since_cap
        return self._obs(), {"episode": ep, "frame": self._frame, "cond": self.core.conds[self._cond]}

    def step(self, action):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"잘못된 행동: {action}")
        r, disp, since, terms = self.core.transition(
            np.array([self._frame]), np.array([int(action)]), np.array([self._prev]),
            np.array([self._since]))
        info = {"frame": self._frame, "displayed": int(disp[0]),
                **{k: float(v[0]) for k, v in terms.items()}}
        self._prev, self._since = int(disp[0]), int(since[0])
        self._frame += 1
        terminated = self._frame >= self._end
        if terminated:
            self._frame -= 1                     # 마지막 관측은 의미 없음 — 범위 안에 둔다
        return self._obs(), float(r[0]), terminated, False, info


class BatchedViewSwitchEnv:
    """n개 병렬 에피소드. 끝난 환경은 무작위 에피소드·조건으로 즉시 다시 시작한다.

    관측 대신 상태 인덱스(frame, prev, since, cond)도 돌려준다 — DQN 리플레이 버퍼는 이것만
    저장하고 관측은 다시 조립한다(로그 재생 환경이라 가능하다).
    """

    def __init__(self, log: ViewLog, n_envs: int, conds: list[str] | str = "gt",
                 reward: RewardConfig | None = None, cfg: EnvConfig | None = None, seed: int = 0):
        self.core = _Core(log, [conds] if isinstance(conds, str) else conds,
                          reward or RewardConfig(), cfg or EnvConfig())
        self.episodes = np.array(episodes_of(log, self.core.cfg.concat_scenes), dtype=np.int64)
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.frame = np.zeros(n_envs, dtype=np.int64)
        self.end = np.zeros(n_envs, dtype=np.int64)
        self.cond = np.zeros(n_envs, dtype=np.int64)
        self.prev = np.full(n_envs, NONE_VIEW, dtype=np.int64)
        self.since = np.full(n_envs, self.core.cfg.since_cap, dtype=np.int64)
        self.ep_return = np.zeros(n_envs)
        self.finished_returns: list[float] = []
        self._reset_idx(np.arange(n_envs))

    def _reset_idx(self, idx):
        if len(idx) == 0:
            return
        ep = self.rng.integers(len(self.episodes), size=len(idx))
        self.frame[idx], self.end[idx] = self.episodes[ep, 0], self.episodes[ep, 1]
        self.cond[idx] = self.rng.integers(len(self.core.conds), size=len(idx))
        self.prev[idx] = NONE_VIEW
        self.since[idx] = self.core.cfg.since_cap
        self.ep_return[idx] = 0.0

    def state(self):
        return self.frame.copy(), self.prev.copy(), self.since.copy(), self.cond.copy()

    def obs(self):
        return self.core.gather(self.frame, self.prev, self.since, self.cond)

    def step(self, actions):
        r, disp, since, _ = self.core.transition(self.frame, np.asarray(actions), self.prev, self.since)
        self.prev, self.since = disp.astype(np.int64), since.astype(np.int64)
        self.ep_return += r
        self.frame += 1
        done = self.frame >= self.end
        idx = np.flatnonzero(done)
        self.finished_returns.extend(self.ep_return[idx].tolist())
        self._reset_idx(idx)
        return r, done


def dp_oracle(log: ViewLog, start: int, end: int, cfg: RewardConfig) -> tuple[np.ndarray, float]:
    """GT·미래를 아는 최적 행동열 (Viterbi). 전이가 행동과 무관해서 정확히 풀린다.

    표시 상태 d ∈ {없음, 0..5}. 진행 방향 채널은 표시 상태가 될 수 없다('없음'과 같음).
    반환: (행동열, 최적 누적 보상).
    """
    n = end - start
    D = N_DISPLAY                                          # 인덱스 6 = 없음
    disp_of = np.array(list(range(NUM_CAMERAS)) + [NONE_VIEW])
    haz = log.haz[cfg.hazard]
    gain = gain_matrix(log, cfg)
    # 프레임별 즉시 보상(전환 비용 제외)
    rr = np.full((n, D), -np.inf)
    for j in range(D):
        d = disp_of[j]
        f = np.arange(start, end)
        ok = d != log.primary[f]
        if d == NONE_VIEW:
            ok[:] = True
        g, _, m = reward_terms(gain[f], haz[f], log.primary[f],
                               np.full(n, d), np.full(n, d), cfg)
        rr[:, j] = np.where(ok, g - m, -np.inf)
    sw = cfg.lambda_sw * (1.0 - np.eye(D))                 # sw[p, d]
    best = rr[0] - sw[D - 1]                               # 시작 표시 상태 = 없음
    back = np.zeros((n, D), dtype=np.int64)
    for i in range(1, n):
        cand = best[:, None] - sw                          # (p, d)
        back[i] = cand.argmax(0)
        best = cand.max(0) + rr[i]
    j = int(best.argmax())
    total = float(best[j])
    path = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    disp = disp_of[path]
    actions = np.where(disp == NONE_VIEW, log.primary[start:end], disp)
    return actions, total
