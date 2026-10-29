"""비교 정책 (README 실험 설계 표).

B1      고정 전방 — 보조 뷰 없음 (진행 방향만 상시 표시). 하한선.
RISK    관측 위험도가 가장 큰 채널 (진행 방향 제외). 위험이 전혀 없으면 보조 뷰 없음.
        '매 순간 가장 중요한 뷰'를 고르는 근시안적 정책 — 전환 비용을 모른다.
B2      연구 저장소 규칙(오라클 기동) — 위험 EMA + 기동 적합도의 argmax. 기동을 미리 안다(상한 성격).
B2c     같은 규칙, 기동은 과거 궤적만으로 추정 — RL·SUP와 같은 정보 조건.
SUP     연구 저장소 지도학습 모델 — 채널별 강조 확률의 argmax(진행 방향 제외).
+hold   위 정책에 최소 유지 시간 k를 거는 히스테리시스 — 전환 억제를 손으로 넣은 비-RL 경쟁자.
ORACLE  GT·미래를 아는 동적 계획법 최적해 (보상 상한, env.dp_oracle).

모두 (행동 (N,), 순위 점수 (N,6) | None)을 낸다. B2·B2c·SUP 출력은 연구 저장소 코드가 같은
손상 입력으로 계산해 내보낸 값이다(scripts/00_export_research_cache.py).
"""

from __future__ import annotations

import numpy as np

from src.data.log import NUM_CAMERAS, ViewLog
from src.env.view_switch_env import NONE_VIEW, RewardConfig, displayed, dp_oracle, episodes_of


def _mask_primary(x: np.ndarray, primary: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    x[np.arange(len(x)), primary] = -np.inf
    return x


def fixed_front(log: ViewLog, cond: str):
    return log.primary.copy(), None


def risk_greedy(log: ViewLog, cond: str, eps: float = 1e-6):
    r = _mask_primary(log.risk(cond), log.primary)
    a = r.argmax(1)
    a = np.where(r.max(1) > eps, a, log.primary)
    return a, np.where(np.isfinite(r), r, -1.0)


def research_rule(log: ViewLog, cond: str, which: str = "b2c"):
    c = log.cond[cond]
    if f"{which}_att" not in c:
        raise KeyError(f"{log.split}/{cond}: {which} 출력 없음")
    return c[f"{which}_att"].astype(np.int64), c[f"{which}_score"]


def supervised(log: ViewLog, cond: str):
    p = log.cond[cond]["sup_prob"]
    return _mask_primary(p, log.primary).argmax(1), p


def hold(log: ViewLog, actions: np.ndarray, k: int) -> np.ndarray:
    """최소 유지 시간 k 스텝 — 표시 상태를 바꾼 뒤 k 스텝 동안은 다시 바꾸지 않는다."""
    disp = displayed(actions, log.primary)
    out = np.empty_like(disp)
    for s in range(log.n_scenes):
        sl = log.scene_slice(s)
        cur, held = NONE_VIEW, k
        for i in range(sl.start, sl.stop):
            want = disp[i]
            if want != cur and held >= k:
                cur, held = want, 1            # 바꾼 프레임도 표시 1스텝으로 센다
            else:
                held += 1
            if cur == log.primary[i]:          # 진행 방향이 바뀌어 같은 채널이 되면 '없음'
                cur = NONE_VIEW
            out[i] = cur
    return np.where(out == NONE_VIEW, log.primary, out)


def oracle(log: ViewLog, reward: RewardConfig, concat: int = 1):
    a = np.zeros(log.n_frames, dtype=np.int64)
    for s, e in episodes_of(log, concat):
        a[s:e], _ = dp_oracle(log, s, e, reward)
    return a, None


BASELINES = {
    "B1": lambda log, cond, rw: fixed_front(log, cond),
    "RISK": lambda log, cond, rw: risk_greedy(log, cond),
    "B2": lambda log, cond, rw: research_rule(log, cond, "b2"),
    "B2c": lambda log, cond, rw: research_rule(log, cond, "b2c"),
    "SUP": lambda log, cond, rw: supervised(log, cond),
    "ORACLE": lambda log, cond, rw: oracle(log, rw),
}
