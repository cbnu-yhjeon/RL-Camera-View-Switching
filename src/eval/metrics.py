"""평가 지표 — 연구 저장소(eval/metrics.py, scripts/06_evaluate.py)와 같은 정의.

정책 출력은 프레임마다 행동 aₜ(보조 뷰 1장 또는 없음)과, 있으면 채널 순위 점수(골드셋 Top-k용).

- 위험 객체 가시화율(포착률): '실제 위험' 채널이 화면(진행 방향 ∪ 보조 뷰)에 들어온 비율.
  분모는 위험 채널 수 — 연구 저장소 `metrics()`와 같다. 기본은 TTC<5s·DCA<3m(haz5),
  README 초안 정의 TTC<3s(haz3)도 함께 보고한다.
- 위험 슬롯 정밀도: 보조 뷰를 켠 프레임 중 그 채널에 실제 위험이 있던 비율. 가시화율과 반드시 병기.
- Flicker rate: 표시 상태가 바뀐 뒤 1.5초(3 keyframe) 안에 다시 바뀐 전환의 비율.
- 전환 빈도(회/분), 평균 표시 뷰 수.
- 골드셋 Top-1/Top-2/F1/정숙률: 연구 저장소 `gold_scores`와 같은 정의, 역확률 가중이 주 수치.
- 누적 보상: 에피소드(scene) 평균 — 보상 설정에 따라 달라지므로 같은 설정끼리만 비교한다.
"""

from __future__ import annotations

import numpy as np

from src.data.log import FRAME_HZ, NUM_CAMERAS, ViewLog
from src.env.view_switch_env import NONE_VIEW, RewardConfig, displayed, gain_matrix, reward_terms


def flicker_rate(seq_by_scene: list[np.ndarray], window: int = 3) -> float:
    transitions = flickers = 0
    for seq in seq_by_scene:
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                transitions += 1
                for k in range(i + 1, min(i + 1 + window, len(seq))):
                    if seq[k] != seq[i]:
                        flickers += 1
                        break
    return flickers / transitions if transitions else 0.0


def gold_scores(sel: np.ndarray, rank: np.ndarray | None, gold: list, primary: np.ndarray,
                weight: np.ndarray | None = None) -> dict:
    """연구 저장소 eval/metrics.gold_scores와 같은 정의 (진행 방향은 정답·예측에서 뺀다)."""
    sel = np.asarray(sel, dtype=bool).copy()
    n = len(sel)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
    prim = np.asarray(primary, dtype=int)
    G = np.zeros_like(sel)
    for i, g in enumerate(gold):
        G[i, [c for c in g if c != prim[i]]] = True
    sel[np.arange(n), prim] = False
    nonempty = G.any(1)
    out = dict(n=int(n), n_nonempty=int(nonempty.sum()),
               mean_aux=float(np.average(sel.sum(1), weights=w)) if n else 0.0)
    for k in (1, 2):
        if rank is None or not nonempty.any():
            out[f"top{k}"] = None
            continue
        r = np.asarray(rank, dtype=float).copy()
        r[np.arange(n), prim] = -np.inf
        top = np.argsort(-r, axis=1)[:, :k]
        hit = np.array([G[i, top[i]].any() for i in range(n)])
        out[f"top{k}"] = float(np.average(hit[nonempty], weights=w[nonempty]))
    f1s = []
    for c in range(sel.shape[1]):
        m = prim != c
        tp = (w * (sel[:, c] & G[:, c] & m)).sum()
        fp = (w * (sel[:, c] & ~G[:, c] & m)).sum()
        fn = (w * (~sel[:, c] & G[:, c] & m)).sum()
        if tp + fp + fn > 0:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    out["f1"] = float(np.mean(f1s)) if f1s else 0.0
    empty = ~nonempty
    out["quiet"] = float(np.average(~sel[empty].any(1), weights=w[empty])) if empty.any() else None
    return out


def episode_returns(log: ViewLog, actions: np.ndarray, reward: RewardConfig,
                    episodes: list[tuple[int, int]]) -> dict:
    disp = displayed(actions, log.primary)
    gain = gain_matrix(log, reward)
    rets, parts = [], {"gain": 0.0, "switch": 0.0, "miss": 0.0}
    for s, e in episodes:
        prev = np.concatenate([[NONE_VIEW], disp[s:e - 1]])
        g, sw, m = reward_terms(gain[s:e], log.haz[reward.hazard][s:e],
                                log.primary[s:e], disp[s:e], prev, reward)
        rets.append(float((g - sw - m).sum()))
        parts["gain"] += g.sum(); parts["switch"] += sw.sum(); parts["miss"] += m.sum()
    n = max(len(episodes), 1)
    return dict(ret=float(np.mean(rets)), **{f"ret_{k}": float(v) / n for k, v in parts.items()})


def evaluate_actions(log: ViewLog, actions: np.ndarray, scores: np.ndarray | None,
                     reward: RewardConfig, episodes: list[tuple[int, int]],
                     gold: dict | None = None) -> dict:
    actions = np.asarray(actions, dtype=np.int64)
    n = log.n_frames
    disp = displayed(actions, log.primary)
    on = disp >= 0
    sel = np.zeros((n, NUM_CAMERAS), dtype=bool)
    sel[np.flatnonzero(on), disp[on]] = True
    screen = sel.copy()
    screen[np.arange(n), log.primary] = True

    out: dict = dict(n=int(n))
    for name, haz in log.haz.items():
        out[f"visibility_{name}"] = round(100 * float((haz & screen).sum()) / max(int(haz.sum()), 1), 2)
    haz = log.haz["haz5"]
    out["precision"] = round(100 * float(haz[sel].sum()) / max(int(on.sum()), 1), 2)
    out["mean_aux"] = round(float(on.mean()), 4)
    seqs = [disp[log.scene_slice(k)] for k in range(log.n_scenes)]
    out["flicker"] = round(flicker_rate(seqs, 3), 4)
    n_sw = sum(int((np.diff(s) != 0).sum()) for s in seqs)
    minutes = n / FRAME_HZ / 60.0
    out["switch_per_min"] = round(n_sw / minutes, 3)
    out["label_top1"] = round(float((actions == log.label_att).mean()), 4)
    out.update({k: round(v, 4) for k, v in episode_returns(log, actions, reward, episodes).items()})
    if gold:
        row = {t: i for i, t in enumerate(log.tokens.tolist())}
        idx = [row[t] for t in gold if t in row]
        if idx:
            toks = log.tokens[idx]
            G = [gold[t]["set"] for t in toks]
            w = np.array([gold[t]["weight"] for t in toks])
            rank = None if scores is None else np.asarray(scores)[idx]
            gw = gold_scores(sel[idx], rank, G, log.primary[idx], w)
            gu = gold_scores(sel[idx], rank, G, log.primary[idx])
            out["gold_n"] = len(idx)
            for k in ("top1", "top2", "f1", "quiet"):
                out[f"gold_{k}"] = None if gw[k] is None else round(gw[k], 4)
                out[f"gold_{k}_unw"] = None if gu[k] is None else round(gu[k], 4)
    return out
