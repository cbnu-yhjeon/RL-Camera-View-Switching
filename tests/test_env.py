"""환경 전이·보상 단위 테스트 (합성 로그 — nuScenes 없이 돈다)."""

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.policies import hold, oracle  # noqa: E402
from src.data.log import synthetic_log  # noqa: E402
from src.env.view_switch_env import (  # noqa: E402
    NONE_VIEW, BatchedViewSwitchEnv, EnvConfig, RewardConfig, ViewSwitchEnv, displayed,
    dp_oracle, episodes_of, step_reward,
)
from src.eval.metrics import episode_returns  # noqa: E402
from src.eval.rollout import rollout  # noqa: E402

RW = RewardConfig(w_risk=1.0, lambda_sw=0.2, lambda_miss=0.5)


def test_reward_terms():
    risk = np.array([0.0, 0.8, 0.0, 0.0, 0.3, 0.0])
    haz = np.array([False, True, False, False, True, False])
    # 채널 1을 새로 켬: 위험 0.8, 전환 0.2, 채널 4 위험이 화면 밖 → 놓침 0.5
    assert step_reward(risk, haz, 0, 1, NONE_VIEW, RW)[0] == pytest.approx(0.8 - 0.2 - 0.5)
    # 같은 채널 유지: 전환 비용 없음
    assert step_reward(risk, haz, 0, 1, 1, RW)[0] == pytest.approx(0.8 - 0.5)
    # 보조 뷰 없음 유지: 위험 이득 0, 두 위험 채널 모두 밖 → 놓침 한 번(프레임 단위)
    assert step_reward(risk, haz, 0, NONE_VIEW, NONE_VIEW, RW)[0] == pytest.approx(-0.5)
    # 진행 방향에 있는 위험은 이미 보인다
    haz2 = np.array([True, False, False, False, False, False])
    assert step_reward(risk, haz2, 0, NONE_VIEW, NONE_VIEW, RW)[0] == pytest.approx(0.0)


def test_primary_action_means_no_aux():
    assert displayed(np.array([0, 3, 2]), np.array([0, 3, 0])).tolist() == [NONE_VIEW, NONE_VIEW, 2]


def test_gym_api_and_action_independent_world():
    log = synthetic_log()
    env = ViewSwitchEnv(log, "gt", RW)
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)
    # 같은 에피소드를 서로 다른 행동열로 돌려도 외생 관측(xₜ)은 같다
    xs = []
    for policy in (lambda t: 0, lambda t: (t % 5) + 1):
        o, _ = env.reset(options={"episode": 1})
        seq = [o["risk"].copy()]
        for t in range(11):
            o, *_ = env.step(policy(t))
            seq.append(o["risk"].copy())
        xs.append(np.stack(seq))
    np.testing.assert_array_equal(xs[0], xs[1])


def test_history_padding_at_scene_start():
    log = synthetic_log(scene_len=6)
    env = ViewSwitchEnv(log, "gt", RW, EnvConfig(history=4))
    o, _ = env.reset(options={"episode": 2})
    assert o["pad"].tolist() == [0, 0, 0, 1]
    assert not o["risk"][:3].any()
    np.testing.assert_allclose(o["risk"][3], log.risk_gt[12])
    o, *_ = env.step(0)
    assert o["pad"].tolist() == [0, 0, 1, 1]


def test_episode_lengths_and_prev_view_obs():
    log = synthetic_log(scene_len=5)
    env = ViewSwitchEnv(log, "gt", RW)
    o, _ = env.reset(options={"episode": 0})
    assert o["prev"][-1] == 1.0                 # 시작은 보조 뷰 없음
    n, done = 0, False
    while not done:
        o, r, done, trunc, info = env.step(2)
        n += 1
    assert n == 5
    assert o["prev"][2] == 1.0 and info["displayed"] == 2


def test_batched_matches_single_env():
    log = synthetic_log(n_scenes=3, scene_len=7)
    rng = np.random.default_rng(1)
    acts = rng.integers(6, size=7)
    single = ViewSwitchEnv(log, "gt", RW)
    single.reset(options={"episode": 1})
    rs = [single.step(a)[1] for a in acts]
    b = BatchedViewSwitchEnv(log, 1, "gt", RW, seed=0)
    b.frame[:] = 7; b.end[:] = 14
    rb = [float(b.step(np.array([a]))[0][0]) for a in acts]
    np.testing.assert_allclose(rs, rb)
    assert b.finished_returns and b.finished_returns[0] == pytest.approx(sum(rs))


def test_rollout_matches_episode_returns():
    log = synthetic_log()
    pol = lambda o: (o["risk"][:, -1].argmax(1), None)
    a, _, eps = rollout(pol, log, "gt", RW, EnvConfig())
    single = ViewSwitchEnv(log, "gt", RW)
    total = []
    for k in range(len(eps)):
        o, _ = single.reset(options={"episode": k})
        ret, done = 0.0, False
        while not done:
            o, r, done, *_ = single.step(int(o["risk"][-1].argmax()))
            ret += r
        total.append(ret)
    assert episode_returns(log, a, RW, eps)["ret"] == pytest.approx(np.mean(total))


def test_dp_oracle_is_optimal_bruteforce():
    log = synthetic_log(n_scenes=1, scene_len=5, seed=3)
    acts, best = dp_oracle(log, 0, 5, RW)
    vals = []
    for seq in itertools.product(range(6), repeat=5):
        vals.append(episode_returns(log, np.array(seq), RW, [(0, 5)])["ret"])
    assert best == pytest.approx(max(vals))
    assert episode_returns(log, acts, RW, [(0, 5)])["ret"] == pytest.approx(best)


def test_switch_cost_zero_makes_oracle_myopic():
    log = synthetic_log(seed=5)
    rw = RewardConfig(lambda_sw=0.0, lambda_miss=0.5)
    a, _ = oracle(log, rw)
    eps = episodes_of(log)
    # 전환 비용이 없으면 프레임별 최선을 독립적으로 고른 것과 누적 보상이 같다
    best_each = []
    for i in range(log.n_frames):
        best_each.append(max(step_reward(log.risk_gt[i], log.haz["haz5"][i], 0,
                                         displayed(c, 0), 0, rw)[0] for c in range(6)))
    assert episode_returns(log, a, rw, eps)["ret"] == pytest.approx(
        np.sum(best_each) / len(eps))


def test_hold_enforces_min_dwell():
    log = synthetic_log(n_scenes=1, scene_len=20)
    flip = np.array([1, 2] * 10)
    out = displayed(hold(log, flip, 3), log.primary)
    changes = np.flatnonzero(np.diff(out) != 0) + 1
    assert np.all(np.diff(np.concatenate([[0], changes])) >= 3)


def test_concat_scenes():
    log = synthetic_log(n_scenes=5, scene_len=4)
    assert episodes_of(log, 2) == [(0, 8), (8, 16), (16, 20)]


def test_maneuver_term_adds_fit_to_gain():
    log = synthetic_log(n_scenes=1, scene_len=3)
    log.maneuver[:] = 1                                   # TURN_LEFT → FRONT_LEFT(5) 적합도 1.0
    from src.env.view_switch_env import gain_matrix
    g0 = gain_matrix(log, RewardConfig())
    g1 = gain_matrix(log, RewardConfig(w_maneuver=0.4))
    np.testing.assert_allclose(g1[:, 5] - g0[:, 5], 0.4, rtol=1e-6)
    np.testing.assert_allclose(g1[:, 2] - g0[:, 2], 0.0)
