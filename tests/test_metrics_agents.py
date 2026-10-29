"""지표와 에이전트 스모크 테스트."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.a2c import A2CConfig, train_a2c, train_ppo  # noqa: E402
from src.agents.dqn import DQNConfig, train_dqn  # noqa: E402
from src.data.log import synthetic_log  # noqa: E402
from src.encoders.state_encoder import EncoderConfig  # noqa: E402
from src.env.view_switch_env import BatchedViewSwitchEnv, RewardConfig  # noqa: E402
from src.eval.metrics import evaluate_actions, flicker_rate, gold_scores  # noqa: E402


def test_flicker_rate():
    # 전환 3번: 0→1(바로 1→0 재전환 = flicker), 1→0, 0→2(유지)
    assert flicker_rate([np.array([0, 1, 0, 0, 0, 0, 2, 2, 2, 2])]) == pytest.approx(1 / 3)
    assert flicker_rate([np.array([3, 3, 3])]) == 0.0


def test_gold_scores():
    sel = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=bool)
    rank = np.array([[0, .9, .1, 0, 0, 0], [0, .1, .9, 0, 0, 0]])
    g = gold_scores(sel, rank, [[1], []], np.array([0, 0]))
    assert g["top1"] == 1.0 and g["quiet"] == 1.0 and g["f1"] == 1.0


def test_visibility_counts_primary_and_aux():
    log = synthetic_log(n_scenes=1, scene_len=3)
    log.haz["haz5"][:] = False
    log.haz["haz5"][0, 2] = True
    log.haz["haz5"][1, 0] = True          # 진행 방향 — 항상 보인다
    log.haz["haz5"][2, 4] = True
    res = evaluate_actions(log, np.array([2, 0, 0]), None, RewardConfig(), [(0, 3)])
    assert res["visibility_haz5"] == pytest.approx(100 * 2 / 3, abs=0.01)
    assert res["mean_aux"] == pytest.approx(1 / 3, abs=1e-3)


@pytest.mark.parametrize("algo", ["dqn", "dqn_aa", "bandit_aa", "a2c", "ppo"])
def test_agents_smoke(algo):
    torch.manual_seed(0)
    log = synthetic_log(n_scenes=3, scene_len=10)
    enc = EncoderConfig(feat_dim=log.feat_dim, d=16, heads=2, ff=32)
    env = BatchedViewSwitchEnv(log, 4, "gt", RewardConfig(), seed=0)
    if algo.startswith(("dqn", "bandit")):
        cfg = DQNConfig(total_steps=200, n_envs=4, learning_starts=40, batch_size=16,
                        all_actions=algo.endswith("_aa"), gamma=0.0 if algo.startswith("bandit") else 0.9)
        net = train_dqn(env, enc, cfg, "cpu")
    else:
        cfg = A2CConfig(total_steps=256, n_envs=4, n_steps=8, minibatch=16, epochs=1)
        net = (train_a2c if algo == "a2c" else train_ppo)(env, enc, cfg, "cpu")
    assert sum(p.numel() for p in net.parameters()) > 0


def test_q_all_displays_matches_forward():
    """전 행동 타깃용 일괄 평가가 표시 상태를 바꾼 관측의 forward와 같아야 한다."""
    from src.agents.dqn import QNet, _onehot_prev
    from src.env.view_switch_env import _Core, EnvConfig
    torch.manual_seed(0)
    log = synthetic_log()
    core = _Core(log, ["gt"], RewardConfig(), EnvConfig())
    q = QNet(EncoderConfig(feat_dim=log.feat_dim, d=16, heads=2, ff=32)).eval()
    f = np.array([3, 15, 30])
    disps = [np.array([-1, 2, 5]), np.array([1, 1, -1])]
    sinces = [np.array([0, 3, 10]), np.array([2, 0, 1])]
    base = {k: torch.as_tensor(v) for k, v in core.gather(f, disps[0], sinces[0], np.zeros(3, int)).items()}
    batch = q.q_all_displays(base, [_onehot_prev(d, "cpu") for d in disps],
                             [torch.as_tensor(s / 10, dtype=torch.float32)[:, None] for s in sinces])
    for k in range(2):
        o = {kk: torch.as_tensor(v) for kk, v in core.gather(f, disps[k], sinces[k], np.zeros(3, int)).items()}
        torch.testing.assert_close(batch[k], q(o))
