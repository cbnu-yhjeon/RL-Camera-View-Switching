"""RL 정책 학습 + 시험 평가.

    python scripts/train.py --config configs/dqn.yaml --seed 0 --set reward.lambda_sw=0.2

결과: {results_dir}/{exp_name}[_tag]/seed{seed}/
    config.yaml   실제로 쓴 설정 (덮어쓰기 반영)
    model.pt      검증 누적 보상이 가장 높았던 가중치
    history.json  학습 중 검증 지표
    test_{조건}.json  시험 분할 지표 (조건별)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.a2c import A2CConfig, ActorCritic, train_a2c, train_ppo  # noqa: E402
from src.agents.common import Validator, device_of, greedy_policy, set_seed  # noqa: E402
from src.agents.dqn import DQNConfig, QNet, train_dqn  # noqa: E402
from src.data.log import load_gold, load_log  # noqa: E402
from src.encoders.state_encoder import EncoderConfig  # noqa: E402
from src.env.view_switch_env import BatchedViewSwitchEnv, EnvConfig, RewardConfig  # noqa: E402
from src.eval.rollout import evaluate_policy  # noqa: E402
from src.utils.config import load_config, set_path  # noqa: E402


def build_net(cfg: dict, feat_dim: int):
    enc = EncoderConfig.from_dict(cfg.get("encoder"), feat_dim=feat_dim)
    if cfg["algo"] == "dqn":
        return QNet(enc, cfg.get("dqn", {}).get("dueling", True)), enc
    return ActorCritic(enc), enc


def score_fn(net, algo):
    return net if algo == "dqn" else net.logits


def load_policy(run_dir: str, device=None):
    """저장된 실행 → (정책, 설정)."""
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    torch.set_num_threads(int(cfg.get("threads", 3)))       # 기본값(전 코어)은 작은 연산에서 오히려 느리다
    device = device or device_of(cfg.get("device"))
    net, _ = build_net(cfg, state["feat_dim"])
    net.load_state_dict(state["model"])
    net.to(device).eval()
    return greedy_policy(score_fn(net, cfg["algo"]), device), cfg


def run_dir_of(cfg: dict, seed: int, tag: str | None) -> str:
    name = cfg["exp_name"] + (f"_{tag}" if tag else "")
    return os.path.join(cfg.get("results_dir", "results/runs"), name, f"seed{seed}")


def evaluate_run(run_dir: str, conds: list[str] | None = None, split: str | None = None) -> dict:
    policy, cfg = load_policy(run_dir)
    split = split or cfg["eval"]["split"]
    conds = conds or cfg["eval"]["conds"]
    log = load_log(cfg["cache_dir"], split, conds, cfg.get("img_pca"))
    gold = load_gold(cfg["cache_dir"]) if split == "test" else None
    rw, ec = RewardConfig.from_dict(cfg["reward"]), EnvConfig.from_dict(cfg["env"])
    out = {}
    base_rw = RewardConfig.from_dict({**cfg["reward"], "w_maneuver": 0.0})
    for c in conds:
        res = evaluate_policy(policy, log, c, base_rw, ec, gold)
        if rw.w_maneuver:
            # 보상이 다른 실행(DQN-M)도 표에서는 기본 보상으로 비교한다. 자기 보상 값은 따로 둔다.
            own = evaluate_policy(policy, log, c, rw, ec)
            res.update({f"own_{k}": v for k, v in own.items() if k.startswith("ret")})
        with open(os.path.join(run_dir, f"{split}_{c}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        out[c] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="설정 덮어쓰기 key.sub=value")
    ap.add_argument("--tag", default=None, help="실행 디렉터리 접미사 (예: lsw0.2)")
    ap.add_argument("--steps", type=int, default=None, help="총 환경 스텝 덮어쓰기(스모크용)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_path(cfg, k, v)
    seed = cfg["seed"] if args.seed is None else args.seed
    cfg["seed"] = seed
    algo = cfg["algo"]
    akey = "dqn" if algo == "dqn" else "a2c"
    if args.steps:
        cfg.setdefault(akey, {})["total_steps"] = args.steps
        cfg["train"]["eval_every"] = max(args.steps // 4, 1)
    out = run_dir_of(cfg, seed, args.tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    set_seed(seed)
    torch.set_num_threads(int(cfg.get("threads", 3)))
    device = device_of(args.device or cfg.get("device"))
    rw, ec = RewardConfig.from_dict(cfg["reward"]), EnvConfig.from_dict(cfg["env"])
    tr = load_log(cfg["cache_dir"], "train", cfg["train"]["conds"], cfg.get("img_pca"))
    va = load_log(cfg["cache_dir"], "val", [cfg["train"]["val_cond"]], cfg.get("img_pca"))
    _, enc = build_net(cfg, tr.feat_dim)
    acfg = (DQNConfig if algo == "dqn" else A2CConfig).from_dict(cfg.get(akey))
    env = BatchedViewSwitchEnv(tr, acfg.n_envs, cfg["train"]["conds"], rw, ec, seed=seed)

    holder = {}
    val_fn = lambda: evaluate_policy(greedy_policy(score_fn(holder["net"], algo), device),
                                     va, cfg["train"]["val_cond"], rw, ec)

    class _V(Validator):
        def maybe(self, step, module, extra=None, force=False):
            holder["net"] = module
            return super().maybe(step, module, extra, force)

    val = _V(val_fn, cfg["train"]["eval_every"])
    print(f"[학습] {algo} seed={seed} λ_sw={rw.lambda_sw} λ_miss={rw.lambda_miss} "
          f"conds={cfg['train']['conds']} → {out}", flush=True)
    t0 = time.time()
    if algo == "dqn":
        net = train_dqn(env, enc, acfg, device, val, seed)
    elif algo == "a2c":
        net = train_a2c(env, enc, acfg, device, val, seed)
    else:
        net = train_ppo(env, enc, acfg, device, val, seed)
    torch.save({"model": net.state_dict(), "feat_dim": tr.feat_dim}, os.path.join(out, "model.pt"))
    with open(os.path.join(out, "history.json"), "w", encoding="utf-8") as f:
        json.dump(dict(history=val.history, best_val_ret=val.best,
                       train_time_s=round(time.time() - t0, 1)), f, indent=2)
    del tr, env
    res = evaluate_run(out)
    for c, r in res.items():
        print(f"  [test/{c}] ret {r['ret']:.3f} vis {r['visibility_haz5']:.1f}% flicker {r['flicker']:.3f} "
              f"sw/min {r['switch_per_min']:.2f} gold top1 {r.get('gold_top1')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
