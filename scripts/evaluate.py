"""평가 — 비교 정책(B1·RISK·B2·B2c·SUP·+hold·ORACLE)과 학습된 RL 실행.

    # 비교 정책: λ_sw 값마다 누적 보상이 달라지므로 보상 설정별로 저장
    python scripts/evaluate.py baselines --lsw 0 0.05 0.1 0.2 0.3 0.5 --conds gt noisy-medium
    # 학습된 실행을 다른 입력 조건(강건성)으로 재평가
    python scripts/evaluate.py runs results/runs/dqn_lsw0.1/seed* --conds robustness

결과: results/baselines/{split}_{조건}_lsw{λ}.json  ({정책: 지표})
      {run}/{split}_{조건}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.policies import BASELINES, hold  # noqa: E402
from src.data.log import load_gold, load_log  # noqa: E402
from src.env.view_switch_env import RewardConfig, episodes_of  # noqa: E402
from src.eval.metrics import evaluate_actions  # noqa: E402
from src.utils.config import load_config  # noqa: E402

LEVELS = ("light", "medium", "heavy")
TYPES = ("miss", "fp", "pos", "drop")
ROBUSTNESS = ["gt"] + [f"noisy-{lv}" for lv in LEVELS] + [f"noisy-{lv}:{t}" for lv in LEVELS for t in TYPES]
HOLD_K = 4      # 최소 유지 2초 — flicker 창(1.5초)보다 길어 정의상 flicker 0


def expand(conds: list[str]) -> list[str]:
    return ROBUSTNESS if conds == ["robustness"] else conds


def eval_baselines(cfg: dict, split: str, conds: list[str], lsws: list[float], out_dir: str):
    log = load_log(cfg["cache_dir"], split, conds)
    gold = load_gold(cfg["cache_dir"]) if split == "test" else None
    eps = episodes_of(log, cfg["env"].get("concat_scenes", 1))
    os.makedirs(out_dir, exist_ok=True)
    for lsw in lsws:
        rw = RewardConfig.from_dict({**cfg["reward"], "lambda_sw": lsw})
        for c in conds:
            res = {}
            for name, fn in BASELINES.items():
                try:
                    a, s = fn(log, c, rw)
                except KeyError:
                    continue                                  # val에는 B2(오라클 기동) 출력이 없다
                res[name] = evaluate_actions(log, a, s, rw, eps, gold)
                if name in ("RISK", "B2c", "SUP"):
                    res[f"{name}+hold"] = evaluate_actions(log, hold(log, a, HOLD_K), s, rw, eps, gold)
            path = os.path.join(out_dir, f"{split}_{c}_lsw{lsw:g}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"[{split}/{c}/λ_sw={lsw:g}] " + "  ".join(
                f"{k} {v['ret']:.2f}" for k, v in res.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("--config", default="configs/base.yaml")
    b.add_argument("--split", default="test")
    b.add_argument("--conds", nargs="+", default=["gt", "noisy-medium"])
    b.add_argument("--lsw", nargs="+", type=float, default=[0.1])
    b.add_argument("--out", default="results/baselines")
    r = sub.add_parser("runs")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--conds", nargs="+", default=None)
    r.add_argument("--split", default=None)
    args = ap.parse_args()

    if args.cmd == "baselines":
        eval_baselines(load_config(args.config), args.split, expand(args.conds), args.lsw, args.out)
    else:
        from train import evaluate_run
        for d in args.dirs:
            res = evaluate_run(d, expand(args.conds) if args.conds else None, args.split)
            print(d, {c: round(v["ret"], 3) for c, v in res.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
