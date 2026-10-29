"""결과 표·그림 생성 — 수치를 손으로 옮기지 않는다 (연구 저장소 원칙 4).

입력: results/baselines/*.json, results/runs/{exp}_lsw{λ}/seed*/{test_*.json, history.json}
출력: results/report.md, results/summary.json, results/figures/*.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import LEVELS, TYPES  # noqa: E402

RUNS = "results/runs"
BASE = "results/baselines"
FIG = "results/figures"
LSWS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
MAIN_LSW = 0.1

RL_LABEL = {"bandit": "Bandit (γ=0)", "dqn": "DQN", "dqn_vanilla": "DQN (단일 행동 타깃)",
            "dqn_maneuver": "DQN-M (보상 + 기동 적합도)",
            "a2c": "A2C", "ppo": "PPO", "dqn_noisy": "DQN-N (노이즈 학습)"}
BASE_LABEL = {"B1": "B1 고정 전방", "RISK": "RISK 위험 최대 채널", "RISK+hold": "RISK + 2초 유지",
              "B2": "B2 규칙(오라클 기동)", "B2c": "B2c 규칙(과거만)", "B2c+hold": "B2c + 2초 유지",
              "SUP": "SUP 지도학습", "SUP+hold": "SUP + 2초 유지", "ORACLE": "ORACLE (DP, 미래·GT)"}
# 그림 범례는 영문 — 이 환경에 한글 글꼴이 없어 글자가 깨진다(표는 한글).
EN = {"bandit": "Bandit (γ=0)", "dqn": "DQN", "dqn_vanilla": "DQN (single-action target)", "a2c": "A2C",
      "ppo": "PPO", "dqn_noisy": "DQN-N (noise-trained)", "dqn_maneuver": "DQN-M (+maneuver fit)", "B1": "B1 fixed front", "RISK": "RISK max-risk channel",
      "RISK+hold": "RISK + 2 s hold", "B2": "B2 rule (oracle maneuver)", "B2c": "B2c rule (causal)",
      "B2c+hold": "B2c + 2 s hold", "SUP": "SUP supervised", "SUP+hold": "SUP + 2 s hold", "ORACLE": "ORACLE (DP)"}
COLS = [("ret", "누적 보상", "{:.2f}"), ("visibility_haz5", "가시화율 5s(%)", "{:.1f}"),
        ("visibility_haz3", "가시화율 3s(%)", "{:.1f}"), ("precision", "정밀도(%)", "{:.1f}"),
        ("flicker", "Flicker", "{:.3f}"), ("switch_per_min", "전환/분", "{:.1f}"),
        ("gold_top1", "골드 Top-1", "{:.3f}"), ("gold_top2", "골드 Top-2", "{:.3f}")]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_results(exp: str, lsw: float, split_cond: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(f"{RUNS}/{exp}_lsw{lsw:g}/seed*")):
        p = os.path.join(d, f"{split_cond}.json")
        if os.path.exists(p):
            out.append(load(p))
    return out


def agg(rows: list[dict], key: str):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v)) if len(v) > 1 else 0.0)


def fmt(rows, key, f):
    m, s = agg(rows, key)
    if m is None:
        return "—"
    return f.format(m) + (f" ± {f.format(s)}" if len(rows) > 1 else "")


def baseline(cond: str, lsw: float, split="test") -> dict:
    p = f"{BASE}/{split}_{cond}_lsw{lsw:g}.json"
    return load(p) if os.path.exists(p) else {}


def main_table(cond: str, lsw: float, exps: list[str]) -> str:
    lines = ["| 정책 | n | " + " | ".join(c[1] for c in COLS) + " |",
             "|---|---:|" + "---:|" * len(COLS)]
    for k, v in baseline(cond, lsw).items():
        lines.append(f"| {BASE_LABEL.get(k, k)} | 1 | " + " | ".join(fmt([v], c, f) for c, _, f in COLS) + " |")
    for e in exps:
        rows = run_results(e, lsw, f"test_{cond}")
        if rows:
            lines.append(f"| **{RL_LABEL.get(e, e)}** | {len(rows)} | "
                         + " | ".join(fmt(rows, c, f) for c, _, f in COLS) + " |")
    return "\n".join(lines)


def sweep_table(cond: str) -> str:
    lines = ["| λ_sw | ORACLE | RISK | B2c | SUP | Bandit (γ=0) | DQN | DQN − Bandit | DQN 전환/분 | Bandit 전환/분 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lsw in LSWS:
        b = baseline(cond, lsw)
        if not b:
            continue
        bd, dq = run_results("bandit", lsw, f"test_{cond}"), run_results("dqn", lsw, f"test_{cond}")
        mb, _ = agg(bd, "ret")
        md, _ = agg(dq, "ret")
        diff = "—" if mb is None or md is None else f"{md - mb:+.2f}"
        lines.append(f"| {lsw:g} | {b['ORACLE']['ret']:.2f} | {b['RISK']['ret']:.2f} | {b['B2c']['ret']:.2f} | "
                     f"{b['SUP']['ret']:.2f} | {fmt(bd, 'ret', '{:.2f}')} | {fmt(dq, 'ret', '{:.2f}')} | {diff} | "
                     f"{fmt(dq, 'switch_per_min', '{:.1f}')} | {fmt(bd, 'switch_per_min', '{:.1f}')} |")
    return "\n".join(lines)


def pareto_fig(cond: str, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, xkey, xlabel in ((axes[0], "switch_per_min", "switches per minute"),
                             (axes[1], "flicker", "Flicker rate")):
        for exp, color, marker in (("dqn", "C0", "o"), ("bandit", "C1", "s")):
            pts = []
            for lsw in LSWS:
                rows = run_results(exp, lsw, f"test_{cond}")
                if rows:
                    pts.append((agg(rows, xkey)[0], agg(rows, "visibility_haz5")[0], lsw))
            if pts:
                xs, ys, ls = zip(*pts)
                ax.plot(xs, ys, marker=marker, color=color, label=EN[exp])
                for x, y, l in pts:
                    ax.annotate(f"{l:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        b = baseline(cond, MAIN_LSW)
        orc = [(baseline(cond, l).get("ORACLE", {}).get(xkey), baseline(cond, l).get("ORACLE", {}).get("visibility_haz5"))
               for l in LSWS if baseline(cond, l)]
        if orc:
            ax.plot(*zip(*orc), "k--", lw=1, alpha=0.6, label="ORACLE (DP, future+GT)")
        for k, m in (("RISK", "^"), ("B2c", "D"), ("SUP", "v"), ("B2c+hold", "P"), ("RISK+hold", "X")):
            if k in b:
                ax.scatter(b[k][xkey], b[k]["visibility_haz5"], marker=m, s=50, label=EN[k], zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("hazard visibility (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Visibility vs. switching Pareto (test, input {cond}; labels = λ_sw)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def return_fig(cond: str, path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, src in (("ORACLE", "b"), ("RISK", "b"), ("B2c", "b"), ("SUP", "b"), ("bandit", "r"), ("dqn", "r")):
        xs, ys, es = [], [], []
        for lsw in LSWS:
            if src == "b":
                b = baseline(cond, lsw)
                if name in b:
                    xs.append(lsw); ys.append(b[name]["ret"]); es.append(0)
            else:
                rows = run_results(name, lsw, f"test_{cond}")
                if rows:
                    m, s = agg(rows, "ret")
                    xs.append(lsw); ys.append(m); es.append(s)
        if xs:
            lab = EN.get(name, name)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=lab,
                        ls="--" if name == "ORACLE" else "-")
    ax.set_xlabel("switching cost λ_sw")
    ax.set_ylabel("mean episode return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"Return vs. switching cost (test, {cond})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robustness(path_fig: str) -> str:
    methods = [("dqn", "r"), ("dqn_noisy", "r"), ("dqn_maneuver", "r"), ("B2c", "b"), ("SUP", "b"), ("RISK", "b")]
    conds = lambda t: ["gt"] + [f"noisy-{lv}:{t}" for lv in LEVELS]

    def value(m, src, c, key):
        if src == "b":
            v = baseline(c, MAIN_LSW).get(m, {}).get(key)
            return v, 0.0
        return agg(run_results(m, MAIN_LSW, f"test_{c}"), key)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey="row")
    for row, (key, ylab) in enumerate((("ret", "return"), ("visibility_haz5", "hazard visibility (%)"))):
        for j, t in enumerate(TYPES):
            ax = axes[row, j]
            for m, src in methods:
                ys = [value(m, src, c, key) for c in conds(t)]
                if all(y[0] is not None for y in ys):
                    ax.errorbar(range(4), [y[0] for y in ys], yerr=[y[1] or 0 for y in ys], marker="o",
                                capsize=3, label=EN.get(m, m))
            ax.set_xticks(range(4), ["clean", "light", "medium", "heavy"])
            if row == 0:
                ax.set_title(t)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Robustness to detection noise (test, λ_sw={MAIN_LSW:g}) by noise type and level")
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)

    lines = ["| 조건 | " + " | ".join(RL_LABEL.get(m, BASE_LABEL.get(m, m)) for m, _ in methods) + " |",
             "|---|" + "---:|" * len(methods)]
    for c in ["gt"] + [f"noisy-{lv}" for lv in LEVELS] + [f"noisy-{lv}:{t}" for t in TYPES for lv in LEVELS]:
        cells = []
        for m, src in methods:
            r, _ = value(m, src, c, "ret")
            v, _ = value(m, src, c, "visibility_haz5")
            cells.append("—" if r is None else f"{r:.2f} / {v:.1f}%")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def learning_fig(exps: list[str], path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, e in enumerate(exps):
        for k, d in enumerate(sorted(glob.glob(f"{RUNS}/{e}_lsw{MAIN_LSW:g}/seed*"))):
            hp = os.path.join(d, "history.json")
            if not os.path.exists(hp):
                continue
            h = load(hp)["history"]
            ax.plot([r["step"] for r in h], [r["ret"] for r in h], color=f"C{i}", alpha=0.7,
                    label=EN.get(e, e) if k == 0 else None)
    b = baseline("gt", MAIN_LSW, "val")
    for k, ls in (("ORACLE", "--"), ("B2c", ":"), ("RISK", "-.")):
        if k in b:
            ax.axhline(b[k]["ret"], color="k", ls=ls, lw=1, label=f"{EN[k]} (val)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("validation return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"Learning curves (val, gt, λ_sw={MAIN_LSW:g})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rl0_check() -> str:
    """RL-0: λ_sw=0에서 DQN(γ>0)과 Bandit(γ=0)의 행동 일치·성능 차이."""
    sys.path.insert(0, "scripts")
    from train import load_policy
    from src.data.log import load_log
    from src.env.view_switch_env import EnvConfig, RewardConfig
    from src.eval.rollout import rollout
    lines = ["| seed | DQN 보상 | Bandit 보상 | 행동 일치율 (전체) | 행동 일치율 (선택이 갈리는 프레임) |",
             "|---:|---:|---:|---:|---:|"]
    dq = sorted(glob.glob(f"{RUNS}/dqn_lsw0/seed*"))
    bd = sorted(glob.glob(f"{RUNS}/bandit_lsw0/seed*"))
    if not dq or not bd:
        return "(실행 없음)"
    log = None
    for d, b in zip(dq, bd):
        pd_, cfg = load_policy(d)
        pb, _ = load_policy(b)
        if log is None:
            log = load_log(cfg["cache_dir"], "test", ["gt"], cfg.get("img_pca"))
        rw, ec = RewardConfig.from_dict(cfg["reward"]), EnvConfig.from_dict(cfg["env"])
        a1, _, _ = rollout(pd_, log, "gt", rw, ec)
        a2, _, _ = rollout(pb, log, "gt", rw, ec)
        r1, r2 = load(f"{d}/test_gt.json")["ret"], load(f"{b}/test_gt.json")["ret"]
        # 모든 보조 뷰의 이득이 0이고 위험도 없으면 어느 행동이든 즉시 보상이 같다 — 그런 프레임은 뺀 일치율도 낸다
        g = log.risk_gt.copy()
        g[np.arange(log.n_frames), log.primary] = 0.0
        matters = (g.max(1) > 1e-3) | log.haz["haz5"].any(1)
        lines.append(f"| {os.path.basename(d)} | {r1:.3f} | {r2:.3f} | {100 * (a1 == a2).mean():.1f}% | "
                     f"{100 * (a1 == a2)[matters].mean():.1f}% (n={int(matters.sum())}) |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/report.md")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    exps = [e for e in ("bandit", "dqn", "dqn_vanilla", "a2c", "ppo", "dqn_noisy", "dqn_maneuver")
            if glob.glob(f"{RUNS}/{e}_lsw{MAIN_LSW:g}/seed*/test_gt.json")]
    for c in ("gt", "noisy-medium"):
        pareto_fig(c, f"{FIG}/pareto_{c}.png")
        return_fig(c, f"{FIG}/return_vs_lsw_{c}.png")
    learning_fig([e for e in exps if e not in ("dqn_noisy", "dqn_maneuver")], f"{FIG}/learning_curves.png")
    rob = robustness(f"{FIG}/robustness.png")
    md = [f"# 실험 결과 (자동 생성 — scripts/make_report.py)\n",
          f"시험 분할 127 scene / 5,108 프레임, 골드셋 285장(역확률 가중). RL 행은 seed 평균 ± 표준편차.\n",
          f"## 표 1. 정책 비교 (λ_sw={MAIN_LSW:g}, 입력 gt)\n", main_table("gt", MAIN_LSW, exps),
          f"\n## 표 2. 정책 비교 (λ_sw={MAIN_LSW:g}, 입력 noisy-medium — 연구 저장소 기본 평가 조건)\n",
          main_table("noisy-medium", MAIN_LSW, exps),
          "\n## 표 3. 전환 비용 스윕 — 누적 보상 (입력 gt)\n", sweep_table("gt"),
          "\n## 표 4. 전환 비용 스윕 — 누적 보상 (입력 noisy-medium)\n", sweep_table("noisy-medium"),
          "\n## 표 5. RL-0 — λ_sw=0에서 DQN과 Bandit 동등성 (test, gt)\n", rl0_check(),
          f"\n## 표 6. 노이즈 강건성 (λ_sw={MAIN_LSW:g}) — 누적 보상 / 가시화율\n", rob,
          "\n## 그림\n",
          "- `figures/pareto_gt.png`, `figures/pareto_noisy-medium.png` — 가시화율–전환 파레토",
          "- `figures/return_vs_lsw_gt.png`, `figures/return_vs_lsw_noisy-medium.png` — λ_sw별 누적 보상",
          "- `figures/robustness.png` — 노이즈 강건성", "- `figures/learning_curves.png` — 학습 곡선"]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
