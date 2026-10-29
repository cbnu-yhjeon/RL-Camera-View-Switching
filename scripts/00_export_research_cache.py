"""연구 저장소의 캐시를 RL 환경용 로그로 내보낸다.

연구 저장소(Adaptive-Camera-View-Selection-System)의 **가상환경과 작업 디렉터리**에서 실행한다.
nuScenes 래퍼·특징 캐시·위험 기여 캐시·노이즈 시뮬레이터·규칙 베이스라인·지도학습 모델을
그대로 import하므로, 이 저장소의 RL 실험은 연구 저장소와 같은 입력·같은 손상·같은 정의를 본다.

    cd ~/Adaptive-Camera-View-Selection-System
    .venv/bin/python ~/RL-Camera-View-Switching/scripts/00_export_research_cache.py \
        --out ~/RL-Camera-View-Switching/data/cache

출력 (분할마다)
    {split}.npz       scene 시간순 프레임 배열 — 행동과 무관한 로그
        tokens, scene_names, scene_offsets(S+1)
        img (N,6,576) float16   MobileNetV3 GAP 특징
        can (N,8)               정규화 CAN 7 + 결측 마스크
        primary (N,)            진행 방향 채널(라벨)
        maneuver (N,)           라벨 기동(constants.MANEUVERS 인덱스)
        label_att (N,) label_aux (N,6)
        risk_gt (N,6)           GT 채널 위험 Σw — 보상 신호
        haz5 (N,6) haz3 (N,6)   '실제 위험' 채널 (TTC<5s·DCA<3m, 그리고 TTC<3s·DCA<3m)
    {split}_cond.npz  입력 조건별 배열. 키는 "{조건}__{이름}".
        risk (N,6)              관측에 들어가는 위험도 (frame_risk — 연구 저장소 평가와 같은 손상)
        blackout (N,6)          영상 결손 채널
        sup_prob (N,6)          지도학습 모델(SUP) 강조 확률            (val·test)
        b2_att/b2_score         B2 규칙(오라클 기동)                     (test)
        b2c_att/b2c_score       B2c 규칙(과거만) — 같은 정보 조건의 규칙 (val·test)
    gold.json         골드셋 {token: {set, weight, stratum}}
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())

from src.baselines.rule_based import RuleBased  # noqa: E402
from src.data import contrib_cache  # noqa: E402
from src.data.dataset import ViewSelectionDataset, collate  # noqa: E402
from src.data.feature_cache import FeatureCache  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.data.splits import load_split, split_for_names  # noqa: E402
from src.labeling.label_function import LabelConfig, is_hazard  # noqa: E402
from src.models.proposed import ModelConfig, ViewSelector  # noqa: E402
from src.utils.config import load_cfgs  # noqa: E402
from src.utils.constants import MANEUVER_INDEX, NUM_CAMERAS  # noqa: E402

LEVELS = ("light", "medium", "heavy")
TYPES = ("miss", "fp", "pos", "drop")
# 학습 증강 실현 — 연구 저장소 noise_mix(clean·light·medium·heavy)와 같은 강도 구성.
# clean은 gt 조건 그대로 쓰므로 여기엔 손상 실현만 둔다. seed를 달리해 실현을 늘린다.
TRAIN_AUG = [("light", 1001), ("medium", 1002), ("heavy", 1003), ("medium", 1004)]


def conditions(split: str) -> list[tuple[str, str, str, int]]:
    """(이름, risk_source, preset, seed)."""
    out = [("gt", "gt", "clean", 42)]
    if split == "train":
        out += [(f"aug{k}", "noisy", p, s) for k, (p, s) in enumerate(TRAIN_AUG)]
    elif split == "val":
        out += [("noisy-medium", "noisy", "medium", 42)]
    else:
        out += [(f"noisy-{lv}", "noisy", lv, 42) for lv in LEVELS]
        out += [(f"noisy-{lv}:{t}", "noisy", f"{lv}:{t}", 42) for lv in LEVELS for t in TYPES]
    return out


@torch.no_grad()
def sup_probs(model, ds, device) -> np.ndarray:
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    out = []
    for b in dl:
        for k in ("img_feat", "can", "risk", "pad_mask", "primary_view"):
            b[k] = b[k].to(device)
        out.append(torch.sigmoid(model(b)["attn_logits"]).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def rule_arrays(rule, ds, tokens):
    pred = rule.run(ds)
    att = np.array([pred[t]["attention"] for t in tokens], dtype=np.int8)
    score = np.array([pred[t]["score"] for t in tokens], dtype=np.float32)
    return att, score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--ckpt", default=None, help="SUP 체크포인트 (기본: 최신 trainval proposed)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base, lab, tc = load_cfgs("configs/train_proposed.yaml")
    version = base["version"]
    cfg = LabelConfig.from_dict(lab)
    cfg3 = dataclasses.replace(cfg, hazard_ttc=3.0)
    w = NuscWrapper(base["dataroot"], version, require_image_files=base.get("require_image_files", True))
    cache_dir = os.path.join(base.get("cache_dir", "cache"), version)
    fc = FeatureCache(cache_dir)
    df = pd.read_parquet(f"data/labels_{version}.parquet")
    split = split_for_names(load_split(base["splits_file"]), {w.scene_name(t) for t in w.scene_tokens()})
    cc = contrib_cache.load_or_build(w, w.all_sample_tokens(), cfg, cache_dir)
    seq_len = tc["train"].get("seq_len", 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt
    if ckpt is None:
        sys.path.insert(0, "scripts")
        ev = __import__("06_evaluate")
        ckpt = ev.latest_ckpt("proposed", version)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = ViewSelector(ModelConfig(**state["mcfg"])).to(device).eval()
    model.load_state_dict(state["model"])
    print(f"[SUP] {ckpt}")
    b2c = RuleBased(cfg, "causal", wrapper=w)        # 과거 기동 추정은 조건과 무관 — 인스턴스 재사용
    b2 = RuleBased(cfg, "oracle")

    def dataset(names, source, preset, seed):
        return ViewSelectionDataset(w, fc, df, cfg, names, seq_len=seq_len, risk_source=source,
                                    noise_preset=preset, train_risk_noise=False, seed=seed,
                                    contrib_cache=cc)

    for sp in args.splits:
        t0 = time.time()
        ds0 = dataset(split[sp], "gt", "clean", 42)
        tokens = [ds0.scene_tokens[sc][pos] for sc, pos in ds0.flat]
        scenes = list(ds0.scene_tokens)
        offsets = np.cumsum([0] + [len(ds0.scene_tokens[s]) for s in scenes])
        L = df.set_index("sample_token").loc[tokens]
        n = len(tokens)
        img = np.stack([fc.get(t) for t in tokens]).astype(np.float16)
        can = np.zeros((n, 8), dtype=np.float32)
        haz5 = np.zeros((n, NUM_CAMERAS), dtype=bool)
        haz3 = np.zeros((n, NUM_CAMERAS), dtype=bool)
        for i, t in enumerate(tokens):
            f, m = cc.can(t)
            can[i, :len(f)] = f
            can[i, -1] = m
            for c in cc.get(t):
                if is_hazard(c, cfg):
                    haz5[i, c["vis"]] = True
                if is_hazard(c, cfg3):
                    haz3[i, c["vis"]] = True
        np.savez(os.path.join(args.out, f"{sp}.npz"),
                 tokens=np.array(tokens), scene_names=np.array([w.scene_name(s) for s in scenes]),
                 scene_offsets=offsets, img=img, can=can,
                 primary=L["primary_view"].to_numpy().astype(np.int8),
                 maneuver=np.array([MANEUVER_INDEX[m] for m in L["maneuver"]], dtype=np.int8),
                 label_att=L["attention_view"].to_numpy().astype(np.int8),
                 label_aux=np.stack([L[f"aux_{c}"].to_numpy() for c in range(NUM_CAMERAS)], 1).astype(bool),
                 risk_gt=np.stack([ds0.frame_risk(t) for t in tokens]).astype(np.float32),
                 haz5=haz5, haz3=haz3)
        print(f"[{sp}] {len(scenes)} scene / {n} 프레임 / 위험 채널 {haz5.sum()} ({time.time()-t0:.0f}s)")

        cond = {}
        for name, source, preset, seed in conditions(sp):
            t1 = time.time()
            ds = dataset(split[sp], source, preset, seed)
            assert [ds.scene_tokens[sc][pos] for sc, pos in ds.flat] == tokens
            cond[f"{name}__risk"] = np.stack([ds.frame_risk(t) for t in tokens]).astype(np.float32)
            blk = np.zeros((n, NUM_CAMERAS), dtype=bool)
            for k, s in enumerate(scenes):
                chs, _ = ds._scene_blackout(s, source, preset)
                blk[offsets[k]:offsets[k + 1], chs] = True
            cond[f"{name}__blackout"] = blk
            if sp != "train":
                cond[f"{name}__sup_prob"] = sup_probs(model, ds, device)
                cond[f"{name}__b2c_att"], cond[f"{name}__b2c_score"] = rule_arrays(b2c, ds, tokens)
                if sp == "test":
                    cond[f"{name}__b2_att"], cond[f"{name}__b2_score"] = rule_arrays(b2, ds, tokens)
            print(f"  {name}: {time.time()-t1:.0f}s")
        np.savez(os.path.join(args.out, f"{sp}_cond.npz"), **cond)

    sys.path.insert(0, "scripts")
    gold = __import__("06_evaluate").load_gold("data/goldset_v1.json")
    with open(os.path.join(args.out, "gold.json"), "w", encoding="utf-8") as f:
        json.dump({t: dict(set=sorted(v["set"]), weight=v["weight"], stratum=v["stratum"])
                   for t, v in gold.items()}, f, ensure_ascii=False)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(version=version, sup_ckpt=os.path.abspath(ckpt), seq_len=seq_len,
                       train_aug=TRAIN_AUG, hazard=dict(ttc=cfg.hazard_ttc, dca=cfg.d_hazard),
                       ema_alpha=cfg.ema_alpha, lambda_m=cfg.lambda_m), f, indent=2)
    print(f"[완료] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
