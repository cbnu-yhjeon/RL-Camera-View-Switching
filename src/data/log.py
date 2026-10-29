"""주행 로그 — 연구 저장소에서 내보낸 분할별 배열 (scripts/00_export_research_cache.py).

뷰 선택은 세계에 영향을 주지 않으므로 환경의 외생 상태 xₜ는 전부 이 배열에서 읽는다.
프레임 i는 scene 시간순으로 평탄화된 keyframe(2 Hz)이고, scene k는 [scene_offsets[k], scene_offsets[k+1]).

관측 위험도는 '조건(cond)'마다 다르다 — gt(검출 오류 없음) 또는 연구 저장소의 검출 오류 시뮬레이터를
거친 noisy 실현. 보상은 조건과 무관하게 항상 GT 위험(risk_gt)·GT 위험 판정(haz)으로 계산한다
(연구 저장소 원칙 2: 정답은 GT로 만들고, 입력은 손상될 수 있다).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
NUM_CAMERAS = len(CAMERAS)
FRAME_HZ = 2.0                                   # nuScenes keyframe 주기

# 기동 → 채널 적합도 m(c) — 연구 저장소 constants.MANEUVER_VIEW_MAP 과 같은 값, 같은 순서
# (MANEUVERS: STRAIGHT, TURN_LEFT, TURN_RIGHT, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, REVERSE, STOP).
MANEUVER_FIT = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0, 0.3, 1.0],
    [0.5, 1.0, 0.3, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0, 1.0, 0.5],
    [0.5, 0.5, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.5, 0.0],
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
], dtype=np.float32)


@dataclass
class ViewLog:
    split: str
    tokens: np.ndarray
    scene_names: np.ndarray
    scene_offsets: np.ndarray
    img: np.ndarray            # (N,6,D) float16
    can: np.ndarray            # (N,8)
    primary: np.ndarray        # (N,)
    maneuver: np.ndarray
    label_att: np.ndarray
    label_aux: np.ndarray
    risk_gt: np.ndarray        # (N,6)
    haz: dict[str, np.ndarray]                     # "haz5"/"haz3" → (N,6) bool
    cond: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    scene_of: np.ndarray = field(init=False)       # 프레임 → scene 번호
    scene_start: np.ndarray = field(init=False)    # 프레임 → 그 scene의 첫 프레임

    def __post_init__(self):
        lens = np.diff(self.scene_offsets)
        self.scene_of = np.repeat(np.arange(len(lens)), lens)
        self.scene_start = self.scene_offsets[self.scene_of]

    @property
    def n_frames(self) -> int:
        return len(self.tokens)

    @property
    def n_scenes(self) -> int:
        return len(self.scene_offsets) - 1

    @property
    def feat_dim(self) -> int:
        return self.img.shape[-1]

    def scene_slice(self, k: int) -> slice:
        return slice(int(self.scene_offsets[k]), int(self.scene_offsets[k + 1]))

    def conditions(self) -> list[str]:
        return list(self.cond)

    def risk(self, cond: str) -> np.ndarray:
        return self.cond[cond]["risk"]

    def blackout(self, cond: str) -> np.ndarray:
        return self.cond[cond]["blackout"]


def img_pca(cache_dir: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """영상 특징 PCA (mean, W (D,k)) — **train 분할에서만** 적합한다(분할 누수 금지).

    백본 특징은 고정(MobileNetV3 GAP 576차원)이고, RL 학습은 같은 특징을 수십만 번 다시 읽는다.
    64차원으로 줄이면 학습이 수 배 빨라지고 분산은 대부분 남는다(설명 분산비는 파일에 기록).
    """
    path = os.path.join(cache_dir, f"pca{k}.npz")
    if not os.path.exists(path):
        x = np.load(os.path.join(cache_dir, "train.npz"))["img"].reshape(-1, 576).astype(np.float64)
        mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        w = vt[:k].T / (s[:k, None].T / np.sqrt(len(x)))            # 백색화 — 성분별 단위 분산
        np.savez(path, mean=mean.astype(np.float32), w=w.astype(np.float32),
                 explained=float((s[:k] ** 2).sum() / (s ** 2).sum()))
    z = np.load(path)
    return z["mean"], z["w"]


def load_log(cache_dir: str, split: str, conds: list[str] | None = None,
             pca: int | None = None) -> ViewLog:
    z = np.load(os.path.join(cache_dir, f"{split}.npz"))
    img = z["img"]
    if pca:
        mean, w = img_pca(cache_dir, pca)
        img = ((img.astype(np.float32) - mean) @ w).astype(np.float32)
    c = np.load(os.path.join(cache_dir, f"{split}_cond.npz"))
    cond: dict[str, dict[str, np.ndarray]] = {}
    for key in c.files:
        name, what = key.split("__", 1)
        if conds is not None and name not in conds:
            continue
        cond.setdefault(name, {})[what] = c[key]
    if conds is not None:
        missing = [k for k in conds if k not in cond]
        if missing:
            raise KeyError(f"{split}: 조건 없음 {missing} (있는 조건: {sorted({k.split('__')[0] for k in c.files})})")
    return ViewLog(split=split, tokens=z["tokens"], scene_names=z["scene_names"],
                   scene_offsets=z["scene_offsets"].astype(np.int64), img=img,
                   can=z["can"], primary=z["primary"].astype(np.int64),
                   maneuver=z["maneuver"].astype(np.int64),
                   label_att=z["label_att"].astype(np.int64), label_aux=z["label_aux"],
                   risk_gt=z["risk_gt"], haz={"haz5": z["haz5"], "haz3": z["haz3"]}, cond=cond)


def load_gold(cache_dir: str) -> dict | None:
    path = os.path.join(cache_dir, "gold.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def synthetic_log(n_scenes: int = 4, scene_len: int = 12, feat_dim: int = 8,
                  seed: int = 0, split: str = "synthetic") -> ViewLog:
    """단위 테스트·스모크용 합성 로그 (nuScenes 없이)."""
    rng = np.random.default_rng(seed)
    n = n_scenes * scene_len
    risk = (rng.random((n, NUM_CAMERAS)) ** 4).astype(np.float32)
    risk[:, 0] = 0.0
    haz = risk > 0.6
    blk = np.zeros((n, NUM_CAMERAS), dtype=bool)
    return ViewLog(split=split, tokens=np.array([f"t{i}" for i in range(n)]),
                   scene_names=np.array([f"scene-{k}" for k in range(n_scenes)]),
                   scene_offsets=np.arange(n_scenes + 1, dtype=np.int64) * scene_len,
                   img=rng.standard_normal((n, NUM_CAMERAS, feat_dim)).astype(np.float16),
                   can=rng.standard_normal((n, 8)).astype(np.float32),
                   primary=np.zeros(n, dtype=np.int64), maneuver=np.zeros(n, dtype=np.int64),
                   label_att=risk[:, 1:].argmax(1) + 1, label_aux=haz,
                   risk_gt=risk, haz={"haz5": haz, "haz3": haz},
                   cond={"gt": {"risk": risk, "blackout": blk}})
