"""상태 인코더 — 연구 저장소 제안 모델(B.5)의 구조를 RL 상태에 맞춰 재사용한다.

    img_feat → Linear(D_f→d) ┐
    카메라 위치 임베딩(6) ───┤→ 채널 토큰 6개
    risk → Linear(1→d) 가산  ┘
    can → Linear(d_can→d) → CAN 토큰 1개
    [6 채널 토큰 + CAN 토큰] → TransformerEncoder → 채널 토큰 (B,T,6,d)
    → 채널별 GRU(시간축) → 마지막 스텝 (B,6,d)

RL용으로 더한 것: 표시 상태 vₜ₋₁·진행 방향·전환 후 경과 시간을 채널별 특징으로 붙인다.
행동 a가 곧 채널 a이므로 Q(s,a)·정책 logit(a)는 **채널 a 토큰 하나에서 나온다** (채널 공유 헤드).
채널 순서를 바꿔도 같은 함수가 되어 6채널을 하나의 헤드로 학습한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.data.log import NUM_CAMERAS


@dataclass
class EncoderConfig:
    feat_dim: int = 576
    d_can: int = 8
    d: int = 64
    heads: int = 4
    ff: int = 128
    layers: int = 1
    dropout: float = 0.0
    use_img: bool = True
    use_risk: bool = True
    use_can: bool = True
    use_gru: bool = True

    @classmethod
    def from_dict(cls, d: dict | None, **over) -> "EncoderConfig":
        d = {**(d or {}), **over}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


N_EXTRA = 4   # 채널별 추가 특징: [이 채널이 표시 중, 진행 방향, 보조 뷰 없음 상태, 경과 시간]


class UnrolledGRU(nn.Module):
    """채널별 GRU(시간축, 1층) — 마지막 스텝 은닉 (B,T,6,d) → (B,6,d).

    nn.GRU와 같은 식이지만 행렬곱으로 풀어 쓴다. T=4로 짧고 시퀀스가 B×6개로 많아
    cuDNN GRU 커널이 오히려 느렸다(실측 수십 ms/스텝).
    """

    def __init__(self, d: int):
        super().__init__()
        self.wi = nn.Linear(d, 3 * d)
        self.wh = nn.Linear(d, 3 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gi = self.wi(x)
        h = torch.zeros_like(x[:, 0])
        for t in range(x.shape[1]):
            ir, iz, inn = gi[:, t].chunk(3, -1)
            hr, hz, hn = self.wh(h).chunk(3, -1)
            r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            h = (1 - z) * torch.tanh(inn + r * hn) + z * h
        return h


class StateEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.img_proj = nn.Sequential(nn.LayerNorm(cfg.feat_dim), nn.Linear(cfg.feat_dim, d))
        self.cam_emb = nn.Parameter(torch.randn(NUM_CAMERAS, d) * 0.02)
        self.risk_proj = nn.Linear(1, d)
        self.can_proj = nn.Linear(cfg.d_can, d)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, cfg.layers)
        self.gru = UnrolledGRU(d) if cfg.use_gru else None
        self.out_dim = d + N_EXTRA

    def core(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """외생 상태 xₜ(과거 T 프레임) → 채널별 표현 (B, 6, d)."""
        img, risk, can, pad = obs["img"], obs["risk"], obs["can"], obs["pad"]
        B, T = risk.shape[:2]
        tok = self.cam_emb.expand(B, T, NUM_CAMERAS, -1)
        if self.cfg.use_img:
            tok = tok + self.img_proj(img)
        if self.cfg.use_risk:
            tok = tok + self.risk_proj(risk.unsqueeze(-1))
        can_tok = self.can_proj(can) if self.cfg.use_can else torch.zeros_like(tok[:, :, 0])
        x = torch.cat([tok, can_tok.unsqueeze(2)], 2).reshape(B * T, NUM_CAMERAS + 1, -1)
        x = self.tf(x)[:, :NUM_CAMERAS].reshape(B, T, NUM_CAMERAS, -1)
        x = x * pad[:, :, None, None]                                   # 패딩 프레임 제거
        return self.gru(x) if self.gru is not None else x[:, -1]

    @staticmethod
    def extras(prev: torch.Tensor, primary: torch.Tensor, since: torch.Tensor) -> torch.Tensor:
        """표시 상태 특징 (B,6,N_EXTRA) — 외생 상태 표현과 분리해 두면 같은 xₜ에 표시 상태만
        바꾼 여러 상태를 인코더 한 번으로 평가할 수 있다(DQN 전 행동 타깃)."""
        return torch.stack([
            prev[:, :NUM_CAMERAS],
            primary,
            prev[:, NUM_CAMERAS:].expand(-1, NUM_CAMERAS),
            since.expand(-1, NUM_CAMERAS),
        ], -1)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """obs → 채널별 표현 (B, 6, d + N_EXTRA)."""
        h = self.core(obs)
        return torch.cat([h, self.extras(obs["prev"], obs["primary"], obs["since"])], -1)


class ChannelHead(nn.Module):
    """채널 공유 헤드 — 채널 표현 → 행동 점수 (Q값 또는 logit) (B,6)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class ValueHead(nn.Module):
    """상태 가치 V(s) — 채널 표현 평균 풀링 → MLP."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h.mean(1), h.amax(1)], -1)).squeeze(-1)


def to_torch(obs: dict, device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
