"""DQN (Mnih et al., 2015) — Experience Replay + Target Network, Double DQN 타깃.

리플레이 버퍼는 관측 대신 **상태 인덱스**(frame, 표시 상태, 경과 시간, 입력 조건)만 저장한다.
로그 재생 환경이라 관측은 언제든 같은 값으로 다시 조립되므로, (T,6,576) 특징을 버퍼에 복사하지
않아도 된다. 다음 상태는 (frame+1, 새 표시 상태, 새 경과 시간, 같은 조건)이다.

gamma=0이면 즉시 보상만 보는 contextual bandit이 된다 (RL-0 비교용).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.state_encoder import ChannelHead, EncoderConfig, StateEncoder, ValueHead, to_torch
from src.env.view_switch_env import BatchedViewSwitchEnv


@dataclass
class DQNConfig:
    total_steps: int = 200_000      # 환경 전이 수
    n_envs: int = 16
    gamma: float = 0.9
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    train_every: int = 1            # 병렬 env 스텝마다 경사 스텝 수
    target_tau: float = 0.01        # 소프트 타깃 갱신
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3
    grad_clip: float = 10.0
    double: bool = True
    # 전 행동(반사실) 타깃: 세계 전이가 행동과 무관하고 보상은 로그로 계산되므로, 한 전이에서
    # 6개 행동 모두의 타깃 r(s,a) + γ·max Q(s'_a, ·)를 만든다. s'_a는 같은 xₜ₊₁에 표시 상태만 a.
    all_actions: bool = False
    dueling: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


class QNet(nn.Module):
    """Q(s,a) = V(xₜ) + A(xₜ, vₜ₋₁, a) − mean_a A   (dueling, Wang et al. 2016).

    가치 흐름 V는 **외생 상태 xₜ만** 본다. 이 MDP에서 표시 상태 d에 따른 가치 차이는
    |V(x,d) − V(x,d')| ≤ λ_sw로 작고(언제든 λ_sw를 내고 d로 바꿀 수 있다), 가치의 대부분은
    앞으로 들어올 위험 이득이라 행동·표시 상태와 무관하다. 둘을 한 출력으로 근사하면 큰 값의
    근사 오차(수 %)가 λ_sw 크기의 차이를 덮어 전환이 늘었다(실측). 분리하면 표시 상태 효과는
    작은 A 흐름에만 남는다. dueling=False면 채널 헤드 하나로 Q를 낸다.
    """

    def __init__(self, enc: EncoderConfig, dueling: bool = True):
        super().__init__()
        self.encoder = StateEncoder(enc)
        self.head = ChannelHead(self.encoder.out_dim)
        self.value = ValueHead(enc.d) if dueling else None

    def _q(self, h, extra):
        a = self.head(torch.cat([h, extra], -1))
        if self.value is None:
            return a
        return self.value(h)[:, None] + a - a.mean(1, keepdim=True)

    def forward(self, obs):
        h = self.encoder.core(obs)
        return self._q(h, self.encoder.extras(obs["prev"], obs["primary"], obs["since"]))

    def q_all_displays(self, obs, prevs: list[torch.Tensor], sinces: list[torch.Tensor]):
        """같은 xₜ에서 표시 상태만 다른 K개 상태의 Q (K,B,6) — 인코더는 한 번만 돈다."""
        h = self.encoder.core(obs)
        return torch.stack([self._q(h, self.encoder.extras(p, obs["primary"], s))
                            for p, s in zip(prevs, sinces)])


class IndexReplay:
    def __init__(self, size: int, rng: np.random.Generator):
        self.size, self.rng = size, rng
        self.ptr = self.n = 0
        z = lambda dt=np.int64: np.zeros(size, dtype=dt)
        self.frame, self.prev, self.since, self.cond = z(), z(), z(), z()
        self.action, self.nprev, self.nsince = z(), z(), z()
        self.reward, self.done = z(np.float32), z(np.float32)

    def add(self, frame, prev, since, cond, action, reward, nprev, nsince, done):
        k = len(frame)
        idx = (self.ptr + np.arange(k)) % self.size
        for arr, v in ((self.frame, frame), (self.prev, prev), (self.since, since),
                       (self.cond, cond), (self.action, action), (self.reward, reward),
                       (self.nprev, nprev), (self.nsince, nsince), (self.done, done)):
            arr[idx] = v
        self.ptr = (self.ptr + k) % self.size
        self.n = min(self.n + k, self.size)

    def sample(self, b: int):
        i = self.rng.integers(self.n, size=b)
        return {k: getattr(self, k)[i] for k in
                ("frame", "prev", "since", "cond", "action", "reward", "nprev", "nsince", "done")}


def _onehot_prev(disp: np.ndarray, device) -> torch.Tensor:
    from src.env.view_switch_env import N_DISPLAY
    p = torch.zeros(len(disp), N_DISPLAY, device=device)
    p[torch.arange(len(disp)), torch.as_tensor(disp, device=device)] = 1.0   # -1 → '없음' 칸
    return p


def _all_action_loss(q, tgt, core, b, cfg: DQNConfig, device):
    """한 상태의 6개 행동 전부를 회귀한다 (Double DQN 타깃)."""
    o = to_torch(core.gather(b["frame"], b["prev"], b["since"], b["cond"]), device)
    q_all = q(o)                                                          # (B,6)
    if cfg.gamma == 0:
        ys = [core.transition(b["frame"], np.full(len(b["frame"]), a), b["prev"], b["since"])[0]
              for a in range(6)]
        y = torch.as_tensor(np.stack(ys, 1), dtype=torch.float32, device=device)
        return F.smooth_l1_loss(q_all, y)
    nf = np.minimum(b["frame"] + 1, core.log.n_frames - 1)
    no = to_torch(core.gather(nf, b["prev"], b["since"], b["cond"]), device)
    cap = core.cfg.since_cap
    rs, prevs, sinces = [], [], []
    for a in range(6):
        r, disp, since, _ = core.transition(b["frame"], np.full(len(b["frame"]), a), b["prev"], b["since"])
        rs.append(r)
        prevs.append(_onehot_prev(disp, device))
        sinces.append(torch.as_tensor(np.minimum(since, cap) / cap, dtype=torch.float32,
                                      device=device)[:, None])
    with torch.no_grad():
        qn_t = tgt.q_all_displays(no, prevs, sinces)                     # (6,B,6)
        if cfg.double:
            an = q.q_all_displays(no, prevs, sinces).argmax(-1, keepdim=True)
            qn = qn_t.gather(-1, an).squeeze(-1)
        else:
            qn = qn_t.max(-1).values                                      # (6,B)
    r = torch.as_tensor(np.stack(rs), dtype=torch.float32, device=device)  # (6,B)
    dn = torch.as_tensor(b["done"], device=device)
    y = (r + cfg.gamma * (1 - dn) * qn).T                                  # (B,6)
    return F.smooth_l1_loss(q_all, y)


def train_dqn(env: BatchedViewSwitchEnv, enc: EncoderConfig, cfg: DQNConfig, device,
              validator=None, seed: int = 0) -> QNet:
    rng = np.random.default_rng(seed)
    q = QNet(enc, cfg.dueling).to(device)
    tgt = QNet(enc, cfg.dueling).to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf = IndexReplay(cfg.buffer_size, rng)
    core = env.core
    steps = 0
    loss_ema = None
    while steps < cfg.total_steps:
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end)
                  * steps / max(1.0, cfg.eps_fraction * cfg.total_steps))
        frame, prev, since, cond = env.state()
        with torch.no_grad():
            qa = q(to_torch(env.obs(), device)).argmax(1).cpu().numpy()
        rand = rng.random(env.n) < eps
        actions = np.where(rand, rng.integers(6, size=env.n), qa)
        r, done = env.step(actions)
        # env.step은 끝난 환경을 이미 리셋했으므로 다음 표시 상태는 전이 함수로 다시 구한다.
        _, nprev, nsince, _ = core.transition(frame, actions, prev, since)
        buf.add(frame, prev, since, cond, actions, r, nprev, nsince, done.astype(np.float32))
        steps += env.n

        if buf.n >= cfg.learning_starts:
            for _ in range(cfg.train_every):
                b = buf.sample(cfg.batch_size)
                if cfg.all_actions:
                    loss = _all_action_loss(q, tgt, core, b, cfg, device)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
                    opt.step()
                    with torch.no_grad():
                        for p, pt in zip(q.parameters(), tgt.parameters()):
                            pt.mul_(1 - cfg.target_tau).add_(p, alpha=cfg.target_tau)
                    loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
                    continue
                o = to_torch(core.gather(b["frame"], b["prev"], b["since"], b["cond"]), device)
                nf = np.minimum(b["frame"] + 1, core.log.n_frames - 1)
                no = to_torch(core.gather(nf, b["nprev"], b["nsince"], b["cond"]), device)
                a = torch.as_tensor(b["action"], device=device)
                rew = torch.as_tensor(b["reward"], device=device)
                dn = torch.as_tensor(b["done"], device=device)
                with torch.no_grad():
                    if cfg.gamma > 0:
                        qn_t = tgt(no)
                        if cfg.double:
                            an = q(no).argmax(1, keepdim=True)
                            qn = qn_t.gather(1, an).squeeze(1)
                        else:
                            qn = qn_t.max(1).values
                        y = rew + cfg.gamma * (1 - dn) * qn
                    else:
                        y = rew
                qs = q(o).gather(1, a[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(qs, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
                opt.step()
                with torch.no_grad():
                    for p, pt in zip(q.parameters(), tgt.parameters()):
                        pt.mul_(1 - cfg.target_tau).add_(p, alpha=cfg.target_tau)
                loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if validator is not None:
            tr = float(np.mean(env.finished_returns[-200:])) if env.finished_returns else float("nan")
            validator.maybe(steps, q, dict(eps=round(eps, 3), loss=loss_ema, train_ret=tr))
    if validator is not None:
        validator.maybe(steps, q, force=True)
        if validator.best_state is not None:
            q.load_state_dict(validator.best_state)
    return q
