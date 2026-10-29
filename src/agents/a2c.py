"""A2C (Mnih et al., 2016의 동기식 변형) — n-step 이득 + 엔트로피 정규화.

정책 logit(a)는 채널 a 표현에서, 가치 V(s)는 채널 표현 풀링에서 나온다. PPO도 같은 네트워크를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.encoders.state_encoder import ChannelHead, EncoderConfig, StateEncoder, ValueHead, to_torch
from src.env.view_switch_env import BatchedViewSwitchEnv


class ActorCritic(nn.Module):
    def __init__(self, enc: EncoderConfig):
        super().__init__()
        self.encoder = StateEncoder(enc)
        self.pi = ChannelHead(self.encoder.out_dim)
        self.v = ValueHead(self.encoder.out_dim)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.pi(h), self.v(h)

    def logits(self, obs):
        return self.pi(self.encoder(obs))


@dataclass
class A2CConfig:
    total_steps: int = 400_000
    n_envs: int = 32
    n_steps: int = 8
    gamma: float = 0.9
    gae_lambda: float = 1.0          # 1.0 = 순수 n-step 이득 (A2C)
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    grad_clip: float = 0.5
    # PPO 전용
    epochs: int = 4
    minibatch: int = 512
    clip: float = 0.2

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def collect(env: BatchedViewSwitchEnv, net: ActorCritic, n_steps: int, device, rng):
    """롤아웃 수집 — 관측 대신 상태 인덱스를 저장하고 학습 때 다시 조립한다."""
    S = {k: [] for k in ("frame", "prev", "since", "cond", "action", "logp", "value", "reward", "done")}
    for _ in range(n_steps):
        frame, prev, since, cond = env.state()
        with torch.no_grad():
            logits, v = net(to_torch(env.obs(), device))
            dist = Categorical(logits=logits)
            a = dist.sample()
        a_np = a.cpu().numpy()
        r, done = env.step(a_np)
        for k, x in (("frame", frame), ("prev", prev), ("since", since), ("cond", cond),
                     ("action", a_np), ("logp", dist.log_prob(a).cpu().numpy()),
                     ("value", v.cpu().numpy()), ("reward", r), ("done", done.astype(np.float32))):
            S[k].append(x)
    with torch.no_grad():
        _, last_v = net(to_torch(env.obs(), device))
    out = {k: np.stack(v) for k, v in S.items()}             # (n_steps, n_envs)
    out["last_value"] = last_v.cpu().numpy()
    return out


def gae(roll, gamma, lam):
    T = roll["reward"].shape[0]
    adv = np.zeros_like(roll["reward"], dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nv = roll["last_value"] if t == T - 1 else roll["value"][t + 1]
        nonterm = 1.0 - roll["done"][t]
        delta = roll["reward"][t] + gamma * nv * nonterm - roll["value"][t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + roll["value"]


def _batch_obs(core, roll, idx=None):
    f = lambda k: roll[k].reshape(-1) if idx is None else roll[k].reshape(-1)[idx]
    return core.gather(f("frame"), f("prev"), f("since"), f("cond"))


def train_a2c(env, enc: EncoderConfig, cfg: A2CConfig, device, validator=None, seed=0,
              ppo: bool = False) -> ActorCritic:
    rng = np.random.default_rng(seed)
    net = ActorCritic(enc).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    core = env.core
    steps = 0
    n_updates = cfg.total_steps // (cfg.n_envs * cfg.n_steps)
    for u in range(n_updates):
        for g in opt.param_groups:                        # 선형 학습률 감소
            g["lr"] = cfg.lr * (1 - u / n_updates)
        roll = collect(env, net, cfg.n_steps, device, rng)
        steps += cfg.n_envs * cfg.n_steps
        adv, ret = gae(roll, cfg.gamma, cfg.gae_lambda)
        A = torch.as_tensor(adv.reshape(-1), device=device)
        R = torch.as_tensor(ret.reshape(-1), device=device)
        act = torch.as_tensor(roll["action"].reshape(-1), device=device)
        old_logp = torch.as_tensor(roll["logp"].reshape(-1), device=device)
        N = len(A)
        if not ppo:
            logits, v = net(to_torch(_batch_obs(core, roll), device))
            dist = Categorical(logits=logits)
            pg = -(dist.log_prob(act) * A).mean()
            loss = pg + cfg.vf_coef * (R - v).pow(2).mean() - cfg.ent_coef * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        else:
            for _ in range(cfg.epochs):
                perm = rng.permutation(N)
                for s in range(0, N, cfg.minibatch):
                    mb = perm[s:s + cfg.minibatch]
                    mbt = torch.as_tensor(mb, device=device)
                    logits, v = net(to_torch(_batch_obs(core, roll, mb), device))
                    dist = Categorical(logits=logits)
                    a_mb = A[mbt]
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    ratio = (dist.log_prob(act[mbt]) - old_logp[mbt]).exp()
                    pg = -torch.min(ratio * a_mb, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_mb).mean()
                    loss = pg + cfg.vf_coef * (R[mbt] - v).pow(2).mean() - cfg.ent_coef * dist.entropy().mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        if validator is not None:
            tr = float(np.mean(env.finished_returns[-200:])) if env.finished_returns else float("nan")
            validator.maybe(steps, net, dict(train_ret=tr))
    if validator is not None:
        validator.maybe(steps, net, force=True)
        if validator.best_state is not None:
            net.load_state_dict(validator.best_state)
    return net


def train_ppo(env, enc, cfg: A2CConfig, device, validator=None, seed=0) -> ActorCritic:
    return train_a2c(env, enc, cfg, device, validator, seed, ppo=True)
