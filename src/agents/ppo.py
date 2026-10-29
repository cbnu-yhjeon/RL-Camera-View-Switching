"""PPO (Schulman et al., 2017) — clipped surrogate + GAE. 네트워크·롤아웃은 A2C와 공유한다."""

from src.agents.a2c import A2CConfig as PPOConfig, ActorCritic, train_ppo  # noqa: F401
