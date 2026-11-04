"""Evaluate trained policies deterministically on fresh random layouts.

    python -m arm.evaluate            # both tasks, 2000 episodes each
"""
import sys
from pathlib import Path

import torch

from .sim import HOME, PushEnv, ReachEnv, arm_step, fk
from .train import ActorCritic

ROOT = Path(__file__).resolve().parent.parent


def load(task):
    ck = torch.load(ROOT / "checkpoints" / f"{task}.pt", weights_only=False)
    ac = ActorCritic(ck["obs_dim"])
    ac.load_state_dict(ck["model"])
    mean, std = ck["mean"], torch.sqrt(ck["var"] + 1e-8)
    return lambda obs: ac.actor(((obs - mean) / std).clamp(-5, 5)).clamp(-1, 1)


@torch.no_grad()
def eval_reach(n=2000, steps=150, seed=123):
    act, env = load("reach"), ReachEnv(n, seed=seed)
    env.q[:] = torch.tensor(HOME)
    env.qd[:] = 0
    env.target = env.sample_target(n)
    hit_at = torch.full((n,), -1)
    for t in range(steps):
        obs = env.obs()
        env.q, env.qd = arm_step(env.q, env.qd, act(obs))
        d = (env.target - fk(env.q)).norm(dim=-1)
        hit_at[(d < 0.03) & (hit_at < 0)] = t
    final = (env.target - fk(env.q)).norm(dim=-1)
    ok = hit_at >= 0
    return {"reached_within_3cm": ok.float().mean().item(),
            "median_time_s": (hit_at[ok].float().median().item() + 1) * 0.02 if ok.any() else None,
            "median_final_error_cm": final.median().item() * 100}


@torch.no_grad()
def eval_push(n=2000, seed=123):
    act, env = load("push"), PushEnv(n, seed=seed)
    env.reset(torch.ones(n, dtype=torch.bool))
    env.q[:] = torch.tensor(HOME)
    delivered_at = torch.full((n,), -1)
    for t in range(env.horizon - 1):
        env.step(act(env.obs()))
        d = (env.goal - env.puck).norm(dim=-1)
        delivered_at[(d < 0.05) & (delivered_at < 0)] = t
    final = (env.goal - env.puck).norm(dim=-1)
    ok = delivered_at >= 0
    return {"delivered_within_5cm": ok.float().mean().item(),
            "still_on_goal_at_end": (final < 0.05).float().mean().item(),
            "median_time_s": (delivered_at[ok].float().median().item() + 1) * 0.02 if ok.any() else None}


if __name__ == "__main__":
    for task in sys.argv[1:] or ["reach", "push"]:
        print(task, {"reach": eval_reach, "push": eval_push}[task]())
