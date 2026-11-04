"""Bootstrap the Push policy by imitating the scripted expert with DAgger.

    python -m arm.imitate                  # writes checkpoints/push.pt
    python -m arm.train push --init checkpoints/push.pt --steps 20e6   # then refine with PPO

DAgger: run a mix of expert and student actions, label every visited state with
the expert's action, and retrain the student on everything collected so far. The
student learns to recover from its own mistakes, not just to copy perfect runs.
"""
import argparse
import time
from pathlib import Path

import torch

from .expert import push_expert
from .sim import PushEnv
from .train import ActorCritic, RunningNorm, metric

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    env = PushEnv(args.envs, seed=11)
    ac = ActorCritic(env.obs_dim)
    norm = RunningNorm(env.obs_dim)
    opt = torch.optim.Adam(ac.actor.parameters(), lr=1e-3)
    data_o, data_a, log, steps, t0 = [], [], [], 0, time.time()

    for it in range(args.iters):
        beta = max(0.0, 1.0 - it / 5)          # probability of executing the expert's action
        env.reset(torch.ones(env.n, dtype=torch.bool))
        delivered = torch.zeros(env.n, dtype=torch.bool)
        with torch.no_grad():
            for _ in range(env.horizon - 1):
                obs = env.obs()
                if it == 0:
                    norm.update(obs)           # statistics frozen after the first sweep
                expert = push_expert(env, gain=15.0, push_speed=0.3)
                student = ac.actor(norm(obs)).clamp(-1, 1)
                use_expert = torch.rand(env.n, 1) < beta
                data_o.append(obs)
                data_a.append(expert)
                env.step(torch.where(use_expert, expert, student))
                delivered |= (env.goal - env.puck).norm(dim=-1) < 0.05
        steps += env.n * (env.horizon - 1)

        O, A = torch.cat(data_o), torch.cat(data_a)
        On = norm(O)
        for _ in range(args.epochs):
            perm = torch.randperm(len(O))
            for i in range(0, len(O), 4096):
                idx = perm[i:i + 4096]
                loss = (ac.actor(On[idx]) - A[idx]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        rate = delivered.float().mean().item()
        print(f"[dagger] iter {it + 1:2d}/{args.iters}  expert share {beta:.1f}  delivered {rate:.2f}  "
              f"imitation loss {loss.item():.4f}  data {len(O) / 1e6:.1f}M  {time.time() - t0:.0f}s", flush=True)
        log.append({"steps": steps, "reward": 0.0, "metric": rate, "std": 0.0, "sec": round(time.time() - t0, 1),
                    "phase": "imitation"})

    with torch.no_grad():
        ac.log_std.fill_(-1.6)                 # start PPO refinement with little exploration noise
    out = ROOT / "checkpoints" / "push.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"model": ac.state_dict(), "mean": norm.mean, "var": norm.var, "count": norm.count,
                "obs_dim": env.obs_dim, "task": "push", "log": log}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
