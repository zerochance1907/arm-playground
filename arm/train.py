"""PPO for the arm tasks: thousands of simulated arms stepped in parallel in PyTorch.

    python -m arm.train reach            # ~10M steps
    python -m arm.train push --steps 60e6
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .sim import ENVS, fk

ROOT = Path(__file__).resolve().parent.parent


class RunningNorm:
    """Running mean/variance of observations (exported with the policy)."""

    def __init__(self, dim):
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)
        self.count = 1e-4

    def update(self, x):
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        self.var = (self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total) / total
        self.count = total

    def __call__(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-5, 5)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=4, hidden=128):
        super().__init__()
        self.actor = mlp(obs_dim, hidden, act_dim)
        self.critic = mlp(obs_dim, 256, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())


def metric(env):
    """Task success measure for logging: reach = mean hand-target distance (m),
    push = fraction of pucks within 5 cm of their goal."""
    if hasattr(env, "puck"):
        return ((env.goal - env.puck).norm(dim=-1) < 0.05).float().mean().item()
    return (env.target - fk(env.q)).norm(dim=-1).mean().item()


def train(task, steps, n_envs, rollout, epochs, minibatches, lr, seed, out, ent_coef=0.0,
          init=None, critic_warmup=0):
    torch.manual_seed(seed)
    env = ENVS[task](n_envs, seed=seed)
    norm = RunningNorm(env.obs_dim)
    ac = ActorCritic(env.obs_dim)
    log, step_offset = [], 0
    if init:   # refine an existing policy (e.g. one bootstrapped by arm/imitate.py)
        ck = torch.load(init, weights_only=False)
        ac.load_state_dict(ck["model"])
        norm.mean, norm.var, norm.count = ck["mean"], ck["var"], ck.get("count", 1e6)
        log = list(ck["log"])
        step_offset = log[-1]["steps"] if log else 0
    opt = torch.optim.Adam(ac.parameters(), lr=lr, eps=1e-5)
    gamma, lam, clip = 0.99, 0.95, 0.2
    reward_scale = 0.1   # keeps value targets near unit scale

    obs = env.obs()
    updates = int(steps // (n_envs * rollout))
    t0 = time.time()
    for update in range(1, updates + 1):
        for g in opt.param_groups:
            g["lr"] = lr * (1 - (update - 1) / updates)
        buf_o, buf_a, buf_lp, buf_v, buf_r, buf_d = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(rollout):
                if not init:
                    norm.update(obs)   # a refined policy keeps the statistics it was trained with
                o = norm(obs)
                d = ac.dist(o)
                a = d.sample()
                buf_o.append(o); buf_a.append(a)
                buf_lp.append(d.log_prob(a).sum(-1)); buf_v.append(ac.critic(o).squeeze(-1))
                obs, r, done = env.step(a)
                buf_r.append(r * reward_scale); buf_d.append(done.float())
            last_v = ac.critic(norm(obs)).squeeze(-1)

            adv = torch.zeros(rollout, n_envs)
            gae = torch.zeros(n_envs)
            for t in reversed(range(rollout)):
                next_v = last_v if t == rollout - 1 else buf_v[t + 1]
                nonterm = 1.0 - buf_d[t]
                delta = buf_r[t] + gamma * next_v * nonterm - buf_v[t]
                gae = delta + gamma * lam * nonterm * gae
                adv[t] = gae
            values = torch.stack(buf_v)
            ret = adv + values

        O = torch.stack(buf_o).flatten(0, 1)
        A = torch.stack(buf_a).flatten(0, 1)
        LP = torch.stack(buf_lp).flatten(0, 1)
        ADV, RET = adv.flatten(), ret.flatten()
        batch = O.shape[0]
        mb = batch // minibatches
        for _ in range(epochs):
            perm = torch.randperm(batch)
            for i in range(minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                d = ac.dist(O[idx])
                ratio = (d.log_prob(A[idx]).sum(-1) - LP[idx]).exp()
                a_n = (ADV[idx] - ADV[idx].mean()) / (ADV[idx].std() + 1e-8)
                pg = -torch.min(ratio * a_n, ratio.clamp(1 - clip, 1 + clip) * a_n).mean()
                vloss = 0.5 * (ac.critic(O[idx]).squeeze(-1) - RET[idx]).pow(2).mean()
                actor_loss = pg - ent_coef * d.entropy().sum(-1).mean()
                loss = 0.5 * vloss + (0.0 if update <= critic_warmup else actor_loss)
                opt.zero_grad()
                loss.backward()
                # clip separately: a large early value error must not shrink the policy gradient
                nn.utils.clip_grad_norm_([*ac.actor.parameters(), ac.log_std], 0.5)
                nn.utils.clip_grad_norm_(ac.critic.parameters(), 0.5)
                opt.step()

        if update % 10 == 0 or update == updates:
            done_steps = step_offset + update * n_envs * rollout
            row = {"steps": done_steps, "reward": torch.stack(buf_r).mean().item() / reward_scale,
                   "metric": metric(env),
                   "std": ac.log_std.exp().mean().item(), "sec": round(time.time() - t0, 1)}
            log.append(row)
            print(f"[{task}] {done_steps / 1e6:6.1f}M steps  reward/step {row['reward']:+.3f}  "
                  f"{'success' if task == 'push' else 'dist'} {row['metric']:.3f}  "
                  f"std {row['std']:.2f}  {row['sec']:.0f}s", flush=True)
            torch.save({"model": ac.state_dict(), "mean": norm.mean, "var": norm.var,
                        "obs_dim": env.obs_dim, "task": task, "log": log}, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(ENVS))
    ap.add_argument("--steps", type=float, default=10e6)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ent", type=float, default=0.0, help="entropy bonus (keeps exploring)")
    ap.add_argument("--init", help="checkpoint to refine instead of starting from scratch")
    ap.add_argument("--critic-warmup", type=int, default=0, help="updates that train only the value net")
    args = ap.parse_args()
    out = ROOT / "checkpoints" / f"{args.task}.pt"
    out.parent.mkdir(exist_ok=True)
    train(args.task, args.steps, args.envs, args.rollout, args.epochs, args.minibatches,
          args.lr, args.seed, out, args.ent, args.init, args.critic_warmup)
    print(json.dumps({"saved": str(out)}))


if __name__ == "__main__":
    main()
