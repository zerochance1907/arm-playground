"""Export a trained policy to plain JSON for the browser, plus a parity trace.

    python -m arm.export reach push

The trace is generated in float64 from the *exported JSON* (not the checkpoint),
so tests/parity.test.mjs can check web/sim.js + web/policy.js reproduce it.
"""
import json
import sys
from pathlib import Path

import torch

from .sim import HOME, arm_step, fk, puck_step, push_obs, reach_obs
from .train import ActorCritic

ROOT = Path(__file__).resolve().parent.parent
R = lambda x: [round(float(v), 6) for v in x]


def export(task):
    ck = torch.load(ROOT / "checkpoints" / f"{task}.pt", weights_only=False)
    ac = ActorCritic(ck["obs_dim"])
    ac.load_state_dict(ck["model"])
    linears = [m for m in ac.actor if isinstance(m, torch.nn.Linear)]
    policy = {
        "task": task,
        "obs_dim": ck["obs_dim"],
        "trained_steps": ck["log"][-1]["steps"],
        "mean": R(ck["mean"]),
        "std": R(torch.sqrt(ck["var"] + 1e-8)),
        "layers": [{"w": [R(row) for row in m.weight.detach()], "b": R(m.bias.detach())} for m in linears],
        "log": ck["log"],
    }
    out = ROOT / "web" / "policies" / f"{task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    write_trace(task, policy)


def act(policy, obs):
    """Deterministic policy forward pass, exactly as web/policy.js does it."""
    d = torch.float64
    x = ((obs - torch.tensor(policy["mean"], dtype=d)) / torch.tensor(policy["std"], dtype=d)).clamp(-5, 5)
    layers = policy["layers"]
    for i, layer in enumerate(layers):
        x = torch.tensor(layer["w"], dtype=d) @ x + torch.tensor(layer["b"], dtype=d)
        if i < len(layers) - 1:
            x = torch.tanh(x)
    return x.clamp(-1, 1)


def write_trace(task, policy, steps=200):
    d = torch.float64
    q = torch.tensor(HOME, dtype=d) + torch.tensor((0.4, -0.1, 0.2, 0.1), dtype=d)
    qd = torch.zeros(4, dtype=d)
    init = {"q": R(q), "qd": R(qd)}
    q = torch.tensor(init["q"], dtype=d)
    if task == "reach":
        target = torch.tensor((0.25, -0.30, 0.20), dtype=d)
        init["target"] = R(target)
    else:
        p, v, goal = (torch.tensor(x, dtype=d) for x in ((0.32, 0.10), (0.0, 0.0), (0.30, -0.22)))
        init.update(p=R(p), v=R(v), goal=R(goal))
    rows = []
    for _ in range(steps):
        obs = reach_obs(q, qd, target) if task == "reach" else push_obs(q, qd, p, v, goal)
        a = act(policy, obs)
        ee_prev = fk(q)
        q, qd = arm_step(q, qd, a)
        row = {"action": a.tolist(), "q": q.tolist(), "qd": qd.tolist()}
        if task == "push":
            p, v = puck_step(p, v, ee_prev, fk(q))
            row.update(p=p.tolist(), v=v.tolist())
        rows.append(row)
    out = ROOT / "tests" / "fixtures" / f"{task}_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": task, "init": init, "steps": rows}))
    ee = fk(q)
    final = (target - ee).norm().item() if task == "reach" else (goal - p).norm().item()
    print(f"wrote {out}; final {'hand-target' if task == 'reach' else 'puck-goal'} distance {final:.3f} m")


if __name__ == "__main__":
    for t in sys.argv[1:] or ["reach", "push"]:
        export(t)
