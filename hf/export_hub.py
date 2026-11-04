"""Rebuild the exported policies as drop-in PyTorch / ONNX models for the Hugging Face Hub.

    pip install -r hf/requirements.txt
    python hf/export_hub.py              # writes hf/model/{reach,push}.{safetensors,onnx} + config.json

The trained .pt checkpoints are not in git, so the source of truth is the JSON the
browser runs (web/policies/<task>.json, written by arm/export.py). Each model takes the
RAW observation and does everything web/policy.js does:

    x = clip((obs - mean) / std, -5, 5) -> Linear -> tanh -> Linear -> tanh -> Linear -> clip(-1, 1)

Before writing anything the script checks, for both tasks, that the PyTorch module and
the ONNX graph (run with onnxruntime) reproduce
  1. the actions in the parity trace tests/fixtures/<task>_trace.json (the same trace
     tests/parity.test.mjs checks web/policy.js against), and
  2. a float64 NumPy port of web/policy.js on thousands of random observations,
to a max abs difference below 1e-5. Any mismatch exits non-zero.
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from arm import sim  # noqa: E402  (pure torch; used for obs layout, constants and trace replay)

TASKS = ("reach", "push")
TOL = 1e-5
OPSET = 17

JOINTS = ("base_yaw", "shoulder", "elbow", "wrist")
_JOINT_OBS = [f"sin_q_{j}" for j in JOINTS] + [f"cos_q_{j}" for j in JOINTS] + [f"qd_{j}_over_vmax" for j in JOINTS]
_xyz = lambda name: [f"{name}_{c}" for c in "xyz"]
_xy = lambda name: [f"{name}_{c}" for c in "xy"]
# Order matches arm/sim.py reach_obs / push_obs exactly.
OBS_NAMES = {
    "reach": _JOINT_OBS + _xyz("hand") + _xyz("target") + _xyz("target_minus_hand"),
    "push": _JOINT_OBS + _xyz("hand") + _xy("puck") + _xy("puck_vel") + _xy("goal")
    + _xy("puck_minus_hand") + _xy("goal_minus_puck"),
}
OBS_DIM = {"reach": sim.REACH_OBS, "push": sim.PUSH_OBS}
TASK_TEXT = {
    "reach": "Move the hand to a 3D target (and re-chase it when it moves).",
    "push": "Push a puck across the table onto a goal spot without overshooting.",
}


class ArmPolicy(torch.nn.Module):
    """Raw observation in, joint velocity command in [-1, 1] out. Same math as web/policy.js."""

    def __init__(self, obs_dim, hidden=(128, 128), action_dim=4):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        sizes = (obs_dim, *hidden, action_dim)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, obs):
        x = ((obs - self.obs_mean) / self.obs_std).clamp(-5.0, 5.0)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x.clamp(-1.0, 1.0)


def load_policy_json(task):
    path = ROOT / "web" / "policies" / f"{task}.json"
    return json.loads(path.read_text()), path


def build(policy):
    layers = policy["layers"]
    obs_dim = policy["obs_dim"]
    hidden = tuple(len(l["b"]) for l in layers[:-1])
    model = ArmPolicy(obs_dim, hidden, len(layers[-1]["b"]))
    with torch.no_grad():
        model.obs_mean.copy_(torch.tensor(policy["mean"]))
        model.obs_std.copy_(torch.tensor(policy["std"]))
        for lin, l in zip(model.layers, layers):
            lin.weight.copy_(torch.tensor(l["w"]))
            lin.bias.copy_(torch.tensor(l["b"]))
    return model.eval()


def numpy_policy_js(policy, obs):
    """Line-for-line float64 NumPy port of web/policy.js `act` (batched)."""
    mean, std = np.asarray(policy["mean"], np.float64), np.asarray(policy["std"], np.float64)
    x = np.clip((obs - mean) / std, -5, 5)
    layers = policy["layers"]
    for i, l in enumerate(layers):
        x = x @ np.asarray(l["w"], np.float64).T + np.asarray(l["b"], np.float64)
        if i < len(layers) - 1:
            x = np.tanh(x)
    return np.clip(x, -1, 1)


def trace_obs(task):
    """Observations along the parity trace, rebuilt from its recorded states, plus its actions."""
    trace = json.loads((ROOT / "tests" / "fixtures" / f"{task}_trace.json").read_text())
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    init = trace["init"]
    q, qd = t(init["q"]), t(init["qd"])
    p, v = (t(init["p"]), t(init["v"])) if task == "push" else (None, None)
    obs, actions = [], []
    for row in trace["steps"]:
        o = sim.reach_obs(q, qd, t(init["target"])) if task == "reach" else sim.push_obs(q, qd, p, v, t(init["goal"]))
        obs.append(o.numpy())
        actions.append(row["action"])
        q, qd = t(row["q"]), t(row["qd"])
        if task == "push":
            p, v = t(row["p"]), t(row["v"])
    return np.stack(obs), np.asarray(actions, np.float64)


def export_onnx(model, obs_dim, path):
    dummy = torch.zeros(1, obs_dim)
    kwargs = dict(
        input_names=["obs"], output_names=["action"], opset_version=OPSET,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:  # TorchScript exporter: needs only `onnx`, no onnxscript
            torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
        except TypeError:  # older torch without the `dynamo` argument
            torch.onnx.export(model, (dummy,), str(path), **kwargs)

    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    assert m.opset_import[0].version == OPSET, m.opset_import
    (inp,), (out,) = m.graph.input, m.graph.output
    dims = lambda v: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
    assert inp.name == "obs" and dims(inp) == ["batch", obs_dim], (inp.name, dims(inp))
    assert out.name == "action" and dims(out) == ["batch", 4], (out.name, dims(out))


def verify(task, policy, st_path, onnx_path):
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    obs_trace, act_trace = trace_obs(task)
    mean, std = np.asarray(policy["mean"]), np.asarray(policy["std"])
    # random observations around the training distribution, wide enough to hit the +-5 clip
    obs_rand = mean + std * rng.standard_normal((4096, policy["obs_dim"])) * 3.0
    obs_all = np.concatenate([obs_trace, obs_rand])

    ref = numpy_policy_js(policy, obs_all)
    # reloaded from disk, so we test the files that ship, not the in-memory module
    model = ArmPolicy(policy["obs_dim"], tuple(len(l["b"]) for l in policy["layers"][:-1]))
    model.load_state_dict(load_file(str(st_path)))
    model.eval()
    with torch.no_grad():
        out_torch = model(torch.tensor(obs_all, dtype=torch.float32)).numpy().astype(np.float64)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_onnx = sess.run(["action"], {"obs": obs_all.astype(np.float32)})[0].astype(np.float64)

    n = len(obs_trace)
    checks = {
        "numpy port vs parity trace": np.abs(ref[:n] - act_trace).max(),
        "torch vs parity trace": np.abs(out_torch[:n] - act_trace).max(),
        "onnx vs parity trace": np.abs(out_onnx[:n] - act_trace).max(),
        f"torch vs numpy port ({len(obs_all)} obs)": np.abs(out_torch - ref).max(),
        f"onnx vs numpy port ({len(obs_all)} obs)": np.abs(out_onnx - ref).max(),
        "torch vs onnx": np.abs(out_torch - out_onnx).max(),
    }
    ok = True
    for name, diff in checks.items():
        good = bool(diff < TOL)
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:<40} max abs diff {diff:.2e}")
    return ok


def task_config(task, policy, src):
    layers = policy["layers"]
    assert policy["obs_dim"] == OBS_DIM[task] == len(OBS_NAMES[task]), (task, policy["obs_dim"])
    return {
        "description": TASK_TEXT[task],
        "files": {"safetensors": f"{task}.safetensors", "onnx": f"{task}.onnx"},
        "obs_dim": policy["obs_dim"],
        "action_dim": len(layers[-1]["b"]),
        "hidden_sizes": [len(l["b"]) for l in layers[:-1]],
        "activation": "tanh",
        "obs_normalization": "clip((obs - obs_mean) / obs_std, -5, 5), inside the model",
        "output": "clip(linear_out, -1, 1), inside the model (deterministic policy mean)",
        "observation": OBS_NAMES[task],
        "action": [f"{j}_velocity_command" for j in JOINTS],
        "trained_env_steps": policy.get("trained_steps"),
        "source": {"file": f"web/policies/{task}.json", "sha256": hashlib.sha256(src.read_bytes()).hexdigest()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "hf" / "model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = {
        "name": "arm-playground",
        "framework": "pytorch",
        "repository": "https://github.com/zerochance1907/arm-playground",
        "control_period_ms": sim.DT * 1000,
        "control_hz": round(1 / sim.DT),
        "joints": list(JOINTS),
        "action_semantics": (
            "Each action in [-1, 1] is a target joint velocity, scaled by vmax_rad_s; the motor "
            "reaches it through a first-order lag (tau = 80 ms) and joints stop at their limits."
        ),
        "vmax_rad_s": list(sim.VMAX),
        "actuator_lag_tau_s": 0.08,
        "joint_limits_rad": {"min": list(sim.QMIN), "max": list(sim.QMAX)},
        "link_lengths_m": list(sim.LINKS),
        "shoulder_height_m": sim.H0,
        "onnx": {"opset": OPSET, "input": "obs [batch, obs_dim] float32", "output": "action [batch, 4] float32"},
        "tasks": {},
    }

    all_ok = True
    for task in TASKS:
        policy, src = load_policy_json(task)
        print(f"{task}: obs_dim {policy['obs_dim']}, layers {[ (len(l['w'][0]), len(l['b'])) for l in policy['layers']]}")
        model = build(policy)
        st_path, onnx_path = args.out / f"{task}.safetensors", args.out / f"{task}.onnx"
        tensors = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
        save_file(tensors, str(st_path), metadata={"task": task, "format": "pt", "source": f"web/policies/{task}.json"})
        export_onnx(model, policy["obs_dim"], onnx_path)
        all_ok &= verify(task, policy, st_path, onnx_path)
        config["tasks"][task] = task_config(task, policy, src)
        for p in (st_path, onnx_path):
            print(f"  wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p} ({p.stat().st_size / 1024:.0f} KB)")

    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if not all_ok:
        sys.exit(f"VERIFICATION FAILED: exported models do not match web/policy.js within {TOL:g}")
    print(f"verification passed (tolerance {TOL:g}); wrote {args.out / 'config.json'}")


if __name__ == "__main__":
    main()
