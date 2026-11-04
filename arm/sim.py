"""Batched 4-DOF arm + tabletop puck simulator, written in PyTorch.

`web/sim.js` is a line-for-line JavaScript port used by the browser demo, and
`tests/parity.test.mjs` checks the two produce the same trajectories, so a
policy trained here behaves identically on the web page.

Coordinates: x/y on the table, z up, arm base at the origin. Joint angles q are
[yaw, shoulder, elbow, wrist]; pitch joints are measured from the horizontal and
chained (elbow is relative to shoulder, wrist relative to elbow).
"""
import math

import torch

DT = 0.02                     # 50 Hz control
H0 = 0.12                     # shoulder height above the table
LINKS = (0.30, 0.25, 0.12)    # upper arm, forearm, hand
QMIN = (-2.8, -0.2, -2.6, -2.0)
QMAX = (2.8, 1.6, 0.3, 2.0)
VMAX = (2.0, 2.0, 2.5, 3.0)   # rad/s at action = +-1
ALPHA = DT / 0.08             # first-order actuator lag, tau = 80 ms
HOME = (0.0, 0.9, -1.6, -0.6)

PUCK_R = 0.035
PUSHER_R = 0.025              # the hand acts as a round pusher when it is low
CONTACT_Z = 0.07              # hand must be below this height to touch the puck
BASE_R = 0.07                 # the puck cannot pass through the arm's base
TABLE = 0.65                  # table half-width
PUCK_DECAY = math.exp(-3.0 * DT)
FLOOR_Z = 0.015               # below this the hand is "in" the table (penalised)

REACH_OBS = 21
PUSH_OBS = 25


def _consts(ref):
    t = lambda xs: torch.tensor(xs, dtype=ref.dtype, device=ref.device)
    return t(QMIN), t(QMAX), t(VMAX)


def fk(q):
    """Hand position for joint angles q[..., 4] -> [..., 3]."""
    yaw, a1 = q[..., 0], q[..., 1]
    a2 = a1 + q[..., 2]
    a3 = a2 + q[..., 3]
    l1, l2, l3 = LINKS
    r = l1 * torch.cos(a1) + l2 * torch.cos(a2) + l3 * torch.cos(a3)
    z = H0 + l1 * torch.sin(a1) + l2 * torch.sin(a2) + l3 * torch.sin(a3)
    return torch.stack((r * torch.cos(yaw), r * torch.sin(yaw), z), dim=-1)


def arm_step(q, qd, action):
    """Velocity-controlled joints with actuator lag and hard joint limits."""
    qmin, qmax, vmax = _consts(q)
    a = action.clamp(-1.0, 1.0)
    qd = qd + ALPHA * (a * vmax - qd)
    q_next = q + qd * DT
    q_clamped = torch.maximum(torch.minimum(q_next, qmax), qmin)
    qd = torch.where(q_clamped != q_next, torch.zeros_like(qd), qd)
    return q_clamped, qd


def puck_step(p, v, ee_prev, ee):
    """Puck sliding with friction, pushed by the hand, blocked by base and walls."""
    v = v * PUCK_DECAY
    p = p + v * DT

    d = p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    nhat = p / d
    inside = d < BASE_R + PUCK_R
    p = torch.where(inside, nhat * (BASE_R + PUCK_R), p)
    vr = (v * nhat).sum(-1, keepdim=True)
    v = torch.where(inside & (vr < 0), v - vr * nhat, v)

    exy = ee[..., :2]
    evel = (ee[..., :2] - ee_prev[..., :2]) / DT
    n = p - exy
    d = n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    nhat = n / d
    hit = (ee[..., 2:3] < CONTACT_Z) & (d < PUCK_R + PUSHER_R)
    p = torch.where(hit, exy + nhat * (PUCK_R + PUSHER_R), p)
    vn_hand = (evel * nhat).sum(-1, keepdim=True)
    vn_puck = (v * nhat).sum(-1, keepdim=True)
    v = torch.where(hit & (vn_hand > vn_puck), v + (vn_hand - vn_puck) * nhat, v)

    over = p.abs() > TABLE
    p = p.clamp(-TABLE, TABLE)
    v = torch.where(over, -0.5 * v, v)
    return p, v


def joint_obs(q, qd):
    _, _, vmax = _consts(q)
    return torch.cat((torch.sin(q), torch.cos(q), qd / vmax), dim=-1)


def reach_obs(q, qd, target):
    ee = fk(q)
    return torch.cat((joint_obs(q, qd), ee, target, target - ee), dim=-1)


def push_obs(q, qd, p, v, goal):
    ee = fk(q)
    return torch.cat((joint_obs(q, qd), ee, p, v, goal, p - ee[..., :2], goal - p), dim=-1)


# ---------------------------------------------------------------- training envs

def _polar(n, r_lo, r_hi, ang, g, dtype, device):
    r = r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=dtype, device=device)
    th = (torch.rand(n, generator=g, dtype=dtype, device=device) * 2 - 1) * ang
    return torch.stack((r * torch.cos(th), r * torch.sin(th)), dim=-1)


class ArmEnv:
    """Common batched state: n arms, each with its own episode clock."""

    obs_dim = 0
    horizon = 200

    def __init__(self, n, seed=0, dtype=torch.float32, device="cpu"):
        self.n, self.dtype, self.device = n, dtype, device
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.q = torch.zeros(n, 4, dtype=dtype, device=device)
        self.qd = torch.zeros_like(self.q)
        self.t = torch.zeros(n, dtype=torch.long, device=device)
        self.reset(torch.ones(n, dtype=torch.bool, device=device))
        # stagger episode clocks so resets don't all happen on the same step
        self.t = torch.randint(0, self.horizon, (n,), generator=self.g, device=device)

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g, dtype=self.dtype, device=self.device)

    def reset_arm(self, m):
        k = int(m.sum())
        home = torch.tensor(HOME, dtype=self.dtype, device=self.device)
        qmin, qmax, _ = _consts(self.q)
        q = home + (self.rand(k, 4) * 2 - 1) * torch.tensor((1.5, 0.4, 0.6, 0.8), dtype=self.dtype, device=self.device)
        self.q[m] = torch.maximum(torch.minimum(q, qmax), qmin)
        self.qd[m] = 0
        self.t[m] = 0

    def step(self, action):
        self.ee_prev = fk(self.q)
        self.q, self.qd = arm_step(self.q, self.qd, action)
        reward = self.task_step(action)
        self.t += 1
        done = self.t >= self.horizon
        if done.any():
            self.reset(done)
        return self.obs(), reward, done


class ReachEnv(ArmEnv):
    """Move the hand to a target that occasionally jumps somewhere new."""

    obs_dim = REACH_OBS
    horizon = 200

    def sample_target(self, k):
        xy = _polar(k, 0.12, 0.55, 2.6, self.g, self.dtype, self.device)
        z = 0.03 + 0.42 * self.rand(k, 1)
        return torch.cat((xy, z), dim=-1)

    def reset(self, m):
        if not hasattr(self, "target"):
            self.target = torch.zeros(self.n, 3, dtype=self.dtype, device=self.device)
        self.reset_arm(m)
        self.target[m] = self.sample_target(int(m.sum()))
        self.prev_d = (self.target - fk(self.q)).norm(dim=-1)

    def obs(self):
        return reach_obs(self.q, self.qd, self.target)

    def task_step(self, action):
        ee = fk(self.q)
        d = (self.target - ee).norm(dim=-1)
        # progress term: an immediate, low-noise signal for "that action moved me closer"
        r = -d + 20.0 * (self.prev_d - d) + 0.5 * (d < 0.03) \
            - 0.005 * (action.clamp(-1, 1) ** 2).sum(-1) - 5.0 * (FLOOR_Z - ee[:, 2]).clamp_min(0)
        jump = self.rand(self.n) < 1.0 / 150   # learn to re-chase a moving target
        if jump.any():
            self.target[jump] = self.sample_target(int(jump.sum()))
        self.prev_d = (self.target - ee).norm(dim=-1)
        return r


class PushEnv(ArmEnv):
    """Push the puck onto a goal spot on the table."""

    obs_dim = PUSH_OBS
    horizon = 300

    def reset(self, m):
        if not hasattr(self, "puck"):
            z = lambda: torch.zeros(self.n, 2, dtype=self.dtype, device=self.device)
            self.puck, self.puck_v, self.goal = z(), z(), z()
        k = int(m.sum())
        self.reset_arm(m)
        puck = _polar(k, 0.22, 0.45, 1.6, self.g, self.dtype, self.device)
        self.puck[m] = puck
        self.puck_v[m] = 0
        # goal 10-30 cm from the puck in a random direction, kept off the base and in reach
        th = self.rand(k) * 2 * math.pi
        dist = 0.10 + 0.20 * self.rand(k)
        goal = puck + dist[:, None] * torch.stack((torch.cos(th), torch.sin(th)), dim=-1)
        r = goal.norm(dim=-1, keepdim=True)
        goal = goal / r * r.clamp(0.18, 0.52)
        self.goal[m] = goal
        self.prev_d2 = (self.goal - self.puck).norm(dim=-1)

    def obs(self):
        return push_obs(self.q, self.qd, self.puck, self.puck_v, self.goal)

    def task_step(self, action):
        ee = fk(self.q)
        self.puck, self.puck_v = puck_step(self.puck, self.puck_v, self.ee_prev, ee)
        # shaping: get the hand to the pushing spot just behind the puck (opposite the goal)
        to_goal = self.goal - self.puck
        u = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        behind = self.puck - u * (PUCK_R + PUSHER_R + 0.015)
        behind3 = torch.cat((behind, torch.full_like(behind[:, :1], PUCK_R)), dim=-1)
        d_hand = (behind3 - ee).norm(dim=-1)
        d2 = to_goal.norm(dim=-1)
        prev_dh = self.prev_dh if hasattr(self, "prev_dh") else d_hand
        r = -1.0 * d_hand - 1.0 * d2 + 20.0 * (prev_dh - d_hand) + 40.0 * (self.prev_d2 - d2) \
            + 1.0 * (d2 < 0.04) - 0.005 * (action.clamp(-1, 1) ** 2).sum(-1) \
            - 20.0 * (FLOOR_Z - ee[:, 2]).clamp_min(0)
        self.prev_d2 = d2
        self.prev_dh = d_hand
        return r

    def step(self, action):
        obs, r, done = super().step(action)
        if done.any() and hasattr(self, "prev_dh"):
            # a fresh episode must not be credited with "progress" from the old layout
            to_goal = self.goal - self.puck
            u = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            behind = self.puck - u * (PUCK_R + PUSHER_R + 0.015)
            behind3 = torch.cat((behind, torch.full_like(behind[:, :1], PUCK_R)), dim=-1)
            self.prev_dh = torch.where(done, (behind3 - fk(self.q)).norm(dim=-1), self.prev_dh)
        return obs, r, done


ENVS = {"reach": ReachEnv, "push": PushEnv}
