"""A scripted pushing controller, used only to bootstrap the Push policy.

It hovers behind the puck (on the side away from the goal), descends, pushes with
a speed that shrinks as the puck nears the goal (the puck keeps sliding after
contact), and lifts away once the puck is delivered. Joint commands come from a
Jacobian-transpose step toward each waypoint, computed with autograd.
"""
import torch

from .sim import PUCK_R, PUSHER_R, fk

HOVER_Z, PUSH_Z = 0.10, 0.03


def _toward(q, waypoint, gain):
    q = q.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        err = ((waypoint - fk(q)) ** 2).sum()
        (grad,) = torch.autograd.grad(err, q)
    return (-grad * gain).clamp(-1, 1)


def push_expert(env, gain=15.0, push_speed=0.6):
    ee = fk(env.q)
    to_goal = env.goal - env.puck
    d2 = to_goal.norm(dim=-1, keepdim=True)
    u = to_goal / d2.clamp_min(1e-6)
    behind = env.puck - u * (PUCK_R + PUSHER_R + 0.02)
    xy_err = (behind - ee[:, :2]).norm(dim=-1, keepdim=True)
    z = lambda h: torch.full_like(d2, h)

    lined_up = xy_err < 0.025
    low = ee[:, 2:3] < 0.06
    delivered = d2 < 0.025
    # push through the puck by an amount that shrinks near the goal so it stops on it
    reach_in = (push_speed * d2).clamp(0.005, 0.05)
    pushing = torch.cat((env.puck + u * reach_in - u * (PUCK_R + PUSHER_R), z(PUSH_Z)), -1)
    descend = torch.cat((behind, z(PUSH_Z)), -1)
    hover = torch.cat((behind, z(HOVER_Z)), -1)
    retreat = torch.cat((env.puck - u * 0.12, z(HOVER_Z + 0.05)), -1)

    waypoint = torch.where(lined_up & low, pushing, torch.where(lined_up | (xy_err < 0.035), descend, hover))
    waypoint = torch.where(delivered, retreat, waypoint)
    return _toward(env.q, waypoint, gain)
