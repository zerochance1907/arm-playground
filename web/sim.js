// Line-for-line JavaScript port of arm/sim.py (single arm, float64).
// tests/parity.test.mjs checks it reproduces the Python trajectories exactly.

export const DT = 0.02;
export const H0 = 0.12;
export const LINKS = [0.30, 0.25, 0.12];
export const QMIN = [-2.8, -0.2, -2.6, -2.0];
export const QMAX = [2.8, 1.6, 0.3, 2.0];
export const VMAX = [2.0, 2.0, 2.5, 3.0];
export const ALPHA = DT / 0.08;
export const HOME = [0.0, 0.9, -1.6, -0.6];

export const PUCK_R = 0.035;
export const PUSHER_R = 0.025;
export const CONTACT_Z = 0.07;
export const BASE_R = 0.07;
export const TABLE = 0.65;
export const PUCK_DECAY = Math.exp(-3.0 * DT);

/** Positions of shoulder, elbow, wrist and hand, for rendering. Last one is the hand. */
export function jointPositions(q) {
  const [yaw, a1] = q;
  const a2 = a1 + q[2];
  const a3 = a2 + q[3];
  const [l1, l2, l3] = LINKS;
  const pts = [[0, H0]];
  let r = 0, z = H0;
  for (const [l, a] of [[l1, a1], [l2, a2], [l3, a3]]) {
    r += l * Math.cos(a);
    z += l * Math.sin(a);
    pts.push([r, z]);
  }
  return pts.map(([rr, zz]) => [rr * Math.cos(yaw), rr * Math.sin(yaw), zz]);
}

export function fk(q) {
  const yaw = q[0], a1 = q[1];
  const a2 = a1 + q[2];
  const a3 = a2 + q[3];
  const [l1, l2, l3] = LINKS;
  const r = l1 * Math.cos(a1) + l2 * Math.cos(a2) + l3 * Math.cos(a3);
  const z = H0 + l1 * Math.sin(a1) + l2 * Math.sin(a2) + l3 * Math.sin(a3);
  return [r * Math.cos(yaw), r * Math.sin(yaw), z];
}

const clamp = (x, lo, hi) => Math.max(Math.min(x, hi), lo);

export function armStep(q, qd, action) {
  const qn = new Array(4), qdn = new Array(4);
  for (let i = 0; i < 4; i++) {
    const a = clamp(action[i], -1, 1);
    const v = qd[i] + ALPHA * (a * VMAX[i] - qd[i]);
    const next = q[i] + v * DT;
    const c = clamp(next, QMIN[i], QMAX[i]);
    qn[i] = c;
    qdn[i] = c !== next ? 0 : v;
  }
  return [qn, qdn];
}

const norm2 = (x, y) => Math.max(Math.hypot(x, y), 1e-9);

export function puckStep(p, v, eePrev, ee) {
  let [px, py] = p, [vx, vy] = v;
  vx *= PUCK_DECAY; vy *= PUCK_DECAY;
  px += vx * DT; py += vy * DT;

  let d = norm2(px, py);
  let nx = px / d, ny = py / d;
  if (d < BASE_R + PUCK_R) {
    px = nx * (BASE_R + PUCK_R); py = ny * (BASE_R + PUCK_R);
    const vr = vx * nx + vy * ny;
    if (vr < 0) { vx -= vr * nx; vy -= vr * ny; }
  }

  const evx = (ee[0] - eePrev[0]) / DT, evy = (ee[1] - eePrev[1]) / DT;
  const ox = px - ee[0], oy = py - ee[1];
  d = norm2(ox, oy);
  nx = ox / d; ny = oy / d;
  if (ee[2] < CONTACT_Z && d < PUCK_R + PUSHER_R) {
    px = ee[0] + nx * (PUCK_R + PUSHER_R); py = ee[1] + ny * (PUCK_R + PUSHER_R);
    const vnHand = evx * nx + evy * ny, vnPuck = vx * nx + vy * ny;
    if (vnHand > vnPuck) { vx += (vnHand - vnPuck) * nx; vy += (vnHand - vnPuck) * ny; }
  }

  if (Math.abs(px) > TABLE) { px = clamp(px, -TABLE, TABLE); vx *= -0.5; }
  if (Math.abs(py) > TABLE) { py = clamp(py, -TABLE, TABLE); vy *= -0.5; }
  return [[px, py], [vx, vy]];
}

function jointObs(q, qd) {
  return [...q.map(Math.sin), ...q.map(Math.cos), ...qd.map((v, i) => v / VMAX[i])];
}

export function reachObs(q, qd, target) {
  const ee = fk(q);
  return [...jointObs(q, qd), ...ee, ...target, ...target.map((t, i) => t - ee[i])];
}

export function pushObs(q, qd, p, v, goal) {
  const ee = fk(q);
  return [...jointObs(q, qd), ...ee, ...p, ...v, ...goal,
    p[0] - ee[0], p[1] - ee[1], goal[0] - p[0], goal[1] - p[1]];
}
