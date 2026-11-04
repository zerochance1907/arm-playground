import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  DT, HOME, PUCK_R, PUSHER_R, TABLE, BASE_R,
  armStep, fk, jointPositions, puckStep, pushObs, reachObs,
} from "./sim.js";
import { loadPolicy } from "./policy.js";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ------------------------------------------------------------------ state

const state = {
  task: "reach", brain: "policy", speed: 1,
  q: [...HOME], qd: [0, 0, 0, 0],
  target: [0.3, 0.15, 0.2],
  puck: [0.32, 0.12], puckV: [0, 0], goal: [0.28, -0.22],
  noise: [0, 0, 0, 0], last: null,
};
const policies = {};

// ------------------------------------------------------------------ scene

const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(css("--bg"));
scene.fog = new THREE.Fog(scene.background, 2.2, 4.5);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(0.95, -0.95, 0.75);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0.12, 0, 0.12);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 0.5;
controls.maxDistance = 3;

scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x1a1f26, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(0.8, -0.6, 1.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.01;
Object.assign(sun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1 });
scene.add(sun);

const table = new THREE.Mesh(
  new THREE.BoxGeometry(TABLE * 2, TABLE * 2, 0.02),
  new THREE.MeshStandardMaterial({ color: 0x161b22, roughness: 0.9 }),
);
table.position.z = -0.01;
table.receiveShadow = true;
scene.add(table);
const grid = new THREE.GridHelper(TABLE * 2, 26, 0x30363d, 0x21262d);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.001;
scene.add(grid);

const metal = new THREE.MeshStandardMaterial({ color: 0xc9d1d9, metalness: 0.6, roughness: 0.3 });
const jointMat = new THREE.MeshStandardMaterial({ color: 0x58a6ff, metalness: 0.3, roughness: 0.4 });
const base = new THREE.Mesh(new THREE.CylinderGeometry(BASE_R, BASE_R * 1.2, 0.12, 32), metal);
base.rotation.x = Math.PI / 2;
base.position.z = 0.06;
base.castShadow = true;
scene.add(base);

const links = [0.028, 0.022, 0.016].map((r) => {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 20), metal);
  m.castShadow = true;
  scene.add(m);
  return m;
});
const joints = [0.04, 0.034, 0.026].map((r) => {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), jointMat);
  m.castShadow = true;
  scene.add(m);
  return m;
});
const hand = new THREE.Mesh(new THREE.SphereGeometry(PUSHER_R, 24, 16),
  new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x58a6ff, emissiveIntensity: 0.35 }));
hand.castShadow = true;
scene.add(hand);

const glow = (color) => new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.8 });
const target = new THREE.Mesh(new THREE.SphereGeometry(0.022, 24, 16), glow(css("--accent")));
const targetShadow = new THREE.Mesh(new THREE.RingGeometry(0.018, 0.024, 32),
  new THREE.MeshBasicMaterial({ color: css("--accent"), transparent: true, opacity: 0.5 }));
targetShadow.position.z = 0.002;
const targetStem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineDashedMaterial({ color: css("--accent"), dashSize: 0.01, gapSize: 0.01 }));
scene.add(target, targetShadow, targetStem);

const puck = new THREE.Mesh(new THREE.CylinderGeometry(PUCK_R, PUCK_R, 0.03, 32),
  new THREE.MeshStandardMaterial({ color: css("--accent"), roughness: 0.5 }));
puck.rotation.x = Math.PI / 2;
puck.castShadow = true;
const goalMat = new THREE.MeshBasicMaterial({ color: css("--good"), transparent: true, opacity: 0.85 });
const goal = new THREE.Mesh(new THREE.RingGeometry(0.045, 0.06, 48), goalMat);
goal.position.z = 0.002;
const goalHit = new THREE.Mesh(new THREE.CircleGeometry(0.06, 32), new THREE.MeshBasicMaterial({ visible: false }));
scene.add(puck, goal, goalHit);

const TRAIL = 90;
const trailGeom = new THREE.BufferGeometry();
trailGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x58a6ff, transparent: true, opacity: 0.5 }));
scene.add(trail);
const trailPts = [];

function placeLink(mesh, a, b) {
  const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
  const dir = vb.clone().sub(va);
  mesh.scale.set(1, dir.length(), 1);
  mesh.position.copy(va).addScaledVector(dir, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
}

function render() {
  const pts = jointPositions(state.q);
  for (let i = 0; i < 3; i++) {
    placeLink(links[i], pts[i], pts[i + 1]);
    joints[i].position.set(...pts[i]);
  }
  hand.position.set(...pts[3]);
  trailPts.push(pts[3]);
  if (trailPts.length > TRAIL) trailPts.shift();
  const arr = trailGeom.attributes.position.array;
  for (let i = 0; i < TRAIL; i++) arr.set(trailPts[Math.min(i, trailPts.length - 1)], i * 3);
  trailGeom.attributes.position.needsUpdate = true;

  const reach = state.task === "reach";
  [target, targetShadow, targetStem].forEach((m) => (m.visible = reach));
  [puck, goal].forEach((m) => (m.visible = !reach));
  target.position.set(...state.target);
  targetShadow.position.set(state.target[0], state.target[1], 0.002);
  targetStem.geometry.setFromPoints([new THREE.Vector3(state.target[0], state.target[1], 0), target.position]);
  targetStem.computeLineDistances();
  puck.position.set(state.puck[0], state.puck[1], 0.015);
  goal.position.set(state.goal[0], state.goal[1], 0.002);
  goalHit.position.set(state.goal[0], state.goal[1], 0.003);
}

// ------------------------------------------------------------------ simulation

function randn() {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

function step() {
  const reach = state.task === "reach";
  const obs = reach ? reachObs(state.q, state.qd, state.target)
    : pushObs(state.q, state.qd, state.puck, state.puckV, state.goal);
  let action;
  const policy = policies[state.task];
  if (state.brain === "policy" && policy) {
    const t0 = performance.now();
    const out = policy.act(obs);
    state.inferMs = performance.now() - t0;
    action = out.action;
    state.last = out;
  } else {
    state.noise = state.noise.map((n) => 0.9 * n + 0.45 * randn());
    action = state.noise.map((n) => Math.max(-1, Math.min(1, n)));
    state.last = { action, hidden: null };
  }
  const eePrev = fk(state.q);
  [state.q, state.qd] = armStep(state.q, state.qd, action);
  if (!reach && drag !== puck) [state.puck, state.puckV] = puckStep(state.puck, state.puckV, eePrev, fk(state.q));
}

// ------------------------------------------------------------------ dragging

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let drag = null, dragPlane = null;

function rayFrom(e) {
  const r = canvas.getBoundingClientRect();
  pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
}

canvas.addEventListener("pointerdown", (e) => {
  rayFrom(e);
  const pickable = state.task === "reach" ? [target, targetShadow] : [puck, goalHit];
  const hit = raycaster.intersectObjects(pickable)[0];
  if (!hit) return;
  drag = hit.object === targetShadow ? target : hit.object === goalHit ? goal : hit.object;
  const h = drag === target ? state.target[2] : 0;
  dragPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -h);
  controls.enabled = false;
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener("pointermove", (e) => {
  if (!drag) {
    rayFrom(e);
    const over = raycaster.intersectObjects(state.task === "reach" ? [target, targetShadow] : [puck, goalHit]).length;
    canvas.style.cursor = over ? "grab" : "";
    return;
  }
  rayFrom(e);
  const p = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(dragPlane, p)) return;
  const x = Math.max(-TABLE, Math.min(TABLE, p.x)), y = Math.max(-TABLE, Math.min(TABLE, p.y));
  if (drag === target) state.target = [x, y, state.target[2]];
  else if (drag === goal) state.goal = [x, y];
  else { state.puck = [x, y]; state.puckV = [0, 0]; }
});

const endDrag = () => { drag = null; controls.enabled = true; };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

// ------------------------------------------------------------------ UI

function polar(rLo, rHi, ang) {
  const r = rLo + (rHi - rLo) * Math.random(), th = (Math.random() * 2 - 1) * ang;
  return [r * Math.cos(th), r * Math.sin(th)];
}

function scramble() {
  if (state.task === "reach") {
    state.target = [...polar(0.15, 0.5, 2.4), 0.05 + 0.35 * Math.random()];
    $("height").value = state.target[2];
  } else {
    state.puck = polar(0.25, 0.42, 1.4);
    state.puckV = [0, 0];
    do { state.goal = polar(0.2, 0.48, 1.4); }
    while (Math.hypot(state.goal[0] - state.puck[0], state.goal[1] - state.puck[1]) < 0.15);
  }
}

function segmented(id, key, onChange) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    state[key] = b.dataset[key];
    onChange?.();
  });
}

segmented("tasks", "task", () => {
  state.q = [...HOME]; state.qd = [0, 0, 0, 0];
  const reach = state.task === "reach";
  $("heightRow").style.display = reach ? "" : "none";
  $("metricLabel").textContent = reach ? "Hand to target" : "Puck to goal";
  $("hint").innerHTML = reach
    ? "Drag the <b>orange target</b>. The arm is driven by a neural network."
    : "Drag the <b>puck</b> or the <b style='color:var(--good)'>green goal</b>. The arm pushes it home.";
  trailPts.length = 0;
  showPolicyInfo();
});
segmented("brains", "brain");
$("shove").addEventListener("click", () => { state.qd = state.qd.map(() => (Math.random() * 2 - 1) * 4); });
$("scramble").addEventListener("click", scramble);
$("height").addEventListener("input", (e) => { state.target[2] = +e.target.value; });
$("speed").addEventListener("input", (e) => { state.speed = +e.target.value; });

function showPolicyInfo() {
  const p = policies[state.task];
  const svg = $("curve");
  if (!p) { $("trained").textContent = "–"; svg.innerHTML = ""; return; }
  const steps = p.meta.trained_steps;
  $("trained").textContent = `${(steps / 1e6).toFixed(0)}M simulated steps`;
  // Push starts with an imitation phase measured differently; plot only the RL phase.
  const log = p.meta.log.filter((r) => !r.phase);
  const ys = log.map((r) => r.metric), xs = log.map((r) => r.steps);
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pts = log.map((r, i) => `${(xs[i] / xs.at(-1)) * 288},${66 - ((ys[i] - y0) / (y1 - y0 || 1)) * 60}`).join(" ");
  svg.innerHTML = `<polyline points="${pts}" fill="none" stroke="${css("--accent")}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  $("curveLabel").textContent = state.task === "reach"
    ? "Training curve: hand-target distance ↓"
    : "PPO refinement: pucks on goal ↑";
}

const brain = $("brain").getContext("2d");
const JOINTS = ["yaw", "shoulder", "elbow", "wrist"];

function heat(v) {
  const t = Math.max(-1, Math.min(1, v));
  return t >= 0 ? `rgba(255,138,61,${0.12 + 0.88 * t})` : `rgba(88,166,255,${0.12 + 0.88 * -t})`;
}

function drawBrain() {
  const W = 288, cell = 9, cols = 16;
  brain.clearRect(0, 0, W, 150);
  const hidden = state.last?.hidden;
  for (let l = 0; l < 2; l++) {
    const ox = l * (cols * cell + 8);
    for (let i = 0; i < 128; i++) {
      brain.fillStyle = hidden ? heat(hidden[l][i]) : "#161b22";
      brain.fillRect(ox + (i % cols) * cell, Math.floor(i / cols) * cell, cell - 1, cell - 1);
    }
  }
  const action = state.last?.action ?? [0, 0, 0, 0];
  brain.font = "11px system-ui";
  JOINTS.forEach((name, i) => {
    const y = 80 + i * 17;
    brain.fillStyle = css("--muted");
    brain.fillText(name, 0, y + 10);
    brain.fillStyle = "#21262d";
    brain.fillRect(70, y, 210, 12);
    const a = action[i], mid = 175;
    brain.fillStyle = a >= 0 ? css("--accent") : css("--blue");
    brain.fillRect(a >= 0 ? mid : mid + a * 105, y, Math.abs(a) * 105, 12);
  });
}

function updateStats() {
  const ee = fk(state.q);
  let d, ok;
  if (state.task === "reach") {
    d = Math.hypot(...state.target.map((t, i) => t - ee[i]));
    ok = d < 0.03;
    $("status").textContent = ok ? "on target" : "reaching…";
  } else {
    d = Math.hypot(state.goal[0] - state.puck[0], state.goal[1] - state.puck[1]);
    ok = d < 0.05;
    $("status").textContent = ok ? "delivered!" : "pushing…";
    goalMat.opacity = ok ? 1 : 0.6;
  }
  $("status").style.color = ok ? css("--good") : css("--muted");
  $("metric").textContent = `${(d * 100).toFixed(1)} cm`;
  $("infer").textContent = state.brain === "policy" && state.inferMs != null
    ? `${(state.inferMs * 1000).toFixed(0)} µs` : "–";
}

// ------------------------------------------------------------------ loop

let framedPortrait = null;
function frameCamera(portrait, aspect) {
  // On a tall phone screen the bottom sheet covers the lower half: back off so the
  // whole table fits the narrow view, and aim low so the arm sits above the panel.
  const look = new THREE.Vector3(0.12, 0, portrait ? -0.2 : 0.12);
  const offset = new THREE.Vector3(0.95, -0.95, 0.75).multiplyScalar(portrait ? Math.min(2.1, 0.8 / aspect) : 1);
  controls.target.copy(look);
  camera.position.copy(look).add(offset);
  controls.maxDistance = portrait ? 5 : 3;
  scene.fog.near = offset.length() * 1.4;
  scene.fog.far = offset.length() * 3;
  framedPortrait = portrait;
}

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const portrait = w / h < 0.9;
  if (portrait !== framedPortrait) frameCamera(portrait, w / h);
}
addEventListener("resize", resize);
resize();
window.armPlayground = { state, camera, controls, scene, policies };   // handy in devtools

let acc = 0, prev = performance.now(), frame = 0;
function loop(now) {
  acc += Math.min(0.1, (now - prev) / 1000) * state.speed;
  prev = now;
  while (acc >= DT) { step(); acc -= DT; }
  render();
  controls.update();
  renderer.render(scene, camera);
  if (frame++ % 3 === 0) { drawBrain(); updateStats(); }
  requestAnimationFrame(loop);
}

Promise.allSettled(["reach", "push"].map(async (t) => { policies[t] = await loadPolicy(`policies/${t}.json`); }))
  .then((results) => {
    const failed = results.filter((r) => r.status === "rejected");
    if (failed.length) console.warn("policy load failed:", failed.map((f) => f.reason?.message));
    showPolicyInfo();
  });
requestAnimationFrame(loop);
