// Checks that the browser's sim + policy (web/sim.js, web/policy.js) reproduce the
// trajectory PyTorch produced (tests/fixtures/*_trace.json, from arm/export.py).
// Run: node --test tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { armStep, fk, puckStep, pushObs, reachObs } from "../web/sim.js";
import { Policy } from "../web/policy.js";

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const TOL = 1e-6;

function maxDiff(a, b) {
  return Math.max(...a.map((x, i) => Math.abs(x - b[i])));
}

for (const task of ["reach", "push"]) {
  test(`${task}: JS sim + policy match the PyTorch trace`, () => {
    const policy = new Policy(load(`../web/policies/${task}.json`));
    const { init, steps } = load(`./fixtures/${task}_trace.json`);
    let q = init.q, qd = init.qd, p = init.p, v = init.v;

    steps.forEach((row, t) => {
      const obs = task === "reach" ? reachObs(q, qd, init.target) : pushObs(q, qd, p, v, init.goal);
      const { action } = policy.act(obs);
      assert.ok(maxDiff(action, row.action) < TOL, `step ${t}: action ${action} vs ${row.action}`);
      const eePrev = fk(q);
      [q, qd] = armStep(q, qd, action);
      assert.ok(maxDiff(q, row.q) < TOL, `step ${t}: q drifted`);
      assert.ok(maxDiff(qd, row.qd) < TOL, `step ${t}: qd drifted`);
      if (task === "push") {
        [p, v] = puckStep(p, v, eePrev, fk(q));
        assert.ok(maxDiff(p, row.p) < TOL, `step ${t}: puck position drifted`);
        assert.ok(maxDiff(v, row.v) < TOL, `step ${t}: puck velocity drifted`);
      }
    });
  });
}
