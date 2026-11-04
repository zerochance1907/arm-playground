// Runs an exported PyTorch policy (web/policies/*.json) in plain JavaScript:
// normalize -> Linear -> tanh -> Linear -> tanh -> Linear -> clip. No ML library.

export class Policy {
  constructor(json) {
    this.meta = json;
    this.mean = json.mean;
    this.std = json.std;
    this.layers = json.layers;
  }

  /** Returns the action and every hidden layer's activations (for the brain view). */
  act(obs) {
    let x = obs.map((o, i) => Math.max(-5, Math.min(5, (o - this.mean[i]) / this.std[i])));
    const hidden = [];
    this.layers.forEach((layer, li) => {
      const out = new Array(layer.b.length);
      for (let r = 0; r < out.length; r++) {
        const w = layer.w[r];
        let s = layer.b[r];
        for (let c = 0; c < x.length; c++) s += w[c] * x[c];
        out[r] = li < this.layers.length - 1 ? Math.tanh(s) : s;
      }
      if (li < this.layers.length - 1) hidden.push(out);
      x = out;
    });
    return { action: x.map((a) => Math.max(-1, Math.min(1, a))), hidden };
  }
}

export async function loadPolicy(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url}: ${res.status}`);
  return new Policy(await res.json());
}
