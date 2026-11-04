"""Build the static Hugging Face Space from the browser demo (stdlib only).

    python hf/build_space.py            # writes dist-space/ (gitignored)

Copies web/ (index.html, *.js, policies/*.json) plus the Space card hf/space/README.md
into dist-space/, then checks the result is servable from a Space root: every local
import / src / href / fetch path is relative and exists in the output, and every CDN
import is pinned to an exact version.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
CARD = ROOT / "hf" / "space" / "README.md"

# string literals used as module specifiers or URLs in the demo's HTML/JS
REFS = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(?\s*|\b(?:src|href)\s*=\s*|\bfetch\(\s*|loadPolicy\(\s*)["'`]([^"'`]+)["'`]"""
)
IMPORTMAP = re.compile(r'<script type="importmap">(.*?)</script>', re.S)
CDN_PINNED = re.compile(r"^https://(cdn\.jsdelivr\.net/npm|unpkg\.com)/[\w.@/-]+@\d+\.\d+\.\d+/")


def importmap(out):
    m = IMPORTMAP.search((out / "index.html").read_text(encoding="utf-8"))
    return json.loads(m.group(1))["imports"] if m else {}


def check(out):
    problems = []
    imports = importmap(out)
    mapped = lambda ref: ref in imports or any(k.endswith("/") and ref.startswith(k) for k in imports)
    for f in sorted(out.rglob("*")):
        if f.suffix not in (".html", ".js"):
            continue
        for ref in REFS.findall(f.read_text(encoding="utf-8")):
            if ref.startswith(("data:", "#", "mailto:", "https://")) or mapped(ref):
                continue
            if ref.startswith("http://"):
                problems.append(f"{f.name}: insecure URL {ref}")
            elif ref.startswith("/"):
                problems.append(f"{f.name}: root-absolute path {ref!r} breaks under a Space URL; make it relative")
            elif "${" not in ref and not (f.parent / ref).is_file():
                problems.append(f"{f.name}: {ref!r} is not in the Space output (bare imports need an import map entry)")
    for task in ("reach", "push"):  # main.js loads `policies/${t}.json`
        if not (out / "policies" / f"{task}.json").is_file():
            problems.append(f"policies/{task}.json missing")
    for spec, url in imports.items():
        if not CDN_PINNED.match(url):
            problems.append(f"import map entry {spec!r} -> {url} is not an https CDN URL pinned to an exact version")
    return problems


def build(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB, out, ignore=shutil.ignore_patterns(".*", "__pycache__"))
    shutil.copy2(CARD, out / "README.md")
    if not (out / "index.html").exists():
        sys.exit("build_space: web/index.html missing")
    problems = check(out)
    if problems:
        sys.exit("build_space: the Space would not work:\n  " + "\n  ".join(problems))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files)
    print(f"built {out} ({len(files)} files, {size / 1024:.0f} KB); paths are relative, CDN imports pinned")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist-space")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
