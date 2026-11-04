"""Publish the model repo and the static Space to the Hugging Face Hub.

    python hf/export_hub.py                       # (re)build + verify hf/model/
    python hf/publish.py --user <hf-username> --dry-run
    python hf/publish.py --user <hf-username>     # needs `hf auth login` or HF_TOKEN

Creates (or updates) the model repo <user>/arm-playground from hf/model/ and the
static Space <user>/arm-playground from the demo (built with hf/build_space.py).
`{{HF_USER}}` in the README cards is replaced with <user>. --dry-run validates the
cards and prints what would be uploaded, without contacting the Hub.
"""
import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_space  # noqa: E402

MODEL_DIR = HERE / "model"
MODEL_FILES = ["README.md", "config.json", "reach.safetensors", "reach.onnx", "push.safetensors", "push.onnx"]
REPO_NAME = "arm-playground"
PLACEHOLDER = "{{HF_USER}}"
# remote files of these kinds that are no longer in the upload are removed (keeps .gitattributes)
STALE = ["*.md", "*.json", "*.js", "*.html", "*.onnx", "*.safetensors", "policies/*"]


def stage(src_files, dst, user):
    """Copy files to a staging dir, filling in the username in Markdown cards."""
    for src, rel in src_files:
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".md":
            out.write_text(src.read_text(encoding="utf-8").replace(PLACEHOLDER, user), encoding="utf-8")
        else:
            shutil.copy2(src, out)
    return dst


def validate_card(path, repo_type):
    from huggingface_hub import ModelCard, SpaceCard
    from huggingface_hub.repocard_data import model_index_to_eval_results

    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER in text:
        sys.exit(f"{path}: unfilled {PLACEHOLDER} placeholder")
    card = (ModelCard if repo_type == "model" else SpaceCard).load(path)
    meta = card.data.to_dict()
    if repo_type == "model":
        for key in ("license", "library_name", "pipeline_tag", "tags", "model-index"):
            if key not in meta:
                sys.exit(f"{path}: model card metadata is missing {key!r}")
        name, results = model_index_to_eval_results(meta["model-index"])
        return f"{len(results)} eval results for {name}"
    for key in ("title", "emoji", "colorFrom", "colorTo", "sdk", "app_file", "license", "short_description"):
        if key not in meta:
            sys.exit(f"{path}: Space card metadata is missing {key!r}")
    if meta["sdk"] != "static":
        sys.exit(f"{path}: sdk must be static")
    if len(meta["short_description"]) > 60:
        sys.exit(f"{path}: short_description is over 60 characters")
    return f"static Space, app_file {meta['app_file']}"


def listing(folder):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    for p in files:
        print(f"    {p.relative_to(folder).as_posix():<28} {p.stat().st_size / 1024:8.1f} KB")
    print(f"    {len(files)} files, {sum(p.stat().st_size for p in files) / 1024:.0f} KB total")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--user", required=True, help="Hugging Face username or organization")
    ap.add_argument("--dry-run", action="store_true", help="validate and list files, upload nothing")
    ap.add_argument("--message", default="Sync from github.com/zerochance1907/arm-playground")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.user):
        sys.exit(f"--user {args.user!r} does not look like a Hugging Face username")

    missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).is_file()]
    if missing:
        sys.exit(f"hf/model is missing {missing}; run `python hf/export_hub.py` first")

    model_id = space_id = f"{args.user}/{REPO_NAME}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = stage([(MODEL_DIR / f, f) for f in MODEL_FILES], tmp / "model", args.user)
        built = build_space.build(tmp / "space-build")
        space = stage([(p, p.relative_to(built)) for p in built.rglob("*") if p.is_file()], tmp / "space", args.user)

        plans = [
            ("model", model_id, model, f"https://huggingface.co/{model_id}"),
            ("space", space_id, space, f"https://huggingface.co/spaces/{space_id}"),
        ]
        for repo_type, repo_id, folder, url in plans:
            info = validate_card(folder / "README.md", repo_type)
            print(f"{'[dry run] would upload' if args.dry_run else 'uploading'} {repo_type} {repo_id} ({info}):")
            listing(folder)

        if args.dry_run:
            print("dry run: nothing was uploaded")
            return

        from huggingface_hub import HfApi, get_token

        token = get_token()  # HF_TOKEN env var, or the token saved by `hf auth login`
        if not token:
            sys.exit("no Hugging Face token: run `hf auth login` or set HF_TOKEN")
        api = HfApi(token=token)
        for repo_type, repo_id, folder, url in plans:
            extra = {"space_sdk": "static"} if repo_type == "space" else {}
            api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, **extra)
            commit = api.upload_folder(
                repo_id=repo_id, repo_type=repo_type, folder_path=folder,
                commit_message=args.message, delete_patterns=STALE,
            )
            print(f"published {url} ({getattr(commit, 'commit_url', commit)})")


if __name__ == "__main__":
    main()
