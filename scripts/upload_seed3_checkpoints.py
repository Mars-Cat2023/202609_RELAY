#!/usr/bin/env python3
"""Upload the eight formal seed-3, step-300000 checkpoints without conversion.

Default: local preview only (no network or huggingface_hub dependency).
Run with --upload after `hf auth login` to upload to the existing model repo.
Local checkpoints are only read. Remote files at matching paths are updated;
unrelated remote files and repository visibility are left unchanged.
"""

import argparse
from pathlib import Path


OBJECTIVES = ("mlm_uniform", "rollout", "relay_sg", "relay_bptt_steps2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Mars-Cat2023/RELAY_SUDOKU")
    parser.add_argument(
        "--logs-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "logs",
    )
    parser.add_argument("--upload", action="store_true", help="Execute upload; otherwise preview only.")
    args = parser.parse_args()
    root = args.logs_dir.resolve()
    relative_paths = [
        f"sudoku_extreme_{objective}_300k_{tying}_seed3/checkpoints/40-300000.ckpt"
        for objective in OBJECTIVES
        for tying in ("tied", "untied")
    ]
    missing = [name for name in relative_paths if not (root / name).is_file()]
    if missing:
        parser.error("Missing checkpoints (nothing uploaded):\n" + "\n".join(missing))
    sizes = {name: (root / name).stat().st_size for name in relative_paths}
    if any(size == 0 for size in sizes.values()):
        parser.error("Empty checkpoint found; nothing uploaded.")

    print(f"Destination: https://huggingface.co/{args.repo_id}")
    print("Selection: seed3 only; all 4 objectives x tied/untied; step 300000 only.")
    print("No smoke, best, last, intermediate checkpoints, or other logs.\n")
    for name, size in sizes.items():
        print(f"  {size / 1024**2:8.2f} MiB  {name}")
    print(f"\nTotal: {len(sizes)} files, {sum(sizes.values()) / 1024**3:.3f} GiB", flush=True)
    if not args.upload:
        print("Preview only: no network calls. Add --upload to execute.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        parser.error("Install huggingface_hub in your upload environment, then run hf auth login.")
    api = HfApi(token=True)
    # Require the user-created repo; do not create a potentially public repo implicitly.
    info = api.repo_info(repo_id=args.repo_id, repo_type="model")
    print(f"Existing repository visibility: {'PRIVATE' if info.private else 'PUBLIC'}", flush=True)
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(root),
        path_in_repo="",
        allow_patterns=relative_paths,
        commit_message="Upload all eight seed3 step-300000 Sudoku checkpoints",
    )
    print(f"Upload complete: {result.commit_url}")


if __name__ == "__main__":
    main()
