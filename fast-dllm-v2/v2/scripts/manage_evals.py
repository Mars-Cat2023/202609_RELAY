#!/usr/bin/env python3
"""Living experiment grid + orchestration for Fast-dLLM v2 lm_eval submissions.

Runs live in manage_evals_runs.RUNS (see manage_evals_runs.py). Submit via submit_eval.py (subprocess), with optional
Slurm slot chains (round-robin) and manager-side precheck (same results_*.json glob as
worker; optional slurm log + is_error for ERRORED vs RUNNING). RUNNING rows skip submission when precheck is on (SKIP_RUNNING) unless --force.

Usage (from Fast-dLLM/v2):

    python scripts/manage_evals.py --experiment math500_baselines_apr27 --list
    python scripts/manage_evals.py --experiment foo --num_slots 1 --max 3
    python scripts/manage_evals.py --experiment foo --dry_run  # classifies status + print sbatch, no submit
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from manage_evals_runs import RUNS

# ---------------------------------------------------------------------------
# Repo + submit_eval import (same directory)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import submit_eval as se  # noqa: E402

# Keys that must not appear in RUNS rows (manager / worker CLI only)
_RUN_DICT_EXCLUDED_DESTS = frozenset(
    {"dry_run", "print_command", "force", "sbatch_arg", "help"}
)

_STORE_TRUE_FLAGS = frozenset(
    {"use_carry", "use_block_cache", "no_wandb", "apply_chat_template"}
)


def _parser_run_dests() -> frozenset[str]:
    out: set[str] = set()
    for action in se.build_parser()._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help" or dest == "==SUPPRESS==":
            continue
        out.add(dest)
    return frozenset(out)


_RUN_ALLOWED_KEYS: frozenset[str] | None = None


def _allowed_run_keys() -> frozenset[str]:
    global _RUN_ALLOWED_KEYS
    if _RUN_ALLOWED_KEYS is None:
        _RUN_ALLOWED_KEYS = frozenset(_parser_run_dests() - _RUN_DICT_EXCLUDED_DESTS)
    return _RUN_ALLOWED_KEYS


def _validate_runs(runs: list[dict[str, Any]]) -> None:
    allowed = _allowed_run_keys()
    for i, run in enumerate(runs):
        if "experiment" not in run or not str(run["experiment"]).strip():
            raise ValueError(f"RUNS[{i}]: missing non-empty 'experiment'")
        if "task" not in run or not str(run["task"]).strip():
            raise ValueError(
                f"RUNS[{i}] experiment={run.get('experiment')!r}: missing 'task'"
            )
        if "model_path" not in run or not str(run["model_path"]).strip():
            raise ValueError(
                f"RUNS[{i}] experiment={run.get('experiment')!r}: missing 'model_path'"
            )
        for k in run:
            if k == "experiment":
                continue
            if k not in allowed:
                raise ValueError(
                    f"RUNS[{i}] experiment={run['experiment']!r}: unknown key {k!r} "
                    f"(allowed submit_eval dests minus {_RUN_DICT_EXCLUDED_DESTS})"
                )


_validate_runs(RUNS)


def load_runs() -> list[dict[str, Any]]:
    """Return RUNS after validation (re-validate if RUNS was monkeypatched)."""
    _validate_runs(RUNS)
    return RUNS


def filter_runs(
    runs: list[dict[str, Any]],
    *,
    experiments: list[str] | None,
    filters: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
    out = runs
    if experiments:
        ex_set = {e.strip() for e in experiments if e.strip()}
        out = [r for r in out if str(r.get("experiment", "")).strip() in ex_set]
    for key, want in filters:
        out = [r for r in out if _filter_match(r, key, want)]
    return out


def _parse_bool(s: str) -> bool:
    x = s.strip().lower()
    if x in {"1", "true", "yes", "y"}:
        return True
    if x in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"not a bool: {s!r}")


def _filter_match(run: dict[str, Any], key: str, want: Any) -> bool:
    if key not in run:
        return False
    got = run[key]
    if isinstance(want, bool):
        if isinstance(got, bool):
            return got == want
        return False
    if isinstance(got, bool):
        return False
    return got == want


def parse_filter_tokens(tokens: list[str]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for t in tokens:
        if "=" not in t:
            raise SystemExit(f"Invalid --filter token (need k=v): {t!r}")
        k, v = t.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"Invalid --filter token: {t!r}")
        vv: Any = v
        if v.strip().lower() in {"true", "false", "0", "1"}:
            vv = _parse_bool(v)
        out.append((k, vv))
    return out


def run_dict_to_parse_argv(run: dict[str, Any]) -> list[str]:
    """Build argv for submit_eval.parse_args from one RUNS row (no manager flags)."""
    argv: list[str] = []
    # Stable order: task, model_path first, then rest sorted
    keys = [k for k in run if k != "experiment"]
    rest = sorted(k for k in keys if k not in {"task", "model_path"})
    ordered = []
    if "task" in run:
        ordered.append("task")
    if "model_path" in run:
        ordered.append("model_path")
    ordered.extend(rest)

    for key in ordered:
        if key == "experiment":
            continue
        val = run[key]
        # submit_eval.registered flags use underscores (e.g. --model_path), not GNU-style hyphens.
        flag = f"--{key}"

        if key == "apply_chat_template":
            if val is False:
                argv.append("--no_apply_chat_template")
            elif val is True:
                argv.append("--apply_chat_template")
            else:
                raise ValueError(f"apply_chat_template must be bool, got {val!r}")
            continue

        if isinstance(val, bool):
            if key in _STORE_TRUE_FLAGS:
                if val is True:
                    argv.append(flag)
                # False: omit
            else:
                raise ValueError(f"unexpected bool for key {key!r}")
            continue

        if val is None:
            continue

        if isinstance(val, Path):
            sval = str(val)
        else:
            sval = str(val)
        argv.extend([flag, sval])
    return argv


def run_to_argv(
    run: dict[str, Any],
    submit_eval_path: Path,
    *,
    extra: list[str] | None = None,
) -> list[str]:
    """Full argv: python submit_eval.py ..."""
    base = [sys.executable, str(submit_eval_path)]
    base.extend(run_dict_to_parse_argv(run))
    if extra:
        base.extend(extra)
    return base


def format_sbatch_dependency(raw: str, kind: str) -> str:
    """If raw has no ':', prefix kind (e.g. afterany:12345)."""
    raw = raw.strip()
    if not raw:
        return ""
    if ":" in raw:
        return raw
    return f"{kind}:{raw}"


def precheck_existing_results(run: dict[str, Any], repo_root: Path) -> Path | None:
    argv = run_dict_to_parse_argv(run)
    ns = se.parse_args(argv)
    cfg = se.make_cfg(ns, repo_root)
    return se.results_artifact_path(cfg)


# Substrings that indicate the eval job failed (Python trace, shell / accelerate wrapper).
_SLURM_LOG_ERROR_MARKERS = (
    "Traceback (most recent call last):",
    "returned non-zero exit status",
)


def is_error(log_file: Path) -> bool:
    """Return True if the slurm log contains known failure signatures."""
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if any(m in line for m in _SLURM_LOG_ERROR_MARKERS):
                    return True
    except OSError:
        return False
    return False


def _latest_slurm_log(cfg: se.SubmitEvalConfig) -> Path | None:
    """Newest slurm-*.out under the run's output root (avoids path mismatch vs lm-eval subdir)."""
    logs = list(cfg.output_dir.rglob("slurm-*.out"))
    if not logs:
        return None
    try:
        return max(logs, key=lambda p: p.stat().st_mtime)
    except OSError:
        return sorted(logs)[-1]


def classify_status(
    cfg: se.SubmitEvalConfig,
) -> tuple[str, Path | None]:
    """Classify a run: COMPLETED, ERRORED, RUNNING, or NOT_SUBMITTED (artifact: results or log)."""
    results_json = se.results_artifact_path(cfg)
    if results_json is not None:
        return "COMPLETED", results_json
    latest = _latest_slurm_log(cfg)
    if latest is None:
        return "NOT_SUBMITTED", None
    if is_error(latest):
        return "ERRORED", latest
    return "RUNNING", latest


def _try_precheck_or_cfg(
    run: dict[str, Any], repo_root: Path
) -> tuple[se.SubmitEvalConfig | None, Path | None, str | None]:
    """Returns (cfg, existing_results, error_message). cfg is None on failure."""
    try:
        argv = run_dict_to_parse_argv(run)
        ns = se.parse_args(argv)
        cfg = se.make_cfg(ns, repo_root)
    except SystemExit as e:
        msg = str(e) if str(e) else "SystemExit from submit_eval"
        return None, None, msg
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    existing = se.results_artifact_path(cfg)
    return cfg, existing, None


def parse_submit_output(stdout: str) -> dict[str, str]:
    """Parse last SUBMITTED / SKIP / DRY_RUN status line from submit_eval stdout."""
    status_re = re.compile(r"^(SUBMITTED|SKIP|DRY_RUN)\s+(.+)$")
    last: dict[str, str] = {}
    for line in (stdout or "").splitlines():
        m = status_re.match(line.strip())
        if not m:
            continue
        kind, rest = m.group(1), m.group(2)
        last["status"] = kind
        for part in rest.split():
            if "=" in part:
                k, v = part.split("=", 1)
                last[k] = v
    return last


def submit_one(
    run: dict[str, Any],
    *,
    repo_root: Path,
    submit_eval_path: Path,
    prev_dep: str,
    chain_mode: str,
    dry_run: bool,
    force: bool,
    print_command: bool,
    precheck: bool,
    sbatch_args: list[str],
) -> dict[str, Any]:
    exp = str(run.get("experiment", ""))
    extra: list[str] = []
    if dry_run:
        extra.append("--dry_run")
    if force:
        extra.append("--force")
    if print_command:
        extra.append("--print_command")

    for s in sbatch_args:
        extra.append(f"--sbatch_arg={s}")

    dep_injected = ""
    if chain_mode == "serial" and prev_dep:
        dep_injected = f"--dependency={prev_dep}"
        extra.append(f"--sbatch_arg={dep_injected}")

    argv = run_to_argv(run, submit_eval_path, extra=extra)

    record: dict[str, Any] = {
        "experiment": exp,
        "argv": argv,
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "status": "",
        "job_id": "",
        "sbatch_path": "",
        "run_tag": "",
        "task": "",
        "model_path": "",
        "existing_results": "",
        "dep": prev_dep if chain_mode == "serial" and prev_dep else "",
        "invoked_subprocess": False,
        "run_status": "",
        "status_artifact": "",
    }

    # Disk status (same four states as --list): always when precheck or dry_run.
    if precheck or dry_run:
        cfg, _existing, err = _try_precheck_or_cfg(run, repo_root)
        if err:
            record["status"] = "ERROR"
            record["stderr"] = err
            record["returncode"] = 1
            return record
        assert cfg is not None
        record["run_tag"] = cfg.run_tag
        record["task"] = cfg.task
        record["model_path"] = str(run.get("model_path", ""))
        st_pre, artifact = classify_status(cfg)
        record["run_status"] = st_pre
        record["status_artifact"] = str(artifact) if artifact else ""
        if precheck and not force and st_pre == "COMPLETED":
            record["status"] = "SKIP_COMPLETED"
            record["existing_results"] = str(artifact) if artifact else ""
            record["returncode"] = 0
            return record
        if precheck and not force and st_pre == "RUNNING":
            record["status"] = "SKIP_RUNNING"
            record["existing_results"] = str(artifact) if artifact else ""
            record["returncode"] = 0
            return record

    record["invoked_subprocess"] = True
    try:
        r = subprocess.run(
            argv,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        record["status"] = "ERROR"
        record["stderr"] = str(e)
        record["returncode"] = 1
        return record

    record["returncode"] = r.returncode
    record["stdout"] = r.stdout or ""
    record["stderr"] = r.stderr or ""

    if r.returncode != 0:
        record["status"] = "ERROR"
        return record

    if print_command:
        record["status"] = "PRINT_COMMAND"
        return record

    parsed = parse_submit_output(record["stdout"])
    st = parsed.get("status", "")
    if st in {"SUBMITTED", "SKIP", "DRY_RUN"}:
        record["status"] = st
    else:
        record["status"] = "ERROR"
        if not record["stderr"]:
            record["stderr"] = "no SUBMITTED/SKIP/DRY_RUN line in submit_eval stdout"

    record["job_id"] = parsed.get("job_id", "")
    record["sbatch_path"] = parsed.get("sbatch", "")
    record["run_tag"] = parsed.get("run_tag", record["run_tag"])
    record["task"] = parsed.get("task", record["task"])
    if not record["model_path"]:
        record["model_path"] = str(run.get("model_path", ""))
    if st == "SKIP":
        record["existing_results"] = parsed.get("existing", "")

    return record


def _write_runlog(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "idx",
        "experiment",
        "status",
        "run_status",
        "job_id",
        "slot",
        "dep",
        "sbatch_path",
        "run_tag",
        "task",
        "model_path",
        "returncode",
        "existing_results",
        "status_artifact",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows):
            out = {k: row.get(k, "") for k in fieldnames}
            out["idx"] = str(i)
            w.writerow(out)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    cols = [
        "idx",
        "experiment",
        "status",
        "run_status",
        "job_id",
        "slot",
        "dep",
        "run_tag",
        "returncode",
        "existing_results",
    ]
    widths = {c: len(c) for c in cols}
    for i, row in enumerate(rows):
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    def fmt_row(d: dict[str, str]) -> str:
        return "  ".join(str(d.get(c, "")).ljust(widths[c]) for c in cols)

    header = {c: c for c in cols}
    print(fmt_row(header))
    print(fmt_row({c: "-" * widths[c] for c in cols}))
    for i, row in enumerate(rows):
        d = {c: str(row.get(c, "")) for c in cols}
        d["idx"] = str(i)
        print(fmt_row(d))

    n = len(rows)
    if n == 0:
        print("\nSummary: 0 rows")
        return

    def _count_key(key: str) -> Counter[str]:
        c: Counter[str] = Counter()
        for row in rows:
            v = row.get(key, "")
            c[str(v) if v is not None else ""] += 1
        return c

    def _fmt_counter(c: Counter[str]) -> str:
        parts: list[str] = []
        for label, k in sorted(
            ((lbl if lbl else "(empty)", lbl) for lbl in c.keys()),
            key=lambda t: (-c[t[1]], t[0]),
        ):
            parts.append(f"{label}={c[k]}")
        return ", ".join(parts)

    st_c = _count_key("status")
    rs_c = _count_key("run_status")
    completed_disk = rs_c.get("COMPLETED", 0)
    not_submitted = rs_c.get("NOT_SUBMITTED", 0)
    running = rs_c.get("RUNNING", 0)
    errored_disk = rs_c.get("ERRORED", 0)

    print("", flush=True)
    print(f"Summary: {n} row(s)", flush=True)
    print(f"  status:      {_fmt_counter(st_c)}", flush=True)
    print(f"  run_status:  {_fmt_counter(rs_c)}", flush=True)
    print(
        "  On disk: "
        f"COMPLETED={completed_disk} NOT_SUBMITTED={not_submitted} "
        f"RUNNING={running} ERRORED={errored_disk}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Filter by experiment name (repeatable)",
    )
    ap.add_argument(
        "--filter",
        nargs="*",
        default=[],
        metavar="K=V",
        help="Extra equality filters on RUNS dict keys",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print COMPLETED/ERRORED/RUNNING/NOT_SUBMITTED and resolved argv; do not submit",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Run submit_eval with --dry_run; still classifies disk status (COMPLETED/…)",
    )
    ap.add_argument("--print_command", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--num_slots",
        type=int,
        default=0,
        help=(
            "Parallel chain budget: 0 = no chaining; 1 = one serial chain; "
            "N>1 = round-robin into N chains (each chain uses --dependency after the prior job in that slot)"
        ),
    )
    ap.add_argument(
        "--chain_dep_kind",
        default="afterany",
        help="Dependency kind when only a bare job id is given (default: afterany)",
    )
    ap.add_argument(
        "--chain_first_dep",
        default="",
        help="Initial Slurm dependency for the first job in each slot (id or afterok:123)",
    )
    ap.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep submitting after ERROR (default: stop on first ERROR)",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max submit_eval subprocess invocations across all slots (SKIP_COMPLETED / SKIP_RUNNING precheck skips do not count)",
    )
    ap.add_argument(
        "--submit_eval_path",
        type=Path,
        default=_SCRIPTS_DIR / "submit_eval.py",
        help="Path to submit_eval.py",
    )
    ap.add_argument(
        "--sbatch_arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "Forwarded to submit_eval --sbatch_arg (repeatable); "
            "verbatim sbatch CLI flag, e.g. --sbatch_arg=--partition=gpu"
        ),
    )
    ap.add_argument(
        "--no_precheck",
        action="store_true",
        help="Disable manager-side results_*.json precheck",
    )
    args = ap.parse_args()

    if not REPO_ROOT.joinpath("eval.py").is_file():
        print(
            f"ERROR: {REPO_ROOT} does not look like Fast-dLLM/v2 (eval.py missing).",
            file=sys.stderr,
        )
        sys.exit(2)

    abort_on_error = not args.continue_on_error

    filters = parse_filter_tokens(args.filter)
    runs = filter_runs(
        load_runs(),
        experiments=args.experiment or None,
        filters=filters,
    )

    # Plan: --force implies --no_precheck for manager; worker still gets --force.
    precheck = not args.no_precheck and not args.force

    submit_eval_path = args.submit_eval_path.resolve()

    list_sbatch_extra: list[str] = [
        f"--sbatch_arg={s}" for s in (args.sbatch_arg or [])
    ]

    if args.list:
        for i, run in enumerate(runs):
            exp = run.get("experiment", "")
            argv = run_to_argv(
                run, submit_eval_path, extra=list_sbatch_extra or None
            )
            try:
                argv_cfg = run_dict_to_parse_argv(run)
                ns = se.parse_args(argv_cfg)
                cfg = se.make_cfg(ns, REPO_ROOT)
            except SystemExit as e:
                print(f"[{i}] experiment={exp} LIST_ERROR err={e}", flush=True)
                print(f"    {shlex.join(argv)}", flush=True)
                continue
            except Exception as e:
                print(
                    f"[{i}] experiment={exp} LIST_ERROR err={type(e).__name__}: {e}",
                    flush=True,
                )
                print(f"    {shlex.join(argv)}", flush=True)
                continue
            st, artifact = classify_status(cfg)
            art_s = str(artifact) if artifact else ""
            print(
                f"[{i}] {st} experiment={exp} run_tag={cfg.run_tag} "
                f"expected_results={cfg.expected_results_path} artifact={art_s}",
                flush=True,
            )
            print(f"    {shlex.join(argv)}", flush=True)
        return

    num_slots = max(0, args.num_slots)
    first_dep = format_sbatch_dependency(args.chain_first_dep, args.chain_dep_kind)
    prev_deps: list[str] = [first_dep for _ in range(num_slots)]
    chain_mode = "serial" if num_slots > 0 else "none"
    spawned = 0
    records: list[dict[str, Any]] = []

    for i, run in enumerate(runs):
        if args.max is not None and spawned >= args.max:
            print(
                f"[{i}] STOP: reached --max {args.max} subprocess invocations",
                flush=True,
            )
            break

        slot: int | None = (i % num_slots) if num_slots > 0 else None
        dep = prev_deps[slot] if slot is not None else ""

        rec = submit_one(
            run,
            repo_root=REPO_ROOT,
            submit_eval_path=submit_eval_path,
            prev_dep=dep,
            chain_mode=chain_mode,
            dry_run=args.dry_run,
            force=args.force,
            print_command=args.print_command,
            precheck=precheck,
            sbatch_args=list(args.sbatch_arg or []),
        )
        rec["idx"] = i
        rec["slot"] = "" if slot is None else str(slot)
        records.append(rec)

        if rec.get("invoked_subprocess"):
            spawned += 1

        st = rec["status"]
        dep_show = rec.get("dep", "") if chain_mode == "serial" else ""
        slot_s = rec.get("slot", "")
        rs = rec.get("run_status", "")
        rs_s = f" run_status={rs}" if rs else ""
        print(
            f"[{i}] {st} experiment={rec['experiment']} job_id={rec['job_id']}{rs_s} "
            f"slot={slot_s} dep={dep_show}",
            flush=True,
        )
        if rec["stderr"] and st == "ERROR":
            print(rec["stderr"], file=sys.stderr, flush=True)

        if st == "SUBMITTED" and rec.get("job_id") and slot is not None:
            prev_deps[slot] = format_sbatch_dependency(
                rec["job_id"],
                args.chain_dep_kind,
            )

        if st == "ERROR" and abort_on_error:
            break

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = REPO_ROOT / "logs" / "manage_evals" / f"{ts}.runlog.tsv"
    _write_runlog(log_path, records)
    print(f"Wrote {log_path}", flush=True)
    print("", flush=True)
    _print_summary(records)

    if any(r.get("status") == "ERROR" for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
