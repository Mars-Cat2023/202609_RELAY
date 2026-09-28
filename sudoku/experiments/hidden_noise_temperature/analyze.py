#!/usr/bin/env python
"""Audit completed rollout logs and report paired puzzle-bootstrap intervals."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Evaluation has not completed")
    summaries = json.loads((out / "summary.json").read_text())
    n = manifest["arguments"]["n"]
    start = manifest["arguments"]["start"]
    samples = manifest["arguments"]["samples"]
    if len(summaries) != len(manifest["configurations"]):
        raise ValueError("Missing configurations")
    puzzles = [json.loads(line) for line in (out / "puzzles.jsonl").read_text().splitlines()]
    truth = np.array([[int(c)+7 for c in p["answer"]] for p in puzzles])
    rng = np.random.default_rng(20260928)
    indices = rng.integers(0,n,size=(args.bootstrap_replicates,n),dtype=np.int32)
    intervals = []
    baseline = None
    for summary in summaries:
        success = np.zeros((n,samples),dtype=np.float64)
        nfe = np.zeros((n,samples),dtype=np.int32)
        seen = set()
        with (out / (summary["configuration"]+".jsonl")).open() as f:
            for line in f:
                r = json.loads(line)
                i,j = r["puzzle_id"]-start,r["sample_id"]
                if not 0 <= i < n or not 0 <= j < samples or (i,j) in seen:
                    raise ValueError("Duplicate/out-of-range sample")
                seen.add((i,j))
                correct = bool(np.array_equal(r["prediction_ids"],truth[i]))
                if correct != r["exact_match"] or not r["clues_preserved"]:
                    raise ValueError("Per-row metric audit failed")
                success[i,j] = correct
                nfe[i,j] = r["nfe"]
        assert len(seen) == n*samples
        average = success.mean(axis=1)
        passed = success.max(axis=1)
        assert np.isclose(100*average.mean(),summary["avg_at_8"])
        assert np.isclose(100*passed.mean(),summary["pass_at_8"])
        assert np.isclose(nfe.mean(),summary["mean_nfe"])
        if baseline is None:
            baseline = average
        record = {"configuration": summary["configuration"]}
        for name,values in [("avg_at_8",average),("pass_at_8",passed),
                            ("avg_delta_pp",average-baseline),("pass_delta_pp",passed-baseline)]:
            lo,hi = np.quantile(values[indices].mean(axis=1)*100,[0.025,0.975])
            record[name] = {"estimate": float(values.mean()*100),"ci95": [float(lo),float(hi)]}
        intervals.append(record)
    (out / "bootstrap.json").write_text(json.dumps({
        "method":"Paired percentile bootstrap resampling puzzles with all eight outcomes kept together; conditional on this checkpoint and observed rollouts; pointwise intervals, no multiple-comparison correction.",
        "seed":20260928,"replicates":args.bootstrap_replicates,"results":intervals},indent=2)+"\n")
    args_saved = manifest["arguments"]
    lines = ["# RELAY: confidence temperature versus hidden Gaussian noise", "",
        "## Protocol", "",
        f"- Checkpoint: `{args_saved['checkpoint']}`; {args_saved['weights']} weights; {manifest['model']['parameters']:,} parameters.",
        f"- Dataset: {n} unfiltered test puzzles, saved dataset indices {start}–{start+n-1}, no shuffle. All {samples} attempts were actually evaluated for every configuration.",
        f"- Precision: {args_saved['precision']}; batch size: {args_saved['batch_size']}; confidence threshold: {args_saved['threshold']}; max ordinary steps: {args_saved['max_steps']}.",
        "- Top-1 token selection and the original decoder are retained. Hidden noise affects the next forward, not the current logits. No training is performed.", "",
        "- T=0, when requested, means the exact softmax T->0+ limit, with uniform probability on tied maxima.",
        "## Results", "", (out / "summary.md").read_text().strip(), "",
        "NFE counts actual forwards per row, including work on already finished rows and the unconditional final forward. It depends on batch composition. `summary.csv` also reports first-filled steps. NFE / 8 counts the total budget across eight attempts.", "",
        "## Differences from baseline", "",
        "Pointwise 95% paired puzzle-bootstrap intervals (percentage points):", "",
        "| Configuration | avg@8 change [95% CI] | pass@8 change [95% CI] |", "|---|---:|---:|"]
    for r in intervals:
        a,p = r["avg_delta_pp"],r["pass_delta_pp"]
        lines.append(f"| {r['configuration']} | {a['estimate']:+.2f} [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}] | {p['estimate']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] |")
    lines += ["", "## Interpretation limits", "",
        "Fixed confidence temperature is deterministic: avg@8 equals pass@8. Gaussian noise can change the answer across attempts. Distinct outputs do not necessarily yield different rewards; mixed-reward group rates are in summary.csv.", "",
        "This is an exploratory sweep on one checkpoint, with all settings reported. The intervals condition on that checkpoint and these eight rollouts, and are not corrected for multiple comparisons. A best observed setting is not a validated optimum. Shared GPU wall times are not clean speed benchmarks.", "",
        "Task metrics alone do not establish distributional equivalence to temperature scaling. A fixed-state next-step probability matching experiment with held-out evaluation of residual KL and rank changes is still needed to quantify that approximation directly.", "",
        "## Audit", "", "Original baseline trajectory equivalence, repeat determinism, seeded noise reproducibility, first-step timing and clue preservation passed. All per-attempt exact-match labels and aggregate avg@8, pass@8 and NFE values were independently recomputed from saved predictions. See checks.json, manifest.json and bootstrap.json."]
    if all("first_step_filled_rate" in s for s in summaries):
        lines += ["", "## First-step completion", "",
            "This diagnostic measures where outgoing hidden noise can still affect future token decisions.", "",
            "| Configuration | Filled after first step (%) | Mean first-filled step | Mixed-reward groups (%) |",
            "|---|---:|---:|---:|"]
        for s in summaries:
            lines.append(f"| {s['configuration']} | {s['first_step_filled_rate']:.2f} | {s['mean_first_filled_step']:.3f} | {s['mixed_reward_group_rate']:.2f} |")
    (out / "report.md").write_text("\n".join(lines)+"\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        baseline_s = summaries[0]
        noise_temperature = manifest["arguments"].get("noise_temperature", 1.0)
        for ax,field,title in zip(axes,["temperature","sigma"],["No hidden noise",f"Hidden Gaussian noise (T={noise_temperature:g})"]):
            group = [s for s in summaries if (s["sigma"] == 0 if field == "temperature" else s["temperature"] == noise_temperature)]
            group.sort(key=lambda s:s[field])
            x = [s[field] for s in group]
            ax.plot(x,[s["avg_at_8"] for s in group],"o-",label="avg@8")
            ax.plot(x,[s["pass_at_8"] for s in group],"s--",label="pass@8")
            ax.axhline(baseline_s["avg_at_8"],color="gray",linewidth=1,alpha=.7,label="Baseline")
            ax.set(xlabel="Temperature T" if field == "temperature" else "Hidden noise sigma",ylabel="Exact-match success (%)",title=title,ylim=(0,100))
            if field == "sigma":
                ax.set_xscale("symlog",linthresh=0.05)
                ax.set_xlim(-0.003, max(x)*1.1)
                ax.set_xticks(x, [f"{v:g}" for v in x])
            ax.grid(alpha=.2); ax.legend()
        fig.savefig(out / "comparison.png",dpi=180)
        fig.savefig(out / "comparison.pdf")
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; numeric report complete without figure")
    print("AUDIT PASSED:",len(summaries),"configurations,",n*samples*len(summaries),"saved rollouts")
    print(out / "report.md")


if __name__ == "__main__":
    main()
