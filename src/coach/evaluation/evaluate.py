"""Run the coach over labelled videos and report against the prototype targets.

Manifest (JSON list): id, video, volunteer, split (dev|test), exercise, lighting, camera, labels.
Labels JSON: manual_reps, pt_feedback {attempt: [codes]}, optional ref_angles [[t, deg]],
             optional pt_times {"attempt:code": t_s}.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..exercises import REGISTRY
from ..run import load_overrides, process, write_outputs
from ..tracker import PoseTracker
from .metrics import (TARGETS, angle_errors, feedback_counts, feedback_delays,
                      precision_recall)


def evaluate_entry(entry, root, out_root):
    labels = json.loads((root / entry["labels"]).read_text())
    rows, events, summary = process(str(root / entry["video"]), entry["exercise"],
                                    load_overrides(entry["exercise"]), tracker=PoseTracker())
    write_outputs(out_root / entry["id"], rows, events, summary)
    primary = {"knee_extension": "angle", "sit_to_stand": "angle",
               "shoulder_raise": "elevation"}[entry["exercise"]]
    res = {**{k: entry.get(k) for k in ("id", "volunteer", "split", "exercise", "lighting", "camera")},
           "manual_reps": labels["manual_reps"], "system_reps": summary["reps"],
           "rep_exact": summary["reps"] == labels["manual_reps"],
           "unreliable_frac": summary["unreliable_frames"] / max(summary["frames"], 1),
           "no_pose_frac": summary["no_pose_frames"] / max(summary["frames"], 1),
           "latency_ms_mean": summary["latency_ms_mean"], "latency_ms_p95": summary["latency_ms_p95"],
           "fb": feedback_counts(events, labels.get("pt_feedback", {}))}
    if "ref_angles" in labels:
        ae = angle_errors(rows, labels["ref_angles"], primary)
        res["angle_errors"], res["angle_missed"] = ae["errors"], ae["missed"]
    res["fb_delays_s"] = feedback_delays(events, labels.get("pt_times", {}))
    return res


def aggregate(results):
    errs = [e for r in results for e in r.get("angle_errors", [])]
    tp, fp, fn = (sum(r["fb"][k] for r in results) for k in ("tp", "fp", "fn"))
    p, r = precision_recall(tp, fp, fn)
    delays = [d for r_ in results for d in r_["fb_delays_s"]]
    return {
        "sets": len(results),
        "angle_mae_deg": float(np.mean(np.abs(errs))) if errs else None,
        "angle_bias_deg": float(np.mean(errs)) if errs else None,
        "angle_n": len(errs), "angle_missed": sum(r_.get("angle_missed", 0) for r_ in results),
        "exact_rep_frac": float(np.mean([r_["rep_exact"] for r_ in results])),
        "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn,
        "unreliable_frac": float(np.mean([r_["unreliable_frac"] for r_ in results])),
        "latency_ms_mean": float(np.mean([r_["latency_ms_mean"] for r_ in results])),
        "latency_ms_p95": float(np.max([r_["latency_ms_p95"] for r_ in results])),
        "feedback_delay_s_mean": float(np.mean(delays)) if delays else None,
    }


def _fmt(v):
    return "n/a" if v is None or v != v else (f"{v:.2f}" if isinstance(v, float) else str(v))


def markdown(results):
    cols = ["sets", "angle_mae_deg", "exact_rep_frac", "precision", "recall",
            "unreliable_frac", "latency_ms_mean", "feedback_delay_s_mean"]
    out = []
    groups = {"all": lambda r: "all", "split": lambda r: r["split"], "exercise": lambda r: r["exercise"],
              "lighting": lambda r: r["lighting"], "camera": lambda r: r["camera"],
              "volunteer": lambda r: r["volunteer"]}
    for gname, fn in groups.items():
        buckets = defaultdict(list)
        for r in results:
            buckets[str(fn(r))].append(r)
        out += [f"\n### By {gname}\n", "| " + gname + " | " + " | ".join(cols) + " |",
                "|" + "---|" * (len(cols) + 1)]
        for k, rs in sorted(buckets.items()):
            a = aggregate(rs)
            out.append(f"| {k} | " + " | ".join(_fmt(a[c]) for c in cols) + " |")
    t = aggregate([r for r in results if r["split"] == "test"] or results)
    out += ["\n### Targets (test split)\n",
            f"- Mean angle error <= {TARGETS['angle_mae_deg']} deg: {_fmt(t['angle_mae_deg'])}",
            f"- Exact rep count in >= {TARGETS['exact_rep_frac']:.0%} of sets: {_fmt(t['exact_rep_frac'])}",
            f"- Feedback precision >= {TARGETS['precision']}: {_fmt(t['precision'])}",
            f"- Feedback recall >= {TARGETS['recall']}: {_fmt(t['recall'])}"]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args(argv)
    root = a.manifest.resolve().parent.parent  # manifest lives in <root>/data/
    entries = json.loads(a.manifest.read_text())
    bad = [e["id"] for e in entries if e["exercise"] not in REGISTRY]
    if bad:
        raise SystemExit(f"unknown exercise in: {bad}")
    results = [evaluate_entry(e, root, a.out) for e in entries]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    md = markdown(results)
    (a.out / "tables.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
