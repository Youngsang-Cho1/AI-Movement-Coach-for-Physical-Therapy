"""Pure metric functions (no video/model dependency) so they are unit-testable."""
import numpy as np

TARGETS = {"angle_mae_deg": 5.0, "exact_rep_frac": 0.95, "precision": 0.90, "recall": 0.85}
NON_CORRECTIVE = {"good_rep"}


def angle_errors(rows, ref, key="angle", tol_s=0.1):
    """ref: [(t, deg)]. Compare against nearest reliable frame; unmatched refs count as missed."""
    ok = [r for r in rows if r["reliable"] and key in r]
    ts = np.array([r["t"] for r in ok])
    errs, missed = [], 0
    for t, deg in ref:
        if len(ts) == 0 or np.min(np.abs(ts - t)) > tol_s:
            missed += 1
            continue
        errs.append(ok[int(np.argmin(np.abs(ts - t)))][key] - deg)
    return {"errors": errs, "missed": missed}


def feedback_counts(events, pt_feedback):
    """Match (attempt, code) pairs between system events and PT labels; good_rep ignored."""
    sys_pairs = {(e["attempt"], e["code"]) for e in events if e["code"] not in NON_CORRECTIVE}
    pt_pairs = {(int(a), c) for a, codes in pt_feedback.items() for c in codes
                if c not in NON_CORRECTIVE}
    return {"tp": len(sys_pairs & pt_pairs), "fp": len(sys_pairs - pt_pairs),
            "fn": len(pt_pairs - sys_pairs),
            "fp_pairs": sorted(sys_pairs - pt_pairs), "fn_pairs": sorted(pt_pairs - sys_pairs)}


def precision_recall(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r


def feedback_delays(events, pt_times):
    """pt_times: {"attempt:code": t_s when feedback should ideally fire}. Returns seconds."""
    out = []
    for e in events:
        k = f"{e['attempt']}:{e['code']}"
        if k in pt_times:
            out.append(e["t"] - pt_times[k])
    return out
