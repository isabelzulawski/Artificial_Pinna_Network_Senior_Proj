"""Markdown tables and the saved outputs of a run (runs/<name>/)."""
import json

import numpy as np

CORE_COLS = [("gc_mean", "mean (deg)"), ("gc_median", "median"), ("gc_p95", "P95"), ("pct_lt5", "% < 5 deg"),
             ("pct_lt10", "% < 10 deg"), ("pct_gt45", "% > 45 deg"), ("az_mae", "Az-MAE"), ("el_mae", "El-MAE")]


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fmt(v, sd=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    s = f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"
    return s if not sd else f"{s} ± {sd:.2f}"


def core_table(result, reference, protocol, run_label="this run"):
    """The core test: this run next to the reference runs (mean ± sd over their seeds)."""
    rows = [[run_label] + [_fmt(result["metrics"]["core"][k]) for k, _ in CORE_COLS]]
    for fam, F in (reference["protocols"][protocol]["families"].items() if reference else []):
        lab = reference["families"][fam] + f" (seeds {', '.join(map(str, F['seeds']))})"
        rows.append([lab] + [_fmt(*F["core"][k]) for k, _ in CORE_COLS])
    if protocol != "seen" and "core_train" in result["metrics"]:
        rows.append([run_label + ", on its training directions"]
                    + [_fmt(result["metrics"]["core_train"][k]) for k, _ in CORE_COLS])
    return md_table(["model"] + [h for _, h in CORE_COLS], rows)


def groups_table(result, reference, protocol="seen"):
    """Mean error per condition: this run and the references."""
    refs = reference["protocols"][protocol]["families"] if reference else {}
    header = ["condition", "this run"] + [reference["families"][f] for f in refs]
    rows = [[c, _fmt(m["gc_mean"])] + [_fmt(refs[f]["gc_mean"][c][0]) if c in refs[f]["gc_mean"] else "-" for f in refs]
            for c, m in result["metrics"].items()]
    return md_table(header, rows)


def save_run(run_dir, result, history, info):
    """metrics.json (metrics per condition, training history and info) and examples.npz (per-example errors)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": result["protocol"], "conditions": result["conditions"], "metrics": result["metrics"],
               "history": history, "info": info}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    arrays = {f"{c}/{k}": v for c, ex in result["examples"].items() for k, v in ex.items() if v is not None}
    np.savez_compressed(run_dir / "examples.npz", **arrays)
