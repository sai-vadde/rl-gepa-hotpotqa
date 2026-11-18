"""Tables and graphs built ONLY from files saved under results/. No model is called here."""
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "report"

GEPA = "gepa_seed11"
METHODS = [("baseline", "Baseline"), ("mipro", "MIPROv2 light"), (GEPA, "GEPA (seed 11)")]
COLORS = {"baseline": "#7f7f7f", "mipro": "#1f77b4", GEPA: "#d62728"}


def set_results(path):
    """Point everything at another results folder (used to test on a smoke run)."""
    global RESULTS, OUT, GEPA, METHODS, COLORS
    RESULTS = Path(path)
    OUT = RESULTS / "report"
    GEPA = "gepa_seed11"
    if not (RESULTS / GEPA).exists():
        alt = sorted(p.name for p in RESULTS.glob("gepa_seed*"))
        GEPA = alt[0] if alt else GEPA
    seed = GEPA.replace("gepa_seed", "")
    METHODS = [("baseline", "Baseline"), ("mipro", "MIPROv2 light"), (GEPA, f"GEPA (seed {seed})")]
    COLORS = {"baseline": "#7f7f7f", "mipro": "#1f77b4", GEPA: "#d62728"}


def read_jsonl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def load_all(results=None):
    results = results or RESULTS
    out = {}
    for tag, label in METHODS:
        d = Path(results) / tag
        recs = read_jsonl(d / "test_predictions.jsonl")
        if not recs:
            continue
        meta = {}
        for name in ("optimise_meta.json", "test_summary.json"):
            if (d / name).exists():
                meta[name] = json.loads((d / name).read_text(encoding="utf-8"))
        out[tag] = {"label": label, "recs": recs, "meta": meta, "dir": d}
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(values, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(values)
    ms = sorted(mean(values[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def paired_diff(a, b, n=2000, seed=0):
    """Mean of (b - a) over shared IDs with a bootstrap 95% interval."""
    ia = {r["id"]: r["f1"] for r in a}
    d = [r["f1"] - ia[r["id"]] for r in b if r["id"] in ia]
    return mean(d), boot_ci(d, n, seed), len(d)


def results_table(data):
    rows = []
    for tag, v in data.items():
        recs = v["recs"]
        f1s = [r["f1"] for r in recs]
        lo, hi = boot_ci(f1s)
        opt = v["meta"].get("optimise_meta.json", {})
        calls = opt.get("metric_calls_reported") or opt.get("metric_calls") or opt.get("metric_calls_counted") or 0
        sec = opt.get("optimise_sec", 0.0)
        task_lm = opt.get("task_lm", {})
        extra_lm = opt.get("reflection_lm") or opt.get("prompt_lm") or {}
        rows.append({
            "tag": tag, "method": v["label"], "n_test": len(recs),
            "answer_f1": mean(f1s), "f1_lo": lo, "f1_hi": hi,
            "em": mean(r["em"] for r in recs), "sp_f1": mean(r["sp_f1"] for r in recs),
            "failures": sum(1 for r in recs if r["error"]),
            "opt_metric_calls": calls, "opt_minutes": sec / 60,
            "opt_model_calls": task_lm.get("calls", 0) + extra_lm.get("calls", 0),
            "opt_tokens": sum(task_lm.get(k, 0) + extra_lm.get(k, 0) for k in ("prompt_tokens", "completion_tokens")),
            "test_sec_per_rollout": mean(r["sec"] for r in recs),
            "test_tokens_per_q": mean(r["prompt_tokens"] + r["completion_tokens"] for r in recs),
        })
    return rows


def error_breakdown(recs):
    """Split test questions into outcome classes using saved selections only."""
    c = {"correct (F1>=0.5)": 0, "wrong: selector missed evidence": 0,
         "wrong: evidence found, answer wrong": 0, "wrong: pipeline error": 0}
    for r in recs:
        if r["f1"] >= 0.5:
            c["correct (F1>=0.5)"] += 1
        elif r["error"]:
            c["wrong: pipeline error"] += 1
        elif set(r["gold_ids"]) - set(r["selected_ids"]):
            c["wrong: selector missed evidence"] += 1
        else:
            c["wrong: evidence found, answer wrong"] += 1
    return c


def fmt(x, d=3):
    return f"{x:.{d}f}"


def write_tables(data, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    head = "| Method | N | Answer F1 (95% CI) | EM | SP-F1 | Opt. metric calls | Opt. minutes | Test s/question |\n|---|---|---|---|---|---|---|---|\n"
    md = head + "".join(
        f"| {r['method']} | {r['n_test']} | {fmt(r['answer_f1'])} ({fmt(r['f1_lo'])} to {fmt(r['f1_hi'])}) | "
        f"{fmt(r['em'])} | {fmt(r['sp_f1'])} | {r['opt_metric_calls']} | {r['opt_minutes']:.0f} | "
        f"{r['test_sec_per_rollout']:.1f} |\n" for r in rows)
    (OUT / "results_table.md").write_text(md, encoding="utf-8")
    (OUT / "results_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tex = ("\\begin{tabular}{lrrrrrr}\n\\hline\nMethod & Answer F1 & 95\\% interval & EM & SP-F1 & Opt.\\ calls & Opt.\\ min \\\\ \\hline\n"
           + "".join(f"{r['method']} & {fmt(r['answer_f1'])} & {fmt(r['f1_lo'])}--{fmt(r['f1_hi'])} & {fmt(r['em'])} & "
                     f"{fmt(r['sp_f1'])} & {r['opt_metric_calls']} & {r['opt_minutes']:.0f} \\\\\n" for r in rows)
           + "\\hline\n\\end{tabular}\n")
    (OUT / "results_table.tex").write_text(tex, encoding="utf-8")
    diffs = []
    if "baseline" in data:
        for tag in data:
            if tag == "baseline":
                continue
            m, ci, n = paired_diff(data["baseline"]["recs"], data[tag]["recs"])
            diffs.append({"method": data[tag]["label"], "vs": "Baseline", "mean_f1_diff": m, "lo": ci[0], "hi": ci[1], "n": n})
    if "mipro" in data and GEPA in data:
        m, ci, n = paired_diff(data["mipro"]["recs"], data[GEPA]["recs"])
        diffs.append({"method": data[GEPA]["label"], "vs": "MIPROv2 light", "mean_f1_diff": m, "lo": ci[0], "hi": ci[1], "n": n})
    (OUT / "paired_diffs.json").write_text(json.dumps(diffs, indent=1), encoding="utf-8")
    eb = {data[t]["label"]: error_breakdown(data[t]["recs"]) for t in data}
    (OUT / "error_breakdown.json").write_text(json.dumps(eb, indent=1), encoding="utf-8")
    return diffs, eb


def fig_scores(rows):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.26
    for j, (key, name) in enumerate([("answer_f1", "Answer F1"), ("em", "Exact match"), ("sp_f1", "Supporting-fact F1")]):
        xs = [i + (j - 1) * w for i in range(len(rows))]
        vals = [r[key] for r in rows]
        ax.bar(xs, vals, w, label=name, color=["#4c72b0", "#dd8452", "#55a868"][j])
        if key == "answer_f1":
            ax.errorbar(xs, vals, yerr=[[r["answer_f1"] - r["f1_lo"] for r in rows], [r["f1_hi"] - r["answer_f1"] for r in rows]],
                        fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["method"] for r in rows])
    ax.set_ylabel("score on held-out test questions")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scores.png", dpi=200)
    plt.close(fig)


def fig_cost(rows):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
    names = [r["method"] for r in rows]
    cols = [COLORS.get(r["tag"], "#444") for r in rows]
    axs[0].bar(names, [r["opt_metric_calls"] for r in rows], color=cols)
    axs[0].set_ylabel("metric calls during optimisation")
    axs[1].bar(names, [r["opt_minutes"] for r in rows], color=cols)
    axs[1].set_ylabel("optimisation minutes")
    for a in axs:
        a.tick_params(axis="x", labelrotation=20, labelsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost.png", dpi=200)
    plt.close(fig)


def gepa_curve(data):
    """(metric calls when found, validation score) for each GEPA candidate."""
    p = data.get(GEPA, {}).get("dir")
    if not p or not (Path(p) / "detailed_results.json").exists():
        return None
    dr = json.loads((Path(p) / "detailed_results.json").read_text(encoding="utf-8"))
    xs, ys = dr.get("discovery_eval_counts"), dr.get("val_aggregate_scores")
    if not xs or not ys:
        return None
    return xs, ys


def fig_gepa_curve(data):
    c = gepa_curve(data)
    if not c:
        return False
    xs, ys = c
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    best, run = [], -1
    for i in order:
        run = max(run, ys[i])
        best.append(run)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.scatter([xs[i] for i in order], [ys[i] for i in order], color="#d62728", label="candidate", zorder=3)
    ax.step([xs[i] for i in order], best, where="post", color="black", label="best so far")
    ax.set_xlabel("metric calls used when candidate was found")
    ax.set_ylabel("validation answer F1")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gepa_curve.png", dpi=200)
    plt.close(fig)
    return True


def fig_errors(eb):
    labels = list(next(iter(eb.values())).keys())
    cols = ["#55a868", "#c44e52", "#dd8452", "#8172b2"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bottoms = [0] * len(eb)
    for k, lab in enumerate(labels):
        vals = [eb[m][lab] for m in eb]
        ax.bar(list(eb.keys()), vals, bottom=bottoms, label=lab, color=cols[k])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("test questions")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_errors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(results=None):
    from . import report_tex
    if results:
        set_results(results)
    data = load_all()
    if not data:
        print("no saved results yet")
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    rows = results_table(data)
    diffs, eb = write_tables(data, rows)
    fig_scores(rows)
    fig_cost([r for r in rows if r["tag"] != "baseline"] or rows)
    fig_gepa_curve(data)
    fig_errors(eb)
    report_tex.write_all(sys.modules[__name__], data, rows, diffs, eb)
    print((OUT / "results_table.md").read_text(encoding="utf-8"))
    return rows, diffs, eb


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
