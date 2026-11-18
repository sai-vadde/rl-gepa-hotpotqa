"""LaTeX snippets for the report, generated from saved results only.

numbers.tex   one \newcommand per number used in the prose (missing values read n/a)
prompts.tex   the final instructions of each method
grpo_table.tex
"""
import json
import textwrap
from pathlib import Path

BS = "\\"


def tex_escape(t):
    for a, b in ((BS, " "), ("&", BS + "&"), ("%", BS + "%"), ("_", BS + "_"), ("#", BS + "#")):
        t = t.replace(a, b)
    return t


def fmt(x, d=3):
    return f"{x:.{d}f}"


def write_numbers(ra, data, rows, diffs, eb):
    M = {}
    names = {"baseline": "Base", "mipro": "Mipro", ra.GEPA: "Gepa"}
    for r in rows:
        n = names.get(r["tag"], r["tag"])
        M[n + "F1"] = fmt(r["answer_f1"])
        M[n + "Lo"] = fmt(r["f1_lo"])
        M[n + "Hi"] = fmt(r["f1_hi"])
        M[n + "EM"] = fmt(r["em"])
        M[n + "SP"] = fmt(r["sp_f1"])
        M[n + "Calls"] = str(r["opt_metric_calls"])
        M[n + "Min"] = f"{r['opt_minutes']:.0f}"
        M[n + "OptTok"] = f"{r['opt_tokens']:,}".replace(",", "{,}")
        M[n + "SecQ"] = f"{r['test_sec_per_rollout']:.1f}"
        M[n + "TokQ"] = f"{r['test_tokens_per_q']:.0f}"
        M[n + "N"] = str(r["n_test"])
        M[n + "Fail"] = str(r["failures"])
    for d in diffs:
        key = "Gepa" if d["method"].startswith("GEPA") else "Mipro"
        vs = "Base" if d["vs"] == "Baseline" else "Mipro"
        k = f"Diff{key}{vs}"
        M[k] = f"{d['mean_f1_diff']:+.3f}"
        M[k + "Lo"] = f"{d['lo']:+.3f}"
        M[k + "Hi"] = f"{d['hi']:+.3f}"
    for m, c in eb.items():
        n = "Base" if m == "Baseline" else ("Mipro" if m.startswith("MIPRO") else "Gepa")
        for i, v in enumerate(c.values()):
            M[f"{n}Err{'ABCD'[i]}"] = str(v)
    meta = data.get(ra.GEPA, {}).get("meta", {}).get("optimise_meta.json", {})
    M["GepaCountedCalls"] = str(meta.get("metric_calls_counted", "n/a"))
    M["GepaBudget"] = str(meta.get("budget", "n/a"))
    M["GepaSeed"] = str(meta.get("seed", "n/a"))
    env = data.get("baseline", {}).get("meta", {}).get("test_summary.json", {}).get("env", {})
    M["ModelTag"] = tex_escape(str(env.get("model", "n/a")))
    M["EnvPlatform"] = tex_escape(str(env.get("platform", "n/a")))
    M["DspyVersion"] = str(env.get("dspy", "n/a"))
    M["LitellmVersion"] = str(env.get("litellm", "n/a"))
    c = ra.gepa_curve(data)
    if c:
        M["GepaCandidates"] = str(len(c[0]))
        M["GepaBestVal"] = fmt(max(c[1]))
        M["GepaFirstVal"] = fmt(c[1][0])
    g = Path(ra.RESULTS) / "grpo" / "grpo_summary.json"
    if g.exists():
        gs = json.loads(g.read_text(encoding="utf-8"))
        M["GrpoStartF1"] = fmt(gs["start"]["answer_f1"])
        M["GrpoTrainedF1"] = fmt(gs["trained"]["answer_f1"])
        M["GrpoStartEM"] = fmt(gs["start"]["exact_match"])
        M["GrpoTrainedEM"] = fmt(gs["trained"]["exact_match"])
        M["GrpoDiff"] = f"{gs['trained']['answer_f1'] - gs['start']['answer_f1']:+.3f}"
        M["GrpoSteps"] = str(gs["steps"])
        M["GrpoMin"] = f"{gs['train_sec'] / 60:.1f}"
        gd = Path(ra.RESULTS) / "grpo"
        try:
            log = [x for x in json.loads((gd / "train_log.json").read_text(encoding="utf-8")) if "reward" in x]
            M["GrpoRewFirst"] = fmt(log[0]["reward"], 2)
            M["GrpoRewLast"] = fmt(log[-1]["reward"], 2)
            M["GrpoRewBest"] = fmt(max(x["reward"] for x in log), 2)
            M["GrpoRewLastFive"] = fmt(sum(x["reward"] for x in log[-5:]) / 5, 2)
            M["GrpoRewLogs"] = str(len(log))
            a = [json.loads(x) for x in (gd / "start_predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            b = {json.loads(x)["id"]: json.loads(x) for x in (gd / "trained_predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
            M["GrpoChanged"] = str(sum(1 for r in a if b[r["id"]]["pred"] != r["pred"]))
            M["GrpoBetter"] = str(sum(1 for r in a if b[r["id"]]["f1"] > r["f1"]))
            M["GrpoWorse"] = str(sum(1 for r in a if b[r["id"]]["f1"] < r["f1"]))
        except Exception:
            pass
    seeds = {}
    for sd in (11, 22, 33):
        sp_ = Path(ra.RESULTS) / f"gepa_seed{sd}" / "test_summary.json"
        if sp_.exists():
            seeds[sd] = json.loads(sp_.read_text(encoding="utf-8"))["answer_f1"]
    if len(seeds) >= 2:
        import statistics
        vals = list(seeds.values())
        for k_, v_ in seeds.items():
            M["Seed" + {11: "Eleven", 22: "TwentyTwo", 33: "ThirtyThree"}[k_] + "Score"] = fmt(v_)
        M["SeedCount"] = str(len(vals))
        M["SeedMean"] = fmt(statistics.mean(vals))
        M["SeedSd"] = fmt(statistics.stdev(vals))
        M["SeedMin"] = fmt(min(vals))
        M["SeedMax"] = fmt(max(vals))
    g1 = Path(ra.RESULTS) / "grpo_attempt1_thinking_on" / "grpo_summary.json"
    if g1.exists():
        a1 = json.loads(g1.read_text(encoding="utf-8"))
        M["GrpoOneStartF1"] = fmt(a1["start"]["answer_f1"])
        M["GrpoOneTrainedF1"] = fmt(a1["trained"]["answer_f1"])
        M["GrpoOneStartEM"] = fmt(a1["start"]["exact_match"])
        M["GrpoOneTrainedEM"] = fmt(a1["trained"]["exact_match"])
        M["GrpoOneMin"] = f"{a1['train_sec'] / 60:.0f}"
    sp = ra.ROOT / "data" / "splits.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        M["NTrain"], M["NVal"], M["NTest"] = (str(s["sizes"][k]) for k in ("train", "val", "test"))
    lines = ["% generated by src/report_tex.py from saved results; do not edit"]
    # control-sequence names cannot hold digits, so F1 becomes Fa
    lines += [BS + "newcommand{" + BS + k.replace("F1", "Fa") + "}{" + v + "}" for k, v in sorted(M.items())]
    (ra.OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prompts(ra, data):
    out = []
    for tag, label in ra.METHODS:
        p = Path(ra.RESULTS) / tag / "prompts.json"
        if not p.exists():
            continue
        pr = json.loads(p.read_text(encoding="utf-8"))
        demos = ", ".join(f"{k} {v}" for k, v in pr["n_demos"].items())
        out.append(BS + "textbf{" + label + "} (demonstrations per module: " + demos + ")")
        out.append(BS + "begin{verbatim}")
        for mod, text in pr["instructions"].items():
            out.append(f"[{mod}]")
            out += textwrap.wrap(" ".join(text.split())[:3500], 76)
            out.append("")
        out.append(BS + "end{verbatim}")
    (ra.OUT / "prompts.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def write_grpo(ra):
    g = Path(ra.RESULTS) / "grpo" / "grpo_summary.json"
    if not g.exists():
        (ra.OUT / "grpo_table.tex").write_text("The GRPO run did not finish, so there is no table for it.\n", encoding="utf-8")
        return
    gs = json.loads(g.read_text(encoding="utf-8"))
    rows = [("Before GRPO", gs["start"]), (f"After {gs['steps']} steps", gs["trained"])]
    t = [BS + "begin{tabular}{lrr}", BS + "hline", "Qwen3-0.6B answer model & Answer F1 & EM " + BS + BS + " " + BS + "hline"]
    t += [f"{n} & {fmt(v['answer_f1'])} & {fmt(v['exact_match'])} " + BS + BS for n, v in rows]
    t += [BS + "hline", BS + "end{tabular}"]
    (ra.OUT / "grpo_table.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- extra tables for Part C
def write_cost(ra, rows):
    t = [BS + "begin{tabular}{lrrrrrr}", BS + "hline",
         "Method & Metric calls & Model calls & Tokens & Minutes & Test s/question & Test tokens/question " + BS + BS + " " + BS + "hline"]
    for r in rows:
        tok = f"{r['opt_tokens']:,}".replace(",", "{,}")
        t.append(f"{r['method']} & {r['opt_metric_calls']} & {r['opt_model_calls']} & {tok} & {r['opt_minutes']:.0f} & "
                 f"{r['test_sec_per_rollout']:.1f} & {r['test_tokens_per_q']:.0f} " + BS + BS)
    t += [BS + "hline", BS + "end{tabular}"]
    (ra.OUT / "cost_table.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def write_error_table(ra, eb):
    labels = list(next(iter(eb.values())).keys())
    t = [BS + "begin{tabular}{l" + "r" * len(eb) + "}", BS + "hline",
         "Outcome & " + " & ".join(eb.keys()) + " " + BS + BS + " " + BS + "hline"]
    for lab in labels:
        t.append(tex_escape(lab) + " & " + " & ".join(str(eb[m][lab]) for m in eb) + " " + BS + BS)
    t += [BS + "hline", BS + "end{tabular}"]
    (ra.OUT / "error_table.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def write_errors_to_read(ra, data, n=20):
    """Pick up to n wrong answers of the GEPA program (fixed seed) for a hand read."""
    import random
    g = data.get(ra.GEPA)
    if not g:
        return
    wrong = [r for r in g["recs"] if r["f1"] < 0.5]
    random.Random(5).shuffle(wrong)
    pick = wrong[:n]
    base = {r["id"]: r for r in data.get("baseline", {}).get("recs", [])}
    lines = [f"# {len(pick)} wrong GEPA test answers to read by hand (seed 5; {len(wrong)} wrong in total)",
             "", "Label each as one of: `missed_evidence`, `wrong_reasoning`, `format_only`, `other`.",
             "Save the labels as `results/error_labels.json` like {\"<id>\": \"missed_evidence\"}.", ""]
    for r in pick:
        b = base.get(r["id"])
        lines += [f"## {r['id']}", f"Q: {r['question']}", f"gold: {r['gold']!r} | GEPA: {r['pred']!r} (F1 {r['f1']:.2f})"
                  + (f" | baseline: {b['pred']!r} (F1 {b['f1']:.2f})" if b else ""),
                  f"selected IDs {r['selected_ids']} | gold IDs {r['gold_ids']}", ""]
    (ra.OUT / "errors_to_read.md").write_text("\n".join(lines), encoding="utf-8")


def write_label_table(ra):
    p = Path(ra.RESULTS) / "error_labels.json"
    if not p.exists():
        (ra.OUT / "label_table.tex").write_text(BS + "fixme{hand labels not saved yet (file error labels missing in results)}\n", encoding="utf-8")
        (ra.OUT / "label_numbers.tex").write_text("", encoding="utf-8")
        return
    from collections import Counter
    lab = json.loads(p.read_text(encoding="utf-8"))
    c = Counter(lab.values())
    names = [("missed_evidence", "Selector missed needed evidence"), ("wrong_reasoning", "Evidence found, reasoning wrong"),
             ("format_only", "Right content, wrong format or length"), ("other", "Other")]
    t = [BS + "begin{tabular}{lr}", BS + "hline", "Reading of inspected failures & Questions " + BS + BS + " " + BS + "hline"]
    t += [f"{n} & {c.get(k, 0)} " + BS + BS for k, n in names]
    t += [f"Total inspected & {sum(c.values())} " + BS + BS, BS + "hline", BS + "end{tabular}"]
    (ra.OUT / "label_table.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    nums = [BS + "newcommand{" + BS + "Lab" + k.split("_")[0].capitalize() + "}{" + str(c.get(k, 0)) + "}" for k, _ in names]
    nums.append(BS + "newcommand{" + BS + "LabTotal}{" + str(sum(c.values())) + "}")
    (ra.OUT / "label_numbers.tex").write_text("\n".join(nums) + "\n", encoding="utf-8")


def write_run_label(ra):
    smoke = "smoke" in str(ra.RESULTS).lower()
    R = Path(ra.RESULTS)
    partial = (not (R / ra.GEPA / "test_summary.json").exists() or not (R / "mipro" / "test_summary.json").exists()
               or not (R / "grpo" / "grpo_summary.json").exists())
    label = "SMOKE" if smoke else ("PARTIAL" if partial else "")
    with open(ra.OUT / "numbers.tex", "a", encoding="utf-8") as f:
        f.write(BS + "newcommand{" + BS + "RunLabel}{" + label + "}\n")


def export_parts(ra, root):
    """Copy tables and figures into report/partC (self-contained for Overleaf)."""
    import shutil
    plan = {
        "partC": {"tables": ["numbers.tex", "analysis_numbers.tex", "type_table.tex", "label_numbers.tex", "results_table.tex", "prompts.tex", "grpo_table.tex",
                             "cost_table.tex", "error_table.tex", "label_table.tex"],
                  "figures": ["fig_scores.png", "fig_cost.png", "fig_gepa_curve.png", "fig_errors.png"]},
    }
    for part, kinds in plan.items():
        for kind, files in kinds.items():
            d = Path(root) / "report" / part / kind
            d.mkdir(parents=True, exist_ok=True)
            for f in files:
                src = ra.OUT / f
                if src.exists():
                    shutil.copy(src, d / f)
                elif (d / f).exists():
                    (d / f).unlink()  # do not leave an old file behind


def write_all(ra, data, rows, diffs, eb):
    write_numbers(ra, data, rows, diffs, eb)
    write_run_label(ra)
    write_prompts(ra, data)
    write_grpo(ra)
    write_cost(ra, rows)
    write_error_table(ra, eb)
    write_errors_to_read(ra, data)
    write_label_table(ra)
    from . import analysis
    analysis.write(ra)
    export_parts(ra, ra.ROOT)
