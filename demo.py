"""Demo: one HotpotQA test question through the baseline prompt and the GEPA-optimised prompt.

python demo.py                 # replay saved predictions (no model needed, always works)
python demo.py --live          # run both frozen programs on this question with Ollama (about 1 minute)
python demo.py --pick 5        # use test question number 5 instead of the automatic pick
"""
import argparse
import os
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def pick_question(base, gepa, forced=None):
    if forced is not None:
        return forced
    gi = {r["id"]: r for r in gepa}
    # first question GEPA got right and the baseline got wrong, else the first one
    for i, b in enumerate(base):
        g = gi.get(b["id"])
        if g and g["f1"] >= 0.99 and b["f1"] < 0.5:
            return i
    return 0


def show(title, prompts, rec):
    print(f"\n=== {title} ===")
    for mod, text in prompts["instructions"].items():
        print(f"[{mod} instructions]")
        print(textwrap.indent(textwrap.fill(text, 100), "  "))
    print(f"selected sentence IDs: {rec['selected_ids']}   (gold: {rec['gold_ids']})")
    print(f"answer: {rec['pred']!r}   gold: {rec['gold']!r}   answer F1: {rec['f1']:.2f}   SP F1: {rec['sp_f1']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--pick", type=int, default=None)
    a = ap.parse_args()
    base = jl(RES / "baseline" / "test_predictions.jsonl")
    gepa = jl(RES / "gepa_seed11" / "test_predictions.jsonl")
    bp = json.loads((RES / "baseline" / "prompts.json").read_text(encoding="utf-8"))
    gp = json.loads((RES / "gepa_seed11" / "prompts.json").read_text(encoding="utf-8"))
    i = pick_question(base, gepa, a.pick)
    b, g = base[i], next(r for r in gepa if r["id"] == base[i]["id"])
    print(f"Test question #{i}: {b['question']}")

    if not a.live:
        show("Baseline prompt (saved prediction)", bp, b)
        show("GEPA-optimised prompt (saved prediction)", gp, g)
        return

    import dspy
    from src.data import load_split, render_sentences
    from src.lm import make_lm
    from src.runner import load_program
    ex = next(e for e in load_split()["test"] if e["id"] == b["id"])
    ctx = render_sentences(ex["sentences"])
    dspy.configure(lm=make_lm(os.environ.get("GEPA_MODEL", "qwen3:4b"), 256))
    for title, d, prompts in (("Baseline prompt (live)", "baseline", bp), ("GEPA-optimised prompt (live)", "gepa_seed11", gp)):
        pred = load_program(RES / d)(question=ex["question"], context=ctx)
        from src.metrics import answer_f1_em, sp_f1
        f1, _ = answer_f1_em(pred.answer, ex["answer"])
        show(title, prompts, {"selected_ids": pred.selected_ids, "gold_ids": ex["gold_ids"], "pred": pred.answer,
                              "gold": ex["answer"], "f1": f1, "sp_f1": sp_f1(pred.selected_ids, ex["gold_ids"])})


if __name__ == "__main__":
    main()
