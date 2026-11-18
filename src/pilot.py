"""Timing pilot on training questions. usage: python -m src.pilot MODEL N"""
import json
import sys
import time

import dspy

from .data import load_split
from .lm import make_lm, usage_totals
from .metrics import answer_f1_em, sp_f1
from .pipeline import TwoStepQA, to_example


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:4b"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    lm = make_lm(model)
    dspy.configure(lm=lm)
    data = load_split()["train"][:n]
    prog = TwoStepQA()
    rows = []
    for ex in data:
        e = to_example(ex)
        s0 = len(lm.history)
        t = time.time()
        pred = prog(question=e.question, context=e.context)
        dt = time.time() - t
        f1, em = answer_f1_em(pred.answer, ex["answer"])
        u = usage_totals(lm, s0)
        rows.append({"id": ex["id"], "sec": round(dt, 1), "f1": f1, "sp": sp_f1(pred.selected_ids, ex["gold_ids"]),
                     "pred": pred.answer, "gold": ex["answer"], "ids": pred.selected_ids, "gold_ids": ex["gold_ids"],
                     "err": pred.error, **u})
        print(json.dumps(rows[-1]), flush=True)
    print(f"{model}: mean sec/rollout {sum(r['sec'] for r in rows)/len(rows):.1f}  "
          f"F1 {sum(r['f1'] for r in rows)/len(rows):.2f}")


if __name__ == "__main__":
    main()
