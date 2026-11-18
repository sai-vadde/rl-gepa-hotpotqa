"""HotpotQA answer F1/EM, supporting-fact F1 and the GEPA feedback metric."""
import re
import string
from collections import Counter

import dspy


def normalize_answer(s):
    s = s.lower()
    punct = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in punct)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_f1_em(prediction, gold):
    """Official HotpotQA answer scoring (yes/no/noanswer must match exactly)."""
    n_pred, n_gold = normalize_answer(prediction), normalize_answer(gold)
    em = float(n_pred == n_gold)
    special = {"yes", "no", "noanswer"}
    if (n_pred in special or n_gold in special) and n_pred != n_gold:
        return 0.0, em
    p, g = n_pred.split(), n_gold.split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0, em
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec), em


def sp_f1(pred_ids, gold_ids):
    """Set F1 of selected sentence IDs against the gold supporting sentences."""
    p, g = set(pred_ids), set(gold_ids)
    if not p and not g:
        return 1.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def _short(text, n=140):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def feedback_for(ex, pred_ids, answer, pred_name):
    """Return (score, feedback_text) for one example. Uses hidden gold fields.

    GEPA only calls this on training examples, so reference text may appear
    in the feedback. The score is always answer F1, the optimisation target.
    """
    f1, em = answer_f1_em(answer, ex["answer"])
    gold, picked = set(ex["gold_ids"]), set(pred_ids)
    missing, extra = sorted(gold - picked), sorted(picked - gold)
    by_id = {s["id"]: s for s in ex["sentences"]}
    if pred_name and "select" in pred_name:
        parts = [f"Answer F1 of the whole pipeline was {f1:.2f}."]
        if not missing and not extra:
            parts.append("Evidence selection was exactly right.")
        if missing:
            miss = "; ".join(f"[{i}] ({by_id[i]['title']}) {_short(by_id[i]['text'])}" for i in missing)
            parts.append(f"Missed {len(missing)} needed supporting sentence(s): {miss}.")
        extra = [i for i in extra if i in by_id]
        if extra:
            ex_txt = "; ".join(f"[{i}] ({by_id[i]['title']})" for i in extra)
            parts.append(f"Selected {len(extra)} sentence(s) that were not needed: {ex_txt}.")
        if not pred_ids:
            parts.append("No valid sentence IDs were returned.")
        return f1, " ".join(parts)
    if em == 1.0:
        txt = "Answer matched the reference exactly."
    elif f1 > 0:
        txt = (f"Answer '{_short(answer, 80)}' partly overlaps the reference answer "
               f"'{ex['answer']}' (F1 {f1:.2f}). Keep answers short and in the reference style.")
    else:
        txt = f"Answer '{_short(answer, 80)}' did not match the reference answer '{ex['answer']}'."
        if missing:
            txt += " The selected evidence was also missing a needed sentence, so the selector may be at fault."
        else:
            txt += " The needed evidence had been selected, so the answering step failed."
    return f1, txt


def gold_ex(gold):
    """dspy.Example keeps the full example dict under .ex"""
    return gold.ex if hasattr(gold, "ex") else gold


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA callback: (gold, pred, trace, pred_name, pred_trace) -> score and feedback."""
    score, fb = feedback_for(gold_ex(gold), list(getattr(pred, "selected_ids", []) or []),
                             getattr(pred, "answer", "") or "", pred_name)
    return dspy.Prediction(score=score, feedback=fb)


def scalar_metric(gold, pred, trace=None):
    """Plain answer F1, used for MIPROv2 and for evaluation."""
    f1, _ = answer_f1_em(getattr(pred, "answer", "") or "", gold_ex(gold)["answer"])
    return f1
