"""Two-module DSPy program: select evidence sentences, then answer from them."""
import re

import dspy

MAX_SELECTED = 6
FAILURES = {"select_error": 0, "answer_error": 0}


class Select(dspy.Signature):
    """Read the question and the numbered sentences. Return the IDs of the sentences needed to answer it."""

    question: str = dspy.InputField()
    sentences: str = dspy.InputField(desc="numbered candidate sentences, each as [id] (paragraph title) text")
    evidence_ids: list[int] = dspy.OutputField(desc="at most six sentence IDs")


class Respond(dspy.Signature):
    """Answer the question using the evidence. Reply with a short answer only."""

    question: str = dspy.InputField()
    evidence: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="short answer, a few words")


def valid_ids(context):
    return {int(m) for m in re.findall(r"^\[(\d+)\]", context, flags=re.M)}


def lookup(context, ids):
    """Fixed lookup: fetch original sentence lines for valid IDs, drop invented IDs."""
    lines = {int(m.group(1)): m.group(0) for m in re.finditer(r"^\[(\d+)\].*$", context, flags=re.M)}
    return [lines[i] for i in ids if i in lines]


class TwoStepQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.select = dspy.Predict(Select)
        self.respond = dspy.Predict(Respond)

    def forward(self, question, context):
        ok = valid_ids(context)
        ids, err = [], None
        try:
            raw = self.select(question=question, sentences=context).evidence_ids
            for i in raw:
                try:
                    i = int(i)
                except (TypeError, ValueError):
                    continue
                if i in ok and i not in ids:
                    ids.append(i)
            ids = ids[:MAX_SELECTED]
        except Exception as e:  # parse failure or LM error: no evidence, keep going
            FAILURES["select_error"] += 1
            err = repr(e)[:200]
        evidence = "\n".join(lookup(context, ids)) or "(no evidence selected)"
        answer = ""
        try:
            answer = (self.respond(question=question, evidence=evidence).answer or "").strip()
        except Exception as e:
            FAILURES["answer_error"] += 1
            err = (err or "") + " | " + repr(e)[:200]
        return dspy.Prediction(answer=answer, selected_ids=ids, error=err)


def to_example(ex):
    """Model inputs are question and context only. The full dict rides along as hidden `ex`."""
    from .data import render_sentences
    e = dspy.Example(question=ex["question"], context=render_sentences(ex["sentences"]), ex=ex)
    return e.with_inputs("question", "context")
