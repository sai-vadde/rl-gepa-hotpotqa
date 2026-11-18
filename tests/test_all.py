import dspy

from src.data import build_example, make_split, render_sentences
from src.metrics import (answer_f1_em, feedback_for, gepa_metric, normalize_answer,
                         scalar_metric, sp_f1)
from src.pipeline import TwoStepQA, lookup, valid_ids


# ---------- data split ----------
def test_split_disjoint_and_deterministic():
    tr = [f"t{i}" for i in range(500)]
    dv = [f"d{i}" for i in range(300)]
    a = make_split(tr, dv, 50, 50, 100, seed=42)
    b = make_split(list(reversed(tr)), dv, 50, 50, 100, seed=42)
    assert a == b
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (50, 50, 100)
    assert not set(a["train"]) & set(a["val"])
    assert not set(a["train"]) & set(a["test"])
    assert not set(a["val"]) & set(a["test"])
    c = make_split(tr, dv, 50, 50, 100, seed=7)
    assert c["train"] != a["train"]


def test_saved_split_if_present():
    import json
    from src.data import SPLIT_FILE
    if not SPLIT_FILE.exists():
        return
    s = json.loads(SPLIT_FILE.read_text())
    assert not set(s["train"]) & set(s["val"])
    assert not set(s["train"]) & set(s["test"])
    assert not set(s["val"]) & set(s["test"])
    assert len(s["train"]) == s["sizes"]["train"]


REC = {
    "id": "x1", "question": "Who is older, A or B?", "answer": "Alice",
    "type": "comparison", "level": "hard",
    "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 1]},
    "context": {"title": ["A", "B", "C"],
                "sentences": [["A was born in 1950.", " A lives in Rome."],
                              ["B is a city.", " B was born in 1960."],
                              ["C is unrelated."]]},
}


def test_build_example_ids():
    ex = build_example(REC)
    assert [s["id"] for s in ex["sentences"]] == [0, 1, 2, 3, 4]
    assert ex["gold_ids"] == [0, 3]
    text = render_sentences(ex["sentences"])
    assert text.splitlines()[3].startswith("[3] (B) B was born in 1960.")
    assert "answer" not in text.lower().split("\n")[0]  # no gold in the prompt text


def test_lookup_rejects_invented_ids():
    ex = build_example(REC)
    ctx = render_sentences(ex["sentences"])
    assert valid_ids(ctx) == {0, 1, 2, 3, 4}
    got = lookup(ctx, [3, 99, 0])
    assert len(got) == 2 and got[0].startswith("[3]")


# ---------- metrics (hand examples) ----------
def test_normalize():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


def test_answer_f1_em():
    assert answer_f1_em("Paris", "paris") == (1.0, 1.0)
    f1, em = answer_f1_em("the city of Paris", "Paris")  # p=1/3, r=1 -> 0.5
    assert em == 0.0 and abs(f1 - 0.5) < 1e-9
    assert answer_f1_em("London", "Paris") == (0.0, 0.0)
    assert answer_f1_em("yes it is", "yes")[0] == 0.0  # yes/no must match exactly


def test_sp_f1():
    assert sp_f1([1, 2], [1, 2]) == 1.0
    assert sp_f1([], [1]) == 0.0
    assert abs(sp_f1([1, 2, 3], [1, 2]) - 0.8) < 1e-9  # p=2/3 r=1 -> 0.8
    assert sp_f1([5], [1, 2]) == 0.0


# ---------- GEPA feedback callback ----------
def _gold():
    ex = build_example(REC)
    return dspy.Example(question=ex["question"], context=render_sentences(ex["sentences"]), ex=ex)


def test_feedback_select_module_names_missing_sentence():
    gold = _gold()
    pred = dspy.Prediction(answer="Bob", selected_ids=[0, 2])
    out = gepa_metric(gold, pred, None, "select", None)
    assert out.score == 0.0
    assert "Missed 1" in out.feedback and "[3]" in out.feedback
    assert "not needed" in out.feedback and "[2]" in out.feedback


def test_feedback_answer_module_variants():
    gold = _gold()
    good = gepa_metric(gold, dspy.Prediction(answer="alice", selected_ids=[0, 3]), None, "respond", None)
    assert good.score == 1.0 and "exactly" in good.feedback
    bad = gepa_metric(gold, dspy.Prediction(answer="Bob", selected_ids=[0, 3]), None, "respond", None)
    assert bad.score == 0.0 and "answering step failed" in bad.feedback
    bad2 = gepa_metric(gold, dspy.Prediction(answer="Bob", selected_ids=[0]), None, "respond", None)
    assert "selector" in bad2.feedback


def test_scalar_metric_matches_feedback_score():
    gold = _gold()
    pred = dspy.Prediction(answer="Alice Alice", selected_ids=[0, 3])
    s, _ = feedback_for(gold.ex, [0, 3], "Alice Alice", "respond")
    assert scalar_metric(gold, pred) == s


def test_program_with_dummy_lm():
    class Dummy(dspy.utils.DummyLM):
        pass
    lm = dspy.utils.DummyLM([{"evidence_ids": "[0, 3, 77]"}, {"answer": "A"}])
    with dspy.context(lm=lm):
        ex = build_example(REC)
        out = TwoStepQA()(question=ex["question"], context=render_sentences(ex["sentences"]))
    assert out.selected_ids == [0, 3]  # invented ID 77 dropped
    assert out.answer == "A"
