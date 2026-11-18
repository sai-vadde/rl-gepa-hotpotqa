"""Shared run logic: evaluation, MIPROv2, GEPA. Everything is saved under results/<tag>/."""
import json
import platform
import time
from pathlib import Path

import dspy

from .data import ROOT, load_split
from .lm import make_lm, usage_totals
from .metrics import answer_f1_em, gepa_metric, scalar_metric, sp_f1
from .pipeline import FAILURES, TwoStepQA, to_example

RESULTS = ROOT / "results"
MODEL = "qwen3:4b"
GEPA_SEED = 11
TASK_TOKENS = 256
REFLECT_TOKENS = 1024


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def out_dir(tag):
    d = RESULTS / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def prompts_of(program):
    return {name: pred.signature.instructions for name, pred in program.named_predictors()}


def demos_of(program):
    return {name: [dict(d) if hasattr(d, "items") else str(d) for d in pred.demos]
            for name, pred in program.named_predictors()}


def save_program(program, d):
    program.save(str(d / "program.json"))
    (d / "prompts.json").write_text(json.dumps(
        {"instructions": prompts_of(program), "n_demos": {k: len(v) for k, v in demos_of(program).items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")


def load_program(d):
    p = TwoStepQA()
    p.load(str(Path(d) / "program.json"))
    return p


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def evaluate(program, examples, lm, path, limit=None):
    """Score a frozen program one example at a time. Resumable: appends to `path`."""
    path = Path(path)
    done = {r["id"] for r in read_jsonl(path)}
    todo = [e for e in examples if e["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    log(f"evaluate -> {path.name}: {len(done)} done, {len(todo)} to go")
    with open(path, "a", encoding="utf-8") as f:
        for i, ex in enumerate(todo):
            ie = to_example(ex)
            s0, t0 = len(lm.history), time.time()
            pred = program(question=ie.question, context=ie.context)
            sec = time.time() - t0
            f1, em = answer_f1_em(pred.answer, ex["answer"])
            rec = {"id": ex["id"], "question": ex["question"], "gold": ex["answer"], "pred": pred.answer,
                   "selected_ids": pred.selected_ids, "gold_ids": ex["gold_ids"], "f1": f1, "em": em,
                   "sp_f1": sp_f1(pred.selected_ids, ex["gold_ids"]),
                   "sec": round(sec, 2), "error": pred.error, "type": ex["type"], **usage_totals(lm, s0)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                log(f"  {i + 1}/{len(todo)}")
    return read_jsonl(path)


def summarise(records):
    n = len(records)
    if not n:
        return {}
    return {
        "n": n,
        "answer_f1": sum(r["f1"] for r in records) / n,
        "exact_match": sum(r["em"] for r in records) / n,
        "sp_f1": sum(r["sp_f1"] for r in records) / n,
        "failures": sum(1 for r in records if r["error"]),
        "mean_sec_per_rollout": sum(r["sec"] for r in records) / n,
        "model_calls": sum(r["calls"] for r in records),
        "prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "completion_tokens": sum(r["completion_tokens"] for r in records),
    }


def env_info():
    import litellm
    return {"python": platform.python_version(), "dspy": dspy.__version__, "litellm": litellm.__version__
            if hasattr(litellm, "__version__") else "1.77.7", "model": MODEL,
            "platform": platform.platform(), "time": now()}


class Counter:
    """Wrap a metric so actual metric calls are counted."""

    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, *a, **k):
        self.n += 1
        return self.fn(*a, **k)


# ---------------------------------------------------------------- methods
def run_baseline(split_data, limit=None, tag="baseline"):
    d = out_dir(tag)
    lm = make_lm(MODEL, TASK_TOKENS)
    dspy.configure(lm=lm)
    prog = TwoStepQA()
    save_program(prog, d)
    t0 = time.time()
    recs = evaluate(prog, split_data["test"], lm, d / "test_predictions.jsonl", limit)
    return finish_eval(d, tag, recs, extra={"optimise_calls": 0, "optimise_sec": 0.0})


def finish_eval(d, tag, recs, extra):
    summ = summarise(recs)
    summ["failures_total_counter"] = dict(FAILURES)
    summ.update(extra)
    summ["env"] = env_info()
    (d / "test_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log(f"{tag}: test {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items() if k in ('n', 'answer_f1', 'exact_match', 'sp_f1')})}")
    return summ


def run_mipro(split_data, limit=None, tag="mipro"):
    d = out_dir(tag)
    prog_path = d / "program.json"
    meta_path = d / "optimise_meta.json"
    task_lm, prompt_lm = make_lm(MODEL, TASK_TOKENS), make_lm(MODEL, REFLECT_TOKENS)
    dspy.configure(lm=task_lm)
    if not prog_path.exists():
        metric = Counter(scalar_metric)
        opt = dspy.MIPROv2(metric=metric, prompt_model=prompt_lm, task_model=task_lm, auto="light",
                           max_bootstrapped_demos=2, max_labeled_demos=2, seed=GEPA_SEED, num_threads=1,
                           verbose=True, log_dir=str(d / "mipro_log"))
        train = [to_example(e) for e in split_data["train"]]
        val = [to_example(e) for e in split_data["val"]]
        log(f"MIPROv2 light: {len(train)} train, {len(val)} val")
        t0 = time.time()
        compiled = opt.compile(TwoStepQA(), trainset=train, valset=val, requires_permission_to_run=False)
        sec = time.time() - t0
        save_program(compiled, d)
        meta = {"optimise_sec": sec, "metric_calls": metric.n,
                "task_lm": usage_totals(task_lm), "prompt_lm": usage_totals(prompt_lm), "seed": GEPA_SEED,
                "auto": "light", "env": env_info()}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"MIPROv2 done in {sec:.0f}s, {metric.n} metric calls")
    meta = json.loads(meta_path.read_text())
    prog = load_program(d)
    test_lm = make_lm(MODEL, TASK_TOKENS)
    dspy.configure(lm=test_lm)
    recs = evaluate(prog, split_data["test"], test_lm, d / "test_predictions.jsonl", limit)
    return finish_eval(d, tag, recs, extra={"optimise_calls": meta["metric_calls"],
                                            "optimise_sec": meta["optimise_sec"]})


def run_gepa(split_data, budget=900, seed=GEPA_SEED, limit=None, tag=None):
    tag = tag or f"gepa_seed{seed}"
    d = out_dir(tag)
    prog_path = d / "program.json"
    meta_path = d / "optimise_meta.json"
    task_lm, reflect_lm = make_lm(MODEL, TASK_TOKENS), make_lm(MODEL, REFLECT_TOKENS, temperature=1.0)
    dspy.configure(lm=task_lm)
    if not prog_path.exists():
        metric = Counter(gepa_metric)
        opt = dspy.GEPA(metric=metric, max_metric_calls=budget, reflection_minibatch_size=3,
                        candidate_selection_strategy="pareto", reflection_lm=reflect_lm,
                        component_selector="round_robin", use_merge=False, num_threads=1,
                        track_stats=True, seed=seed, log_dir=str(d / "gepa_log"))
        train = [to_example(e) for e in split_data["train"]]
        val = [to_example(e) for e in split_data["val"]]
        log(f"GEPA budget {budget}, seed {seed}: {len(train)} train, {len(val)} val")
        t0 = time.time()
        compiled = opt.compile(TwoStepQA(), trainset=train, valset=val)
        sec = time.time() - t0
        save_program(compiled, d)
        dr = getattr(compiled, "detailed_results", None)
        if dr is not None:
            try:
                (d / "detailed_results.json").write_text(json.dumps(dr.to_dict(), indent=1, default=str), encoding="utf-8")
            except Exception as e:  # keep going; the candidates list is the key output
                log(f"could not dump detailed_results: {e!r}")
        meta = {"optimise_sec": sec, "metric_calls_counted": metric.n,
                "metric_calls_reported": getattr(dr, "total_metric_calls", None),
                "task_lm": usage_totals(task_lm), "reflection_lm": usage_totals(reflect_lm),
                "budget": budget, "seed": seed, "env": env_info(),
                "note": "if this run was resumed, counts cover only the final process"}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"GEPA done in {sec:.0f}s, counted {metric.n} metric calls")
    meta = json.loads(meta_path.read_text())
    prog = load_program(d)
    test_lm = make_lm(MODEL, TASK_TOKENS)
    dspy.configure(lm=test_lm)
    recs = evaluate(prog, split_data["test"], test_lm, d / "test_predictions.jsonl", limit)
    return finish_eval(d, tag, recs, extra={"optimise_calls": meta.get("metric_calls_reported") or meta["metric_calls_counted"],
                                            "optimise_sec": meta["optimise_sec"]})


def run_pilot(split_data, n=3):
    """Time n rollouts and check that replies carry no reasoning text (thinking really off)."""
    d = out_dir("pilot")
    lm = make_lm(MODEL, TASK_TOKENS)
    dspy.configure(lm=lm)
    prog = TwoStepQA()
    rows, thinking = [], 0
    for ex in split_data["train"][:n]:
        ie = to_example(ex)
        s0, t0 = len(lm.history), time.time()
        pred = prog(question=ie.question, context=ie.context)
        sec = time.time() - t0
        for h in lm.history[s0:]:
            for o in h.get("outputs", []):
                if isinstance(o, dict) and o.get("reasoning_content"):
                    thinking += 1
                elif "<think>" in (o if isinstance(o, str) else json.dumps(o)):
                    thinking += 1
        rows.append({"id": ex["id"], "sec": round(sec, 1), "answer": pred.answer, "error": pred.error,
                     **usage_totals(lm, s0)})
    rep = {"model": MODEL, "rollouts": rows, "mean_sec": sum(r["sec"] for r in rows) / len(rows),
           "replies_with_thinking": thinking, "env": env_info(),
           "verdict": "OK, thinking is off" if thinking == 0 else "THINKING DETECTED: use a non-thinking build"}
    (d / "pilot.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log(f"pilot: {rep['verdict']}, mean {rep['mean_sec']:.1f}s per rollout")
    return rep


def get_split(n_train=100, n_val=100, n_test=200):
    from .data import SPLIT_FILE, build_and_save_split
    if not SPLIT_FILE.exists():
        build_and_save_split(n_train, n_val, n_test, 42)
    return load_split()
