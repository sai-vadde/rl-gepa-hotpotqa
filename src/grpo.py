"""Optional RL baseline: GRPO with LoRA on Qwen3-0.6B (answer module only).

Stage `python run_all.py --stage grpo`. Needs: torch (CUDA), transformers, trl, peft, datasets, accelerate.
Steps (all resumable, saved under results/grpo/):
 1. Freeze the baseline selector: run the unoptimised select module with Ollama once on the train and
    test questions and save the chosen sentences (selections.jsonl).
 2. Evaluate the untrained Qwen3-0.6B as answer model on the same test IDs (start_predictions.jsonl).
 3. Train with TRL GRPOTrainer + LoRA, reward = answer F1, 200 steps, on the training questions.
 4. Evaluate the trained model on the same test IDs (trained_predictions.jsonl) and write grpo_summary.json.
"""
import json
import re
import time
from pathlib import Path

from .metrics import answer_f1_em

HF_MODEL = "Qwen/Qwen3-0.6B"
STEPS = 200

SYSTEM = "Answer the question using the evidence. Reply with a short answer only, a few words."


def prompt_for(question, evidence):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Evidenc<path> {question}\nAnswer:"}]


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.splitlines()[0].strip() if text else ""


def freeze_selections(split_data, out):
    """Run the baseline select module (Ollama) once and cache the evidence text per question."""
    import dspy
    from . import runner
    from .lm import make_lm
    from .pipeline import TwoStepQA, lookup, to_example, valid_ids
    done = {r["id"]: r for r in runner.read_jsonl(out)}
    lm = make_lm(runner.MODEL, runner.TASK_TOKENS)
    dspy.configure(lm=lm)
    prog = TwoStepQA()
    with open(out, "a", encoding="utf-8") as f:
        for ex in split_data["train"] + split_data["test"]:
            if ex["id"] in done:
                continue
            ie = to_example(ex)
            ids = []
            try:
                raw = prog.select(question=ie.question, sentences=ie.context).evidence_ids
                ok = valid_ids(ie.context)
                ids = [int(i) for i in raw if str(i).lstrip("-").isdigit() and int(i) in ok][:6]
            except Exception:
                pass
            ev = "\n".join(lookup(ie.context, ids)) or "(no evidence selected)"
            rec = {"id": ex["id"], "selected_ids": ids, "evidence": ev}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[ex["id"]] = rec
    return done


def generate(model, tok, items, max_new=32):
    import torch
    outs = []
    model.eval()
    for i in range(0, len(items), 8):
        batch = items[i:i + 8]
        texts = [tok.apply_chat_template(prompt_for(q, e), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for q, e in batch]
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        for g in gen:
            outs.append(clean(tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)))
    return outs


def score_set(model, tok, test, sel, path):
    outs = generate(model, tok, [(e["question"], sel[e["id"]]["evidence"]) for e in test])
    rows = []
    for e, o in zip(test, outs):
        f1, em = answer_f1_em(o, e["answer"])
        rows.append({"id": e["id"], "pred": o, "gold": e["answer"], "f1": f1, "em": em})
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return {"n": len(rows), "answer_f1": sum(r["f1"] for r in rows) / len(rows),
            "exact_match": sum(r["em"] for r in rows) / len(rows)}


def build_config(out_dir, steps=STEPS, fp16=True):
    from trl import GRPOConfig
    return GRPOConfig(output_dir=str(out_dir / "ckpt"), max_steps=steps, per_device_train_batch_size=8,
                      num_generations=4, max_completion_length=32, learning_rate=5e-6, fp16=fp16, bf16=False, use_cpu=not fp16,
                      logging_steps=5, save_steps=50, report_to=[], seed=11, temperature=0.9,
                      # gpu-claude 17 Nov: without this, TRL renders prompts with Qwen3 thinking ON, every 32-token
                      # completion is an unfinished <think> block, reward is 0 for all, and nothing trains (attempt 1).
                      chat_template_kwargs={"enable_thinking": False})


def main(model_tag=None, steps=STEPS):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer
    from . import runner
    sd = runner.get_split()
    d = runner.out_dir("grpo")
    sel = freeze_selections(sd, d / "selections.jsonl")
    cuda = torch.cuda.is_available()
    runner.log(f"GRPO: cuda={cuda}, model={HF_MODEL}, steps={steps}")
    dtype = torch.float16 if cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=dtype).to("cuda" if cuda else "cpu")
    summary = {"model": HF_MODEL, "steps": steps, "cuda": cuda}
    if not (d / "start_predictions.jsonl").exists():
        summary["start"] = score_set(model, tok, sd["test"], sel, d / "start_predictions.jsonl")
        runner.log(f"start model: {summary['start']}")
    gold = {e["id"]: e["answer"] for e in sd["train"]}
    rows = [{"prompt": prompt_for(e["question"], sel[e["id"]]["evidence"]), "qid": e["id"]} for e in sd["train"]]

    def reward_f1(completions, qid, **kw):
        res = []
        for c, q in zip(completions, qid):
            text = c[0]["content"] if isinstance(c, list) else c
            res.append(answer_f1_em(clean(text), gold[q])[0])
        return res

    t0 = time.time()
    trainer = GRPOTrainer(model=model, reward_funcs=reward_f1, args=build_config(d, steps, fp16=cuda),
                          train_dataset=Dataset.from_list(rows), processing_class=tok,
                          peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    last = max((Path(d / "ckpt").glob("checkpoint-*")), key=lambda p: int(p.name.split("-")[1]), default=None)
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    summary["train_sec"] = time.time() - t0
    (d / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    summary["trained"] = score_set(trainer.model, tok, sd["test"], sel, d / "trained_predictions.jsonl")
    (d / "grpo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    runner.log(f"GRPO done: {summary}")
