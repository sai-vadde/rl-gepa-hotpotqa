# How to run the full experiment (Part A plan, unchanged)

Plan: HotpotQA distractor, 100 train / 100 validation / 200 test questions (seed 42, IDs already saved in `data/splits.json`),
unoptimised baseline vs MIPROv2 `auto="light"` vs `dspy.GEPA` (900 metric calls, seed 11), Qwen3-4B on Ollama, then the
optional GRPO/LoRA run on Qwen3-0.6B. Every stage is resumable and writes to `results/` as it goes; logs go to `logs/`.
The raw HotpotQA download is not needed: `data/subset_splits.jsonl` holds the 400 chosen questions.

## Known issue: the `qwen3:4b` tag may think anyway

On the HP laptop, the installed `qwen3:4b` tag (262,144-token context, the 2507 "thinking" build) reasons before every reply.
Ollama's `think:false` and a `/no_think` prefix did not stop it. `qwen3:1.7b` obeys `think:false`. Part A needs non-thinking mode, so:

1. Try the non-thinking build first (the bare tag does not exist in the Ollama library; the working name is `qwen3:4b-instruct-2507-q4_K_M`, copied to `qwen3:4b-instruct-2507` with `ollama cp`; this is what the GPU run used, and its pilot printed `OK, thinking is off`): `ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama cp qwen3:4b-instruct-2507-q4_K_M qwen3:4b-instruct-2507`, then pass `--model qwen3:4b-instruct-2507`.
2. Run the pilot stage (3 timed rollouts). It reports `OK, thinking is off` or `THINKING DETECTED`.
3. If only the thinking build works, use `--model qwen3:4b` and say so in the report; with `max_tokens=256` the reasoning would truncate replies, so raise `TASK_TOKENS` in `src/runner.py` (for example 1024) and expect a much slower run.
   Whatever model is used, the same model must be used for all methods. The model name is saved in every `optimise_meta.json` / `test_summary.json`.

Timing from the HP laptop (CPU only, qwen3:1.7b): about 20 s per rollout, so the full plan was out of reach there. A GPU should be far faster.

## Setup, Windows PowerShell

```powershell
cd <project folder>
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install --upgrade --index-url http<path> torch
.venv\Scripts\python -m pip install trl peft transformers datasets accelerate
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama cp qwen3:4b-instruct-2507-q4_K_M qwen3:4b-instruct-2507
.venv\Scripts\python -m pytest -q tests
```

If a compiled package is blocked by Application Control (it happened on the HP: newer numpy, newer litellm, pyarrow parquet),
use the pinned versions in `requirements.txt` (numpy 2.3.5, litellm 1.77.7). `requirements.txt` is the short list (`requirements-hp-freeze.txt` is the full HP freeze with a CPU torch);
install torch with the CUDA line above, not from the freeze.
Always call `.venv\Scripts\python -m ...`, not the launcher `.exe` files.

## Setup, macOS

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt  
.venv/bin/python -m pip install trl peft transformers datasets accelerate torch
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama cp qwen3:4b-instruct-2507-q4_K_M qwen3:4b-instruct-2507
.venv/bin/python -m pytest -q tests
```

## Run, stage order

The endpoint is a setting: `--api-base htt<path>` (or env `OLLAMA_HOST_URL`). Add `--model qwen3:4b-instruct-2507`.

```powershell
$M = "qwen3:4b-instruct-2507"
.venv\Scripts\python run_all.py --stage pilot    --model $M
.venv\Scripts\python run_all.py --stage split
.venv\Scripts\python run_all.py --stage baseline --model $M *> logs\baseline.log
.venv\Scripts\python run_all.py --stage mipro    --model $M *> logs\mipro.log
.venv\Scripts\python run_all.py --stage gepa     --model $M --budget 900 --seed 11 *> logs\gepa_seed11.log
.venv\Scripts\python run_all.py --stage grpo     --model $M *> logs\grpo.log
.venv\Scripts\python run_all.py --stage tables
```

(`mkdir logs` first. `--stage all` runs split, baseline, mipro, gepa and tables, but not pilot or grpo.)
Re-running a stage resumes it: test predictions skip finished IDs, GEPA resumes from `results/gepa_seed11/gepa_log`,
and finished optimisers are not repeated.
Optional extra seeds: `--stage gepa --seed 22` and `--seed 33` write to `results/gepa_seed22/`, `gepa_seed33/`.

## What comes back (copy the whole `results/` folder, plus `logs/`)

- `results/baseline`, `results/mipro`, `results/gepa_seed11`: `program.json`, `prompts.json`, `test_predictions.jsonl`, `test_summary.json`, `optimise_meta.json` (calls, tokens, seconds), GEPA also `detailed_results.json`.
- `results/grpo`: `start_predictions.jsonl`, `trained_predictions.jsonl`, `grpo_summary.json`, `train_log.json`.
- `results/pilot/pilot.json`.
- Also note the GPU model, driver, `ollama --version`, and `pip freeze` in a short `RUN_NOTES.md`.

## Things to know

- GEPA's reported `total_metric_calls` can end slightly above 900 (its last batch crosses the limit). Report the reported number, and the extra feedback-metric calls counted separately in `optimise_meta.json`.
- Context window is 12,288 for all calls (reflection prompts hold three full examples). Part A said 8,192; the report should say 12,288.
- Training pool: the first of two parquet shards of the official HotpotQA distractor train file (45,224 of 90,447 questions), as downloaded from Hugging Face. Test comes from the official distractor dev file.
- `python demo.py` replays saved predictions and needs no model, once `results/` is back in place.
- GRPO stage was only checked to import and build its config on CPU (1 step config). It has not trained anything yet. TRL versions change often; if `GRPOTrainer` argument names differ, adjust `src/grpo.py`.

- MIPROv2 in DSPy 3.5.0 imports `optuna`; it is in `requirements.txt` (the GPU run used 5.1.0).
