# RUN_NOTES (gpu-claude, borrowed laptop)

## Machine
- Lenovo LOQ 15AHP10: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, driver 595.95; AMD Ryzen 7 250 (8C/16T); 15.3 GB RAM; Windows 11 Home build 26300 (native, no WSL).
- Ollama 0.40.16 (per-user install via winget), models on W:.
- Python 3.12.10, venv `<path>`; full list in `logs/pip_freeze.txt`.
  Key versions: dspy 3.5.0, gepa 0.1.4, litellm 1.77.7, numpy 2.3.5, optuna 5.1.0, torch 2.11.0+cu128, transformers 5.23.0, trl 1.18.0, peft 0.21.3.

## Model
- `--model qwen3:4b-instruct-2507`. The bare Ollama tag `qwen3:4b-instruct-2507` does not exist in the Ollama library; the model is
  `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB, Q4_K_M), copied to the name `qwen3:4b-instruct-2507` with `ollama cp`. Same model for every method.
- Pilot: "OK, thinking is off", mean 5.0 s per rollout (3 timed rollouts, includes first load).

## Changes from the HP code / RUN.md
1. `optuna` added to the venv (pip, 5.1.0): MIPROv2 in dspy 3.5.0 needs it and `requirements.txt` did not list it.
   The first mipro attempt stopped at import (no model calls); the stage was re-run from the start.
2. Stages are chained by `run_chain.ps1` (same order and arguments as RUN.md, `--api-base htt<path>`).
No code in `src/` was changed for baseline / mipro / gepa.

## GRPO dry run (before the real stage; separate folder `../grpo_dry/`, not results/)
`grpo_dry_run.py`: src/grpo.py's own `build_config` (fp16, LoRA r=16 q/v), prompt and F1 reward, 2 steps on 16 train
questions with placeholder evidence (no Ollama calls). Result: OK with no code changes; Qwen3-0.6B loads on CUDA fp16 in 5 s,
`generate()` works, GRPOTrainer (trl 1.18.0) builds and trains, 5.5 s/step, peak 2.0 GB VRAM while Ollama (GEPA) held ~3 GB.

## Stage log (from logs/chain_status.txt)
- baseline: 4.1 min, test answer F1 0.6589, supporting-fact F1 0.7015, 1.20 s/rollout.
- mipro (auto=light): 31.5 min, test answer F1 0.7031, supporting-fact F1 0.6325, 1.24 s/rollout.
- gepa seed 11 (budget 900): 24.5 min stage (21 min optimising), test answer F1 0.7243, SP-F1 0.6953, 1.14 s/rollout;
  metric calls reported 956 (counted 1022 incl. feedback calls), 1936 task-LM calls, 22 reflection calls.
- tables: see logs/tables.log (Baseline 0.659 / MIPROv2 light 0.703 / GEPA 0.724 answer F1, with 95% CIs).

## GRPO
- Attempt 1 (kept as `results/grpo_attempt1_thinking_on/`, 17.7 min): trained nothing. Every logged step had reward 0,
  reward_std 0, frac_reward_zero_std 1.0, completions always 32 tokens (clipped); trained predictions identical to the start
  model on 200/200 test questions (answer F1 0.4614 both). Cause: TRL rendered the training prompts with Qwen3's default
  chat template, thinking ON, so each 32-token completion was an unfinished `<think>` block that scored 0. (The start/trained
  evaluation in `generate()` already passed `enable_thinking=False`, which is why those scores were non-zero.)
- Fix (config only, `src/grpo.py` `build_config`): `chat_template_kwargs={"enable_thinking": False}`. A 2-step dry run with the
  fix gave mean completion 17.8 tokens and non-zero reward with spread.
- Attempt 2 (`results/grpo/`): same steps (200), seed, LoRA and reward; reuses attempt 1's frozen evidence
  `selections.jsonl` (identical Ollama selector output, so no new selector calls).
  Result (200 test questions, greedy, same evidence): start model answer F1 0.4614 / EM 0.320 -> trained 0.5237 / EM 0.380;
  89/200 predictions changed. Training: 200 steps, 273.5 s (5.5 min stage), mean train reward 0.44 at step 5 -> 0.87 at step 185,
  completions 5-10 tokens. GRPO trains only the answer step of a 0.6B model on fixed evidence, so it is not on the same
  footing as the 4B prompt-optimisation rows (different model, evidence frozen from the baseline selector).

## What is in transfer/gepa-results.tar.gz
`results/` without the GRPO trainer checkpoints (optimizer states, ~300 MB, kept on the GPU laptop), except attempt 2's final
LoRA adapter `results/grpo/ckpt/checkpoint-200/adapter_model.safetensors` + `adapter_config.json` (8.8 MB);
`logs/` (all stage logs, chain_status.txt, pip_freeze.txt, grpo_dry_run logs); `RUN_NOTES.md`; `run_chain.ps1`;
`grpo_dry_run.py`; the changed `src/grpo.py`.

## Final stage times (logs/chain_status.txt)
pilot 5.0 s/rollout | baseline 4.1 min | mipro 31.5 min | gepa 24.5 min | grpo attempt 1 17.7 min (incl. 300 selector calls) |
grpo attempt 2 5.5 min | tables 0.1 min. Whole run ~1.5 h of GPU time.
