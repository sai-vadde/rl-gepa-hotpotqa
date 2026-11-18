# Small-Scale Question Answering with GEPA

Course project for Reinforcement Learning (B.Tech CSE AI and ML, Jain University), by Vadde Sai Hrushikesh.
This repository is the implementation (Part B). The research report is Part C.



## What it does

A two-module [DSPy](https://github.com/stanfordnlp/dspy) program answers HotpotQA questions:

1. **select** reads the question and the numbered sentences of ten paragraphs and returns up to six sentence IDs.
2. A fixed lookup fetches those sentences and drops any ID that does not exist.
3. **respond** reads the question and the selected sentences and returns a short answer.

It then compares four ways to improve it, all on a local quantised Qwen3 served by Ollama:

| System | What changes | Weights |
|---|---|---|
| Baseline | nothing; my own starting instructions | fixed |
| MIPROv2 (`auto="light"`) | instructions and a few demonstrations, searched by score | fixed |
| `dspy.GEPA` | instructions, rewritten from traces plus written feedback, Pareto selection | fixed |
| GRPO + LoRA (extra) | the weights of a smaller Qwen3-0.6B answering model, reward = answer F1 | LoRA adapters |

Metrics: official HotpotQA answer F1 and exact match, plus supporting-fact F1 for the selector. GEPA also gets a feedback function (`src/metrics.py`) that names the missed or surplus sentences for the selector and says whether the answer matched for the answerer.

## How this matches the proposal

The Part A proposal ("Small-Scale Question Answering with GEPA", 12 November 2026) is in [`docs/PartA_proposal.pdf`](docs/PartA_proposal.pdf) with its LaTeX source.
The implementation follows it:
two modules with a fixed lookup, a 100/100/200 split with seed 42, GEPA with 900 metric calls and seed 11 (reflection batches of 3, round-robin module updates, Pareto selection, merge off), MIPROv2 light with two bootstrapped and two labelled demonstrations, answer F1 as the optimised metric, the same model for the task and for reflection, and the optional GRPO/LoRA run on Qwen3-0.6B.
Where a setting differs from the proposal it is listed in `RUN.md` (for example, the context window is 12,288 tokens because GEPA reflection prompts hold three full examples, and the model tag depends on which Qwen3 build Ollama offers). HoVer, which the proposal marked optional, is not implemented.

## Dataset

- **Source:** HotpotQA, distractor setting (Yang et al., EMNLP 2018), from the Hugging Face copy of the official files: https://hotpotqa.github.io/. The data is released under CC BY-SA 4.0 (check the licence on the dataset page before reuse). It is not stored in this repository except for the 400 chosen questions.
- **Filtering:** exact duplicate questions are dropped, and any context over 6,000 estimated tokens would be (none was). Token counts are estimated as 1.3 per word.
- **Split (seed 42):** 100 training and 100 validation questions from the first parquet shard of the official train file; 200 test questions from the official distractor development file. The three sets are disjoint (a test checks this).
- **Saved IDs:** `data/splits.json` holds the question IDs and the pool statistics. `data/subset_splits.jsonl` holds the raw records of those 400 questions, so no download is needed to reproduce.
- Reference answers and supporting-sentence marks never enter the model input. Only training questions give feedback text to GEPA. Test questions are scored after the prompts are frozen.

## Install

You need Python 3.11 to 3.13 and [Ollama](https://ollama.com/).

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install trl peft transformers datasets accelerate torch   # only for the GRPO stage
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama cp qwen3:4b-instruct-2507-q4_K_M qwen3:4b-instruct-2507
.venv\Scripts\python -m pytest -q tests
```

macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install trl peft transformers datasets accelerate torch   # only for the GRPO stage
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama cp qwen3:4b-instruct-2507-q4_K_M qwen3:4b-instruct-2507
.venv/bin/python -m pytest -q tests
```

For a CUDA GPU, install torch from the CUDA wheel index first (see `RUN.md`). Use `python -m ...` rather than launcher executables if your system blocks them.

## Run

The endpoint and model are settings: `--model` (Ollama tag) and `--api-base` (or the environment variable `OLLAMA_HOST_URL`, default `http://localhost:11434`).

```powershell
$M = "qwen3:4b-instruct-2507"
.venv\Scripts\python run_all.py --stage pilot    --model $M     # 3 timed rollouts, checks thinking is off
.venv\Scripts\python run_all.py --stage split                   # writes data/splits.json (already included)
.venv\Scripts\python run_all.py --stage baseline --model $M
.venv\Scripts\python run_all.py --stage mipro    --model $M
.venv\Scripts\python run_all.py --stage gepa     --model $M --budget 900 --seed 11
.venv\Scripts\python run_all.py --stage grpo     --model $M     # needs a CUDA GPU
.venv\Scripts\python run_all.py --stage tables                  # tables and graphs from saved results
```

`run_chain.ps1` runs all of these stages in order, unattended, and writes `logs\chain_status.txt`. Every stage can be resumed after an interruption and writes to `results/`. On macOS or Linux replace `.venv\Scripts\python` with `.venv/bin/python`.

## Reproduce the results

1. Install as above and pull the model.
2. Run the stages in the order shown. Use the same `--model` for all of them.
3. Run `--stage tables`. It rewrites `results/report/` (tables and figures) from the saved predictions only.
4. `python demo.py` replays one question from the saved results; `python demo.py --live` runs it through Ollama.
5. `notebook.ipynb` draws the same tables and graphs from `results/` and runs no model.

The GPU environment of the reported run (machine, driver, versions, stage times) is in `RUN_NOTES.md` and `run_logs/`. The trained LoRA adapter of the GRPO run is in `results/grpo/lora_adapter/` (about 9 MB); to regenerate it, run the `grpo` stage.

Exact numbers can differ a little between machines, because sampling is not bit-identical across hardware. The saved predictions in `results/` are the ones the report uses.

## Results

| Method | N | Answer F1 (95% CI) | EM | SP-F1 | Opt. metric calls | Opt. minutes | Test s/question |
|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.659 (0.605 to 0.715) | 0.495 | 0.701 | 0 | 0 | 1.2 |
| MIPROv2 light | 200 | 0.703 (0.647 to 0.758) | 0.580 | 0.632 | 1208 | 27 | 1.2 |
| GEPA (seed 11) | 200 | 0.724 (0.672 to 0.780) | 0.605 | 0.695 | 956 | 21 | 1.1 |


Separate weight-update run, Qwen3-0.6B with GRPO and LoRA (a different model on the same test questions, not comparable with the table above):

| Qwen3-0.6B answer model | Answer F1 | EM |
|---|---|---|
| Before GRPO | 0.461 | 0.320 |
| After 200 steps | 0.524 | 0.380 |


`results/grpo_attempt1_thinking_on/` is a failed first GRPO attempt kept on purpose (thinking was on, so the reward was zero at every step; see `RUN_NOTES.md`).

Scores are on the 200 held-out test questions with frozen prompts. The interval is a bootstrap 95% interval over questions. Details, graphs, the GRPO table and the failure analysis are in `results/report/` and in the report.

## Demo

```powershell
.venv\Scripts\python demo.py            # replay saved predictions for one question, baseline vs GEPA prompt
.venv\Scripts\python demo.py --live     # same, live through Ollama
```

`DEMO.md` has the steps and viva-style questions.

## Layout

| Path | Purpose |
|---|---|
| `run_all.py` | entry point, one flag per stage |
| `src/data.py` | HotpotQA loading, sentence IDs, deterministic split |
| `src/pipeline.py` | the two modules and the ID lookup |
| `src/metrics.py` | F1, EM, supporting-fact F1, GEPA feedback function |
| `src/runner.py` | evaluation, MIPROv2, GEPA, pilot |
| `src/grpo.py` | GRPO + LoRA on Qwen3-0.6B |
| `src/lm.py` | Ollama client settings (thinking off) |
| `src/report_assets.py`, `src/report_tex.py` | tables and figures from saved results |
| `tests/` | split, metrics and feedback tests |
| `data/` | split IDs and the 400 chosen questions |
| `results/` | prompts, predictions, summaries, figures |
| `docs/` | the Part A proposal |
| `run_chain.ps1`, `RUN_NOTES.md`, `run_logs/` | the unattended stage chain and the environment of the reported run |
| `notebook.ipynb`, `demo.py`, `DEMO.md` | notebook and demo |
| `RUN.md` | detailed run notes and known issues |

## Licence

MIT, see `LICENSE`. HotpotQA keeps its own licence.

## References

GEPA: Agrawal et al., ICLR 2026. DSPy: Khattab et al., ICLR 2024. MIPROv2: Opsahl-Ong et al., EMNLP 2024. HotpotQA: Yang et al., EMNLP 2018. Qwen3 Technical Report, 2025. GRPO: Shao et al., 2024. LoRA: Hu et al., ICLR 2022. TRL: von Werra et al., 2020.
