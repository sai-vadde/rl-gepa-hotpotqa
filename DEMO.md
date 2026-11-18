# Demo and viva notes

## Simple demo (about two minutes, works without a model)

1. Open a terminal in the project folder.
2. Run:
   ```
   .venv\Scripts\python demo.py
   ```
3. It picks one test question where the GEPA prompt got the answer right and the baseline did not (or the first question if there is none).
   It prints the question, the instructions each method used for the **select** and **respond** modules, the sentence IDs each chose against the marked supporting sentences, and the answers with their F1.
   Everything is replayed from `results/*/test_predictions.jsonl` and `results/*/prompts.json`, so it cannot fail on stage.
4. Optional live version (needs Ollama running with the model from `RUN.md`; set `GEPA_MODEL` if it is not `qwen3:4b`):
   ```
   .venv\Scripts\python demo.py --live
   ```
   It loads the two frozen programs (`results/baseline/program.json`, `results/gepa_seed11/program.json`) and runs this one question through both.
5. To show another question: `demo.py --pick 17`.

What to say while it runs: the weights are the same in both runs. Only the instruction text changed, and I can read what changed.

## Viva questions and answers

### Method (answers do not depend on the results)

**What changes in GEPA?** The instructions of each module. The model weights stay fixed.

**What is a rollout?** One full pipeline run on one question, plus scoring it. Here that is two model calls: select, then respond.

**What is a metric call?** One scoring of one program run on one example. GEPA's budget is counted in these. Reflection calls are extra and counted separately.

**Why two modules?** The selector output shows whether evidence selection or answering failed. One big prompt would hide that.

**Why HotpotQA?** Questions need facts from two paragraphs, and the data marks the supporting sentences, so I can score the selector on its own.

**What is the distractor setting?** Each question comes with ten paragraphs, two of them relevant. No retrieval is needed.

**What does GEPA do in one loop?** Run the current program on a few training questions. Collect what each module saw and produced, with feedback text. Ask a reflection model to rewrite one module's instruction. Try it on the same few questions. If it is better, score it on the validation set and add it to the pool. Pick the next parent and repeat until the budget is gone.

**What is Pareto selection?** Collect the candidates that are best on at least one validation question, drop the dominated ones, and sample among the rest, more often for candidates that lead on more questions. The best average candidate is not always chosen.

**How does MIPROv2 differ?** It searches over instruction candidates and a few demonstration examples for each module, using scores only. It does not read written feedback.

**What feedback did I give GEPA?** The answer F1 as the score, plus a sentence: which needed sentences the selector missed (with their text), which chosen ones were not needed, or whether the answer matched, partly matched or missed and who was at fault. Only training questions produce feedback. Validation and test text never reach the reflection model.

**How does this relate to RL?** GRPO changes the weights using rewards. GEPA treats the answer F1 as a reward too, but it changes the prompt using a written account of what went wrong, so it needs fewer rollouts in the paper. The small GRPO run here uses the same F1 as its reward.

**What does answer F1 measure?** Overlap of words between my answer and the reference, balancing precision and recall, after the official normalisation.

**What is supporting-fact F1?** The same idea over sentence IDs: how well the selector found the marked supporting sentences.

**Why bootstrap intervals?** The test set is small. The interval shows how much the mean could move if I had drawn different questions.

**What stops the test set leaking?** The IDs are saved. Train and validation come from the official train file and the test set from the official dev file. Gold answers and supporting sentences never enter the model input. Test text never reaches the optimiser.

**Why did the 4B model not run as planned on my laptop?** The installed `qwen3:4b` tag thinks before every reply, even with thinking switched off. See `RUN.md`. The full runs used the machine and model named in the report.

**What are the limits?** Small validation and test sets, one GEPA seed unless more were run, the model reflects on its own mistakes, quantisation, no retrieval.

### Results-based (numbers from `results/report/` and the saved files, 200 test questions, Qwen3-4B-Instruct-2507)

**What were the three test F1 scores?** Baseline 0.659 (interval 0.605 to 0.715), MIPROv2 light 0.703 (0.647 to 0.758), GEPA 0.724 (0.672 to 0.780). Exact match was 0.495, 0.580 and 0.605.

**Did GEPA beat the baseline?** Yes, on these questions. The paired difference is +0.065 with a 95% interval of 0.028 to 0.101, which stays above zero.

**Did GEPA beat MIPROv2?** Not clearly. The difference is +0.021 and the interval is -0.016 to 0.062, which crosses zero. With one seed and 200 questions I cannot say the gap is real. MIPROv2 over the baseline is +0.044 with an interval of -0.003 to 0.092, which just touches zero.

**What did GEPA change?** Only the instruction of the answer module. It told the model to return only the name or fact in the reference style, with no explanation. The select instruction was left as I wrote it. This is in `results/gepa_seed11/prompts.json`.

**Why did supporting-fact F1 not move for GEPA but drop for MIPROv2?** The selector prompt is unchanged in GEPA, so it stayed at 0.695 against 0.701. MIPROv2 changed the selector's instruction and demonstrations and its supporting-fact F1 fell to 0.632, while its exact match rose. It seems to have improved the answerer at the selector's expense.

**Did the validation gain carry to test?** Validation answer F1 went from 0.730 to 0.804 across 8 candidates. Test went from 0.659 to 0.724. The test gain is smaller, as expected, because GEPA chose among candidates on the validation questions.

**What did the optimisation cost?** MIPROv2 used 1,208 metric calls and 27 minutes. GEPA reported 956 metric calls (1,022 if every feedback call is counted) and 21 minutes, with about 2.6 million tokens against about 5.4 million for MIPROv2.

**Where do the remaining errors come from?** Mostly the selector. Among the 50 wrong GEPA answers, 36 had a needed sentence missing from the selection and 14 had the evidence but the answer was wrong. I read 20 of them by hand: 13 missed evidence, 3 wrong reasoning, 3 format problems, 1 odd reference.

**What did GRPO do?** On Qwen3-0.6B with LoRA, 200 steps, reward = answer F1, same frozen evidence: answer F1 went from 0.461 to 0.524 and exact match from 0.32 to 0.38. 89 answers changed, 32 got better and 11 got worse. It is a different, smaller model, so it is not a head-to-head with the 4B prompt results.

**What went wrong with the first GRPO attempt?** Thinking mode was on in TRL's chat template, so each 32-token completion was spent inside the thinking block, the answer was empty and the reward was 0 at every step. Nothing was learned (F1 0.461 before and after). Setting `enable_thinking=False` fixed it.

**What would I run next?** More GEPA seeds and a larger validation set, a feedback message for the selector that asks it to look for the second entity, GRPO on the main 4B model, and a retrieval step.
