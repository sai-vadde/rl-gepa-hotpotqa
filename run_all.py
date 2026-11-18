"""Entry file. Runs the experiment stages in order; every stage is resumable.

python run_all.py --stage pilot --model qwen3:4b-instruct-2507
python run_all.py --stage all --model qwen3:4b-instruct-2507   # split, baseline, mipro, gepa, then tables
python run_all.py --stage gepa --budget 900 --seed 11
python run_all.py --stage tables         # tables and graphs from saved results only
python run_all.py --smoke                # tiny end-to-end check into results_smoke/
"""
import argparse
import sys

from src import runner
from src.data import SPLIT_FILE, build_and_save_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["pilot", "split", "baseline", "mipro", "gepa", "grpo", "tables", "all"])
    ap.add_argument("--budget", type=int, default=900, help="GEPA max_metric_calls")
    ap.add_argument("--seed", type=int, default=runner.GEPA_SEED, help="GEPA seed")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--model", default="qwen3:4b", help="Ollama model tag, e.g. qwen3:4b-instruct-2507")
    ap.add_argument("--api-base", default=None, help="Ollama endpoint (default env OLLAMA_HOST_URL or htt<path>)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    runner.MODEL = a.model
    if a.api_base:
        from src import lm as _lm
        _lm.API_BASE = a.api_base

    if a.smoke:
        runner.RESULTS = runner.RESULTS.parent / "results_smoke"
        a.n_train, a.n_val, a.n_test = 3, 3, 2
        import src.data as data
        data.SPLIT_FILE = data.ROOT / "data" / "splits_smoke.json"
        build_and_save_split(3, 3, 2, 42, data.SPLIT_FILE)
        sd = data.load_split(data.SPLIT_FILE)
        runner.run_baseline(sd)
        runner.run_gepa(sd, budget=12, seed=1)
        return

    if a.stage in ("split", "all"):
        if not SPLIT_FILE.exists():
            s = build_and_save_split(a.n_train, a.n_val, a.n_test, 42)
            print("split saved:", s["sizes"])
        elif a.stage == "split":
            print("split already exists:", SPLIT_FILE)
    sd = runner.get_split(a.n_train, a.n_val, a.n_test)
    if a.stage == "pilot":
        runner.run_pilot(sd)
    if a.stage in ("baseline", "all"):
        runner.run_baseline(sd)
    if a.stage in ("mipro", "all"):
        runner.run_mipro(sd)
    if a.stage in ("gepa", "all"):
        runner.run_gepa(sd, budget=a.budget, seed=a.seed)
    if a.stage in ("grpo",):  # not part of "all": needs a CUDA GPU and the extra packages
        from src import grpo
        grpo.main(model_tag=a.model)
    if a.stage in ("tables", "all"):
        from src import report_assets
        report_assets.main()


if __name__ == "__main__":
    sys.exit(main())
