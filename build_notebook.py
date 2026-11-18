"""Builds notebook.ipynb. The notebook only reads saved files under results/; it never calls a model."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(t):
    cells.append(nbf.v4.new_markdown_cell(t))


def code(t):
    cells.append(nbf.v4.new_code_cell(t))


md("""# Small-Scale Question Answering with GEPA: results notebook

This notebook reads the files saved under `results/` and draws the tables and graphs for Parts B and C.
It does **not** run any model. Every number comes from `results/*/test_predictions.jsonl`,
`results/*/optimise_meta.json` and `results/gepa_seed11/detailed_results.json`.

Setup: see `RUN.md` and the report (model, split sizes, GEPA budget and seed are printed below from the saved files).
""")
code("""import json, sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from IPython.display import Image, Markdown, display
import os
from src import report_assets as ra
ra.set_results(os.environ.get('RESULTS_DIR', str(ROOT / 'results')))
data = ra.load_all()
print({k: len(v['recs']) for k, v in data.items()}, 'saved test predictions')""")
md("## 1. Split and setup")
code("""split = json.loads((ROOT / 'data' / 'splits.json').read_text())
print(split['sizes'], 'seed', split['seed'])
print('disjoint:', not (set(split['train']) & set(split['val'])) and not (set(split['train']) & set(split['test'])) and not (set(split['val']) & set(split['test'])))
print(json.dumps(split['pool_stats'], indent=1))
print((ROOT / 'requirements.txt').read_text().count('\\n'), 'pinned packages; key versions:')
print(''.join(l for l in (ROOT / 'requirements.txt').read_text().splitlines(True) if l.split('==')[0] in ('dspy', 'litellm', 'gepa', 'numpy')))""")
md("## 2. Main results table (test questions, frozen prompts)")
code("""rows = ra.results_table(data)
diffs, eb = ra.write_tables(data, rows)
display(Markdown((ra.OUT / 'results_table.md').read_text()))""")
md("Paired differences in answer F1 (same test questions, bootstrap 95% interval of the mean difference):")
code("""for d in diffs:
    print(f"{d['method']} minus {d['vs']}: {d['mean_f1_diff']:+.3f}  (95% interval {d['lo']:+.3f} to {d['hi']:+.3f}, n={d['n']})")""")
md("## 3. Scores")
code("""ra.fig_scores(rows); display(Image(str(ra.OUT / 'fig_scores.png')))""")
md("## 4. Optimisation cost")
code("""ra.fig_cost([r for r in rows if r['tag'] != 'baseline']); display(Image(str(ra.OUT / 'fig_cost.png')))
for r in rows:
    print(f"{r['method']:16s} metric calls {r['opt_metric_calls']:>4}  minutes {r['opt_minutes']:6.1f}  model calls {r['opt_model_calls']:>5}  tokens {r['opt_tokens']:>9}")""")
md("## 5. GEPA validation score as candidates are found")
code("""print('drawn' if ra.fig_gepa_curve(data) else 'no GEPA detailed_results.json found')
p = ra.OUT / 'fig_gepa_curve.png'
if p.exists(): display(Image(str(p)))
c = ra.gepa_curve(data)
if c: print(list(zip(*c)))""")
md("## 6. Where do the errors come from?")
code("""ra.fig_errors(eb); display(Image(str(ra.OUT / 'fig_errors.png')))
for m, c in eb.items(): print(m, c)""")
md("## 7. The prompts before and after")
code("""for tag, label in ra.METHODS:
    p = ra.RESULTS / tag / 'prompts.json'
    if p.exists():
        pr = json.loads(p.read_text())
        print('=====', label, '| demos per module:', pr['n_demos'])
        for mod, text in pr['instructions'].items():
            print(f'--- {mod}'); print(text); print()""")
md("## 8. A few test questions side by side")
code("""base = {r['id']: r for r in data['baseline']['recs']}
g = data.get(ra.GEPA)
shown = 0
for r in (g['recs'] if g else []):
    b = base[r['id']]
    if abs(r['f1'] - b['f1']) > 0.5 and shown < 6:
        print(f"Q: {r['question']}\\n   gold: {r['gold']!r} | baseline: {b['pred']!r} (F1 {b['f1']:.2f}) | GEPA: {r['pred']!r} (F1 {r['f1']:.2f})\\n")
        shown += 1
print(shown, 'questions shown where the two differ by more than 0.5 F1')""")
nb["cells"] = cells
nbf.write(nb, "notebook.ipynb")
print("wrote notebook.ipynb")
