"""Execute notebook.ipynb cell by cell in plain Python and store the outputs in the file.

Used because jupyter nbconvert is blocked by Application Control on the HP laptop.
Opening the notebook in Jupyter or VS Code and running all cells gives the same result.
usage: python run_notebook.py [notebook.ipynb]     (RESULTS_DIR env var picks another results folder)
"""
import base64
import contextlib
import io
import sys
from pathlib import Path

import nbformat

path = sys.argv[1] if len(sys.argv) > 1 else "notebook.ipynb"
nb = nbformat.read(path, as_version=4)
shown = []


class Image:
    def __init__(self, filename):
        self.filename = filename


class Markdown:
    def __init__(self, text):
        self.text = text


def display(obj):
    shown.append(obj)


import IPython.display as mod

mod.Image, mod.Markdown, mod.display = Image, Markdown, display

ns = {"__name__": "__main__"}
count = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    count += 1
    shown.clear()
    buf = io.StringIO()
    outs = []
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(cell.source, f"<cell {count}>", "exec"), ns)
    except Exception as e:  # keep the error in the notebook so it is visible
        outs.append(nbformat.v4.new_output("error", ename=type(e).__name__, evalue=str(e), traceback=[repr(e)]))
        print("cell", count, "failed:", repr(e))
    if buf.getvalue():
        outs.insert(0, nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue()))
    for o in shown:
        if isinstance(o, Image):
            data = base64.b64encode(Path(o.filename).read_bytes()).decode()
            outs.append(nbformat.v4.new_output("display_data", data={"image/png": data, "text/plain": "<image>"}))
        elif isinstance(o, Markdown):
            outs.append(nbformat.v4.new_output("display_data", data={"text/markdown": o.text, "text/plain": o.text}))
    cell.outputs = outs
    cell.execution_count = count
nbformat.write(nb, path)
print("executed", count, "code cells ->", path)
