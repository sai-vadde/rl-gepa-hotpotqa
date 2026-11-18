"""One-off helper: HF parquet -> JSONL (run with any Python that has a working pyarrow).

On the HP laptop the project venv's pyarrow parquet DLL is blocked by Windows
Application Control, so this conversion ran in a separate scratch venv
(pyarrow 21). The rest of the project reads only the JSONL files.
usage: python convert_parquet.py in.parquet out.jsonl
"""
import json, sys
import pyarrow.parquet as pq

t = pq.read_table(sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for r in t.to_pylist():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(t.num_rows, "rows ->", sys.argv[2])
