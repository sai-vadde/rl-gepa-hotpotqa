"""HotpotQA distractor loading and the fixed train/val/test split.

Model inputs only ever contain the question and the numbered sentences.
Gold answers and supporting facts stay in the example as hidden fields that
only the metrics read.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SPLIT_FILE = ROOT / "data" / "splits.json"
MAX_CONTEXT_TOKENS = 6000


def approx_tokens(text):
    # No tokenizer is installed; about 1.3 tokens per whitespace word is a
    # rough rule of thumb. Only used for the 6,000-token exclusion rule.
    return int(len(text.split()) * 1.3)


def build_example(rec):
    """Turn a raw HotpotQA record into a flat example with global sentence IDs."""
    sents = []
    gid = 0
    for title, sent_list in zip(rec["context"]["title"], rec["context"]["sentences"]):
        for li, s in enumerate(sent_list):
            sents.append({"id": gid, "title": title, "local": li, "text": s.strip()})
            gid += 1
    key = {(s["title"], s["local"]): s["id"] for s in sents}
    sf = rec["supporting_facts"]
    gold = sorted({key[(t, i)] for t, i in zip(sf["title"], sf["sent_id"]) if (t, i) in key})
    return {
        "id": rec["id"],
        "question": rec["question"],
        "answer": rec["answer"],
        "type": rec.get("type", ""),
        "level": rec.get("level", ""),
        "sentences": sents,
        "gold_ids": gold,
    }


def render_sentences(sentences):
    """Numbered candidate list shown to the selector."""
    return "\n".join(f"[{s['id']}] ({s['title']}) {s['text']}" for s in sentences)


def context_tokens(ex):
    return approx_tokens(render_sentences(ex["sentences"]))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def clean_pool(records):
    """Drop duplicate questions and over-long contexts; return (examples, stats)."""
    seen, out = set(), []
    stats = {"raw": 0, "duplicate": 0, "too_long": 0, "no_gold": 0}
    for rec in records:
        stats["raw"] += 1
        q = " ".join(rec["question"].lower().split())
        if q in seen:
            stats["duplicate"] += 1
            continue
        seen.add(q)
        ex = build_example(rec)
        if not ex["gold_ids"]:
            stats["no_gold"] += 1
            continue
        if context_tokens(ex) > MAX_CONTEXT_TOKENS:
            stats["too_long"] += 1
            continue
        out.append(ex)
    stats["kept"] = len(out)
    return out, stats


def make_split(train_ids_all, dev_ids_all, n_train, n_val, n_test, seed=42):
    """Deterministic, disjoint split of IDs. Train/val from the train pool, test from dev."""
    ids = sorted(set(train_ids_all))
    random.Random(seed).shuffle(ids)
    dev = sorted(set(dev_ids_all))
    random.Random(seed + 1).shuffle(dev)
    return {
        "seed": seed,
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": dev[:n_test],
    }


def build_and_save_split(n_train, n_val, n_test, seed=42, path=SPLIT_FILE):
    train_pool, tstats = clean_pool(load_jsonl(RAW / "train0.jsonl"))
    dev_pool, dstats = clean_pool(load_jsonl(RAW / "validation.jsonl"))
    split = make_split([e["id"] for e in train_pool], [e["id"] for e in dev_pool],
                       n_train, n_val, n_test, seed)
    split["sizes"] = {"train": n_train, "val": n_val, "test": n_test}
    split["pool_stats"] = {"official_train_shard0": tstats, "official_dev_distractor": dstats}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")
    save_subset(split, path.parent / ("subset_" + path.name.replace(".json", ".jsonl")))
    return split


def save_subset(split, out):
    """Keep only the raw records of the chosen IDs, so the run needs no big raw download."""
    wanted = set(split["train"]) | set(split["val"]) | set(split["test"])
    with open(out, "w", encoding="utf-8") as f:
        for fname in ("train0.jsonl", "validation.jsonl"):
            for rec in load_jsonl(RAW / fname):
                if rec["id"] in wanted:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_split(path=SPLIT_FILE):
    """Return dict name -> list of examples, in the saved ID order."""
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    wanted = set(split["train"]) | set(split["val"]) | set(split["test"])
    found = {}
    subset = Path(path).parent / ("subset_" + Path(path).name.replace(".json", ".jsonl"))
    files = [subset] if subset.exists() else [RAW / "train0.jsonl", RAW / "validation.jsonl"]
    for fp in files:
        for rec in load_jsonl(fp):
            if rec["id"] in wanted:
                found[rec["id"]] = build_example(rec)
    return {k: [found[i] for i in split[k]] for k in ("train", "val", "test")}
