#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
koog_ds_convert.py
Normalize multiple benchmark datasets into Koog JSONL schema.

Usage examples:
    python koog_ds_convert.py --src datasets/raw/letta --dst datasets/koog --dataset letta
    python koog_ds_convert.py --src datasets/raw/longmemeval --dst datasets/koog --dataset longmemeval
    python koog_ds_convert.py --src datasets/raw/mem0 --dst datasets/koog --dataset mem0
    python koog_ds_convert.py --src datasets/raw/kg-lm --dst datasets/koog --dataset kg-lm
"""

import argparse
import json
import uuid
from pathlib import Path

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------ MAPPERS ------

def map_letta(src: Path):
    """Map Letta benchmark format to Koog format"""
    rows = []
    for p in sorted(src.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            q = j.get("question") or j.get("message")
            a = j.get("answer") or j.get("gold_answer")
            ev = []
            
            # Optional supporting facts / files
            if "supporting_fact" in j:
                for i, fact in enumerate(j["supporting_fact"]):
                    ev.append({"text": fact, "source": f"{p.name}#fact{i}", "meta": {}})
            if "required_files" in j:
                for fn in j["required_files"]:
                    ev.append({"text": "", "source": fn, "meta": {"file_ref": True}})
                    
            rows.append({
                "id": j.get("id") or str(uuid.uuid4()),
                "dataset": "letta",
                "split": j.get("split") or "test",
                "type": j.get("benchmark_type") or j.get("question_type") or "qa",
                "question": q,
                "gold_answer": a,
                "evidence": ev,
                "meta": {
                    "reasoning_steps": j.get("reasoning_steps") or [],
                    "supporting_fact_indices": j.get("supporting_fact_indices"),
                    "contradiction": {
                        "new_fact": j.get("contradicting_fact"),
                        "new_answer": j.get("contradicting_answer"),
                    }
                }
            })
    return rows

def map_longmemeval(src: Path):
    """Map LongMemEval/LOCOMO format to Koog format"""
    rows = []
    conv_file = src / "conversations.json"
    qas_file = src / "qas.json"
    
    if not (conv_file.exists() and qas_file.exists()):
        raise FileNotFoundError("Expected conversations.json and qas.json")
        
    conversations = json.loads(conv_file.read_text(encoding="utf-8"))
    questions = json.loads(qas_file.read_text(encoding="utf-8"))
    
    # Structure assumption:
    # conversations: [{ "session_id": "...", "turns":[{"role":"user|assistant","content":"...","ts":"..."}]}]
    # questions: [{ "session_id":"...", "question":"...", "answer":"...", "type":"temporal|multi_session|..."}]
    by_session = {c["session_id"]: c for c in conversations}
    
    for q in questions:
        sess = by_session.get(q["session_id"])
        ev = []
        if sess:
            for i, t in enumerate(sess.get("turns", [])):
                ev.append({
                    "text": t.get("content", ""),
                    "source": f"{q['session_id']}#turn{i}",
                    "meta": {"role": t.get("role"), "ts": t.get("ts")}
                })
                
        rows.append({
            "id": q.get("id") or str(uuid.uuid4()),
            "dataset": "longmemeval",
            "split": q.get("split") or "test",
            "type": q.get("type") or "qa",
            "question": q["question"],
            "gold_answer": q["answer"],
            "evidence": ev,
            "meta": {"session_id": q["session_id"]}
        })
    return rows

def map_mem0(src: Path):
    """Map Mem0/LOCOMO format to Koog format"""
    rows = []
    for p in sorted(src.glob("*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        # Expected structure: {"items":[{"question":...,"answer":...,"context":[...]}]}
        items = j.get("items") or []
        for it in items:
            ev = [{"text": c, "source": p.name, "meta": {}} for c in it.get("context", [])]
            rows.append({
                "id": it.get("id") or str(uuid.uuid4()),
                "dataset": "mem0",
                "split": it.get("split") or "test",
                "type": it.get("type") or "qa",
                "question": it["question"],
                "gold_answer": it["answer"],
                "evidence": ev,
                "meta": {}
            })
    return rows

def map_kglm(src: Path):
    """Map Diffbot/Falkor KG-LM benchmark format to Koog format"""
    qf = src / "questions.json"
    if not qf.exists():
        raise FileNotFoundError("Expected questions.json")
        
    data = json.loads(qf.read_text(encoding="utf-8"))
    rows = []
    
    for q in data:
        ev = []
        for e in q.get("evidence", []):
            ev.append({
                "text": e.get("text", ""), 
                "source": e.get("source", "kg"), 
                "meta": e.get("meta", {})
            })
            
        rows.append({
            "id": q.get("id") or str(uuid.uuid4()),
            "dataset": "kg-lm",
            "split": q.get("split") or "test",
            "type": q.get("type") or "qa",
            "question": q["question"],
            "gold_answer": q["answer"],
            "evidence": ev,
            "meta": {"category": q.get("category")}
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to raw dataset directory")
    ap.add_argument("--dst", required=True, help="path to normalized output dir")
    ap.add_argument("--dataset", required=True, 
                    choices=["letta", "longmemeval", "mem0", "kg-lm"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    
    src = Path(args.src)
    dst = Path(args.dst)

    if args.dataset == "letta":
        rows = map_letta(src)
    elif args.dataset == "longmemeval":
        rows = map_longmemeval(src)
    elif args.dataset == "mem0":
        rows = map_mem0(src)
    else:
        rows = map_kglm(src)

    # Force split if provided
    for r in rows:
        r["split"] = args.split

    out = dst / f"{args.dataset}.{args.split}.jsonl"
    write_jsonl(out, rows)
    print(f"✅ Wrote {len(rows)} rows → {out}")

if __name__ == "__main__":
    main()