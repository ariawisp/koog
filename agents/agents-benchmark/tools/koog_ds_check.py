#!/usr/bin/env python3
"""
koog_ds_check.py
Sanity check for Koog JSONL datasets
"""
import json
import sys
from pathlib import Path

def main(path):
    n = 0
    errors = []
    
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        try:
            j = json.loads(line)
            assert "id" in j, f"Line {i+1}: Missing 'id'"
            assert "question" in j, f"Line {i+1}: Missing 'question'"
            assert "gold_answer" in j, f"Line {i+1}: Missing 'gold_answer'"
            assert isinstance(j.get("evidence", []), list), f"Line {i+1}: 'evidence' must be a list"
            n += 1
        except Exception as e:
            errors.append(f"Line {i+1}: {e}")
    
    if errors:
        print("Errors found:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        sys.exit(1)
    else:
        print(f"✅ OK: {n} records in {path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python koog_ds_check.py <path-to-jsonl>")
        sys.exit(1)
    main(sys.argv[1])