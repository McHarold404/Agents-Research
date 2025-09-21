#!/usr/bin/env python3
"""
Count API Gen MT samples that contain the expected fields: id, conversation, and user_instruction.
This script prints the total number of samples that match the minimal surface used by downstream parsers.
"""
import json
from typing import Any, Dict, List
import sys

def load_samples(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Try NDJSON fallback
            f.seek(0)
            lines = [line for line in f if line.strip()]
            data = [json.loads(line) for line in lines if line.strip()]
    if isinstance(data, list):
        return data
    # try common wrappers
    for key in ["data", "samples", "items"]:
        if isinstance(data, dict) and key in data and isinstance(data[key], list):
            return data[key]  # type: ignore[return-value]
    return []

def is_valid_sample(s: Dict[str, Any]) -> bool:
    if not isinstance(s, dict):
        return False
    has_id = "id" in s
    has_conv = ("conversation" in s) or ("conversations" in s) or ("conversaion" in s)
    has_instruction = ("user_instruction" in s) or ("user instruction" in s)
    return bool(has_id and has_conv and has_instruction)

def main(argv: List[str]) -> int:
    if len(argv) < 1:
        print("Usage: python datasets/count_apigen_samples.py <path_to_final_json>")
        return 2
    input_path = argv[0]
    samples = load_samples(input_path)
    total = len(samples)
    valid = [s for s in samples if is_valid_sample(s)]
    print(f"num_samples={len(valid)} of total={total}")
    if valid:
        ids = [str(s.get("id")) for s in valid[:3]]
        print("sample_ids_first3:", ",".join(ids))
    return 0

if __name__ =="__main__":
    sys.exit(main(sys.argv[1:]))


