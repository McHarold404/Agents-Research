#!/usr/bin/env python3
"""
Deterministic split of API Gen MT dataset into Retail vs Airline samples.
Reads the final JSON file from APIGen-MT, detects domain by keyword heuristics,
and writes two JSONL files: Retail and Airline samples.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse

# Domain keyword sets
RETAIL_KEYWORDS = [
    "order", "orders", "product", "products", "cart", "checkout",
    "return", "refund", "invoice", "customer", "purchase"
]
AIRLINE_KEYWORDS = [
    "flight", "flights", "reservation", "reservations", "passenger",
    "boarding", "ticket", "airline", "airport", "departure", "arrival",
    "flight_number"
]

def _extract_text_from_sample(sample: Dict[str, Any]) -> str:
    texts: List[str] = []
    if isinstance(sample, dict):
        for key in ("system", "user_instruction"):
            if key in sample and isinstance(sample[key], str):
                texts.append(sample[key])
        for att in ("conversation", "conversations"):
            if att in sample:
                val = sample[att]
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            if "value" in item:
                                texts.append(str(item["value"]))
                            elif "from" in item and "value" in item:
                                texts.append(str(item["value"]))
                        elif isinstance(item, str):
                            texts.append(item)
                elif isinstance(val, str):
                    texts.append(val)
    return " ".join(t.lower() for t in texts if isinstance(t, str))

def _detect_domain(text: str) -> Optional[str]:
    retail_score = sum(1 for w in RETAIL_KEYWORDS if w in text)
    airline_score = sum(1 for w in AIRLINE_KEYWORDS if w in text)
    if retail_score > airline_score and retail_score > 0:
        return "retail"
    if airline_score > retail_score and airline_score > 0:
        return "airline"
    return None

def detect_domain(sample: Dict[str, Any]) -> Optional[str]:
    text = _extract_text_from_sample(sample)
    if not text:
        return None
    domain = _detect_domain(text)
    if domain:
        return domain
    convs = sample.get("conversation") or sample.get("conversations") or []
    for c in convs:
        if isinstance(c, dict) and "value" in c:
            d = _detect_domain(str(c["value"]).lower())
            if d:
                return d
    return None

def split_apigen_dataset(input_path: str, retail_out: str, airline_out: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Fallback: try to read as NDJSON (one JSON object per line)
            f.seek(0)
            lines = [line for line in f if line.strip()]
            data = []
            for line in lines:
                line = line.strip()
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue
    # If top-level data is a list with length 0, try to salvage by scanning lines as JSON objects
    if isinstance(data, list) and len(data) == 0:
        try:
            with open(input_path, "r", encoding="utf-8") as f2:
                lines = [line.strip() for line in f2 if line.strip()]
            parsed = []
            for line in lines:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and ("conversations" in obj or "conversation" in obj or "system" in obj or "user_instruction" in obj):
                    parsed.append(obj)
            if parsed:
                data = parsed
        except Exception:
            pass
    # Debug: print basic shape info to diagnose 0-sample outcomes
    if isinstance(data, list):
        print(f"[DEBUG split_apigen] top-level data is list, length={len(data)}", flush=True)
        if len(data) > 0 and isinstance(data[0], dict):
            print(f"[DEBUG split_apigen] sample0 keys={list(data[0].keys())[:5]}", flush=True)
    else:
        if isinstance(data, dict):
            print(f"[DEBUG split_apigen] top-level data is dict, keys={list(data.keys())[:5]}", flush=True)
        print(f"[DEBUG split_apigen] top-level data type={type(data).__name__}", flush=True)
    # Robust extraction of samples list from top-level JSON
    def _extract_samples_from_dict(d: dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ["data", "samples", "items", "conversations", "records"]:
            val = d.get(key)
            if isinstance(val, list) and len(val) > 0:
                return val  # type: ignore[return-value]
        return []

    samples: List[Dict[str, Any]] = (
        data if isinstance(data, list) else _extract_samples_from_dict(data)  # type: ignore[arg-type]
    )
    # Debug: show how many samples we've extracted
    if isinstance(samples, list):
        print(f"[DEBUG split_apigen] extracted samples count={len(samples)}", flush=True)
        if len(samples) > 0:
            preview = samples[0]
            print(f"[DEBUG split_apigen] first_sample_keys={list(preview.keys())[:5]}", flush=True)

    retail: List[Dict[str, Any]] = []
    airline: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []

    for sample in samples:
        domain = detect_domain(sample)
        if domain == "retail":
            sample["env"] = "retail"
            retail.append(sample)
        elif domain == "airline":
            sample["env"] = "airline"
            airline.append(sample)
        else:
            unknown.append(sample)

    # Save as JSON arrays for downstream JSON consumption
    with open(retail_out, "w", encoding="utf-8") as f_ret:
        json.dump(retail, f_ret, indent=2, ensure_ascii=False)

    with open(airline_out, "w", encoding="utf-8") as f_air:
        json.dump(airline, f_air, indent=2, ensure_ascii=False)

    # Also save to .json extension variants if requested by the user (keep existing names too)
    def _to_json_path(p: str) -> str:
        return p if p.endswith(".json") else f"{p}.json"

    retail_json_out = _to_json_path(retail_out)
    airline_json_out = _to_json_path(airline_out)

    if retail_json_out != retail_out:
        with open(retail_json_out, "w", encoding="utf-8") as f_ret_json:
            json.dump(retail, f_ret_json, indent=2, ensure_ascii=False)

    if airline_json_out != airline_out:
        with open(airline_json_out, "w", encoding="utf-8") as f_air_json:
            json.dump(airline, f_air_json, indent=2, ensure_ascii=False)

    summary = {
        "total": len(samples),
        "retail": len(retail),
        "airline": len(airline),
        "unknown": len(unknown),
    }
    Path(input_path).with_suffix(".split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Split complete:", summary)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to APIGen-MT final dataset JSON")
    ap.add_argument("--retail-out", required=True, help="Output path for Retail samples (JSONL)")
    ap.add_argument("--airline-out", required=True, help="Output path for Airline samples (JSONL)")
    args = ap.parse_args()
    split_apigen_dataset(args.input, args.retail_out, args.airline_out)

if __name__ == "__main__":
    main()

