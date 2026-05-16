"""Simple smoke test to verify the app imports and dataset loader.

Run locally with: python scripts/smoke_test.py
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running from the scripts/ folder
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from quiz_engine import load_clapnq_sample


def main():
    print("Running smoke test: loading 1 sample question from dataset (non-LLM path)")
    items = load_clapnq_sample(split="train", max_items=1, randomize=True, use_llm=False)
    print("Loaded", len(items), "items")
    if len(items) == 0:
        raise SystemExit("Smoke test failed: no quiz items returned")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
