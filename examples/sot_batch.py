"""Batch SoT evaluation — run multiple questions and compare latencies.

Runs a set of questions through both CoT and SoT, then prints
a summary table showing the speedup for each.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/sot_batch.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sot_vs_cot import run_cot, run_sot

QUESTIONS = [
    "What are the pros and cons of serverless architecture?",
    "Explain how OAuth 2.0 authentication works.",
    "What is the CAP theorem and why does it matter?",
    "How does garbage collection work in modern programming languages?",
    "What are the key principles of API design?",
]


def main():
    print(f"{'Question':<55} {'CoT':>7} {'SoT':>7} {'Speedup':>8}")
    print("-" * 80)

    total_cot = 0
    total_sot = 0

    for q in QUESTIONS:
        _, cot_time = run_cot(q)
        _, sot_time, _ = run_sot(q)
        speedup = cot_time / sot_time if sot_time > 0 else 0
        total_cot += cot_time
        total_sot += sot_time

        label = q[:52] + "..." if len(q) > 55 else q
        print(f"{label:<55} {cot_time:>6.2f}s {sot_time:>6.2f}s {speedup:>7.1f}x")

    print("-" * 80)
    avg_speedup = total_cot / total_sot if total_sot > 0 else 0
    print(f"{'TOTAL':<55} {total_cot:>6.2f}s {total_sot:>6.2f}s {avg_speedup:>7.1f}x")


if __name__ == "__main__":
    main()
