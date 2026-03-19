"""Skeleton-of-Thought vs Chain-of-Thought — head-to-head comparison.

Sends the same question using both prompting strategies and measures
latency and output structure. SoT generates an outline first, then
expands each point in parallel API calls.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/sot_vs_cot.py

    # Custom question:
    python examples/sot_vs_cot.py "Explain how a database index works"
"""

import asyncio
import time
import sys

from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-haiku-4-5-20251001"


# ── Chain-of-Thought (sequential) ────────────────────────────────


def run_cot(question: str) -> tuple[str, float]:
    """Standard Chain-of-Thought: one sequential call."""
    start = time.perf_counter()

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Answer the following question step by step. "
                f"Think through each step carefully before moving to the next.\n\n"
                f"Question: {question}"
            ),
        }],
    )

    elapsed = time.perf_counter() - start
    return response.content[0].text, elapsed


# ── Skeleton-of-Thought (outline + parallel expand) ─────────────


def generate_skeleton(question: str) -> list[str]:
    """Step 1: Generate a skeleton (outline) of the answer."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"I need a concise answer to this question: {question}\n\n"
                f"First, give me ONLY the skeleton — a numbered list of 4-6 key points "
                f"that the answer should cover. One short phrase per point. "
                f"No explanations, no detail, just the outline."
            ),
        }],
    )

    text = response.content[0].text
    points = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # Strip the number/bullet prefix
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned:
                points.append(cleaned)
    return points


async def expand_point_async(question: str, point: str, index: int) -> tuple[int, str]:
    """Step 2: Expand a single skeleton point (runs in parallel)."""
    loop = asyncio.get_event_loop()

    def _call():
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    f"Context: answering the question '{question}'\n\n"
                    f"Expand this single point into 2-3 clear sentences: {point}\n\n"
                    f"Be concise and specific. No preamble."
                ),
            }],
        )
        return response.content[0].text

    text = await loop.run_in_executor(None, _call)
    return index, text


async def run_sot_async(question: str) -> tuple[str, float, int]:
    """Full SoT pipeline: skeleton + parallel expansion."""
    start = time.perf_counter()

    # Step 1: Generate skeleton
    skeleton = generate_skeleton(question)

    # Step 2: Expand all points in parallel
    tasks = [
        expand_point_async(question, point, i)
        for i, point in enumerate(skeleton)
    ]
    expansions = await asyncio.gather(*tasks)
    expansions_sorted = sorted(expansions, key=lambda x: x[0])

    # Assemble
    parts = []
    for i, (idx, text) in enumerate(expansions_sorted):
        parts.append(f"**{skeleton[idx]}**\n{text}")

    elapsed = time.perf_counter() - start
    return "\n\n".join(parts), elapsed, len(skeleton)


def run_sot(question: str) -> tuple[str, float, int]:
    """Synchronous wrapper for SoT."""
    return asyncio.run(run_sot_async(question))


# ── Main ─────────────────────────────────────────────────────────


def main():
    question = (
        sys.argv[1] if len(sys.argv) > 1
        else "What are the key differences between microservices and monolithic architecture?"
    )

    print(f"Question: {question}")
    print(f"Model:    {MODEL}")
    print(f"{'=' * 60}\n")

    # Run CoT
    print("CHAIN-OF-THOUGHT (sequential)")
    print("-" * 40)
    cot_answer, cot_time = run_cot(question)
    print(cot_answer)
    print(f"\nLatency: {cot_time:.2f}s")

    print(f"\n{'=' * 60}\n")

    # Run SoT
    print("SKELETON-OF-THOUGHT (outline + parallel)")
    print("-" * 40)
    sot_answer, sot_time, num_points = run_sot(question)
    print(sot_answer)
    print(f"\nLatency: {sot_time:.2f}s ({num_points} points expanded in parallel)")

    print(f"\n{'=' * 60}\n")

    # Compare
    speedup = cot_time / sot_time if sot_time > 0 else 0
    print(f"COMPARISON")
    print(f"  CoT latency:  {cot_time:.2f}s")
    print(f"  SoT latency:  {sot_time:.2f}s")
    print(f"  Speedup:      {speedup:.1f}x")
    print(f"  CoT tokens:   ~{len(cot_answer.split())} words")
    print(f"  SoT tokens:   ~{len(sot_answer.split())} words")


if __name__ == "__main__":
    main()
