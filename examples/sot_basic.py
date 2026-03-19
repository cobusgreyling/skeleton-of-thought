"""Minimal Skeleton-of-Thought example — the pattern in 40 lines.

Shows the core SoT technique without the benchmarking overhead.
Generate a skeleton. Expand in parallel. Assemble.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/sot_basic.py "Explain how DNS works"
"""

import asyncio
import sys
from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-haiku-4-5-20251001"


def skeleton(question: str) -> list[str]:
    """Generate an outline of 4-6 key points."""
    r = client.messages.create(
        model=MODEL, max_tokens=256,
        messages=[{"role": "user", "content":
            f"List 4-6 key points to answer: {question}\n"
            f"One short phrase per point. Numbered list only. No detail."}],
    )
    return [
        line.lstrip("0123456789.-) ").strip()
        for line in r.content[0].text.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]


async def expand(question: str, point: str) -> str:
    """Expand a single point into 2-3 sentences."""
    loop = asyncio.get_event_loop()
    def _call():
        r = client.messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content":
                f"Context: answering '{question}'\n"
                f"Expand into 2-3 sentences: {point}\nBe concise."}],
        )
        return r.content[0].text.strip()
    return await loop.run_in_executor(None, _call)


async def sot(question: str) -> str:
    """Full Skeleton-of-Thought pipeline."""
    points = skeleton(question)
    expansions = await asyncio.gather(*[expand(question, p) for p in points])
    return "\n\n".join(f"**{p}**\n{e}" for p, e in zip(points, expansions))


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Explain how DNS works"
    print(asyncio.run(sot(q)))
