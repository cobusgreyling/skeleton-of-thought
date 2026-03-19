![Skeleton-of-Thought Latency](images/sot-latency.png)

# Skeleton-of-Thought Prompting — Outline First, Expand in Parallel

**Chain-of-Thought prompting made LLMs better at reasoning. Skeleton-of-Thought prompting makes them faster at it. The idea is simple — generate an outline first, then expand each point in parallel.**

## The Sequential Bottleneck

Chain-of-Thought (CoT) prompting works by asking the model to reason step by step. Each step depends on the previous one. The token generation is sequential by design.

For short answers this is fine. For complex problems with 5, 10 or 15 reasoning steps, the latency adds up linearly. Step 7 cannot start until step 6 completes. The model sits idle on every step except the one it is currently generating.

This is the sequential token bottleneck. It is a structural limitation of how CoT works, not a hardware limitation.

## The SoT Alternative

Skeleton-of-Thought (SoT) prompting splits the work into two phases.

**Phase 1 — Skeleton.** Ask the model to produce a concise outline. A numbered list of key points. No detail. Just structure. This is fast because the output is short.

**Phase 2 — Expansion.** Take each point from the skeleton and expand it independently. Since no point depends on any other, all expansions can run as parallel API calls.

The total latency is roughly the skeleton time plus the slowest expansion — not the sum of all expansions.

```
CoT:  [step 1]──▶[step 2]──▶[step 3]──▶[step 4]──▶[step 5]
      Total = sum of all steps

SoT:  [skeleton]──▶┬──[expand 1]──▶┐
                   ├──[expand 2]──▶│
                   ├──[expand 3]──▶├──▶ [assemble]
                   ├──[expand 4]──▶│
                   └──[expand 5]──▶┘
      Total = skeleton + max(expansions)
```

## When It Works

SoT works best when the answer has a natural structure — multiple independent points that do not depend on each other.

Good for:
- "Explain the differences between X and Y"
- "What are the key principles of Z?"
- "List the pros and cons of A"
- "How does B work?" (when the explanation has distinct components)

Not ideal for:
- Mathematical proofs where each step depends on the last
- Sequential debugging where you follow a chain of causation
- Creative writing where narrative flow matters

The research by Ning et al. reported speed-ups of up to 2x+ on suitable tasks with comparable output quality.

## The Code

Three examples that demonstrate the pattern.

### Minimal SoT (40 lines)

The core technique with no overhead.

```python
from anthropic import Anthropic
import asyncio

client = Anthropic()

def skeleton(question):
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=256,
        messages=[{"role": "user", "content":
            f"List 4-6 key points to answer: {question}\n"
            f"Numbered list only. No detail."}],
    )
    return [line.lstrip("0123456789.-) ").strip()
            for line in r.content[0].text.strip().split("\n")
            if line.strip() and line.strip()[0].isdigit()]

async def expand(question, point):
    loop = asyncio.get_event_loop()
    def _call():
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=300,
            messages=[{"role": "user", "content":
                f"Context: answering '{question}'\n"
                f"Expand into 2-3 sentences: {point}"}],
        )
        return r.content[0].text.strip()
    return await loop.run_in_executor(None, _call)

async def sot(question):
    points = skeleton(question)
    expansions = await asyncio.gather(*[expand(question, p) for p in points])
    return "\n\n".join(f"**{p}**\n{e}" for p, e in zip(points, expansions))

print(asyncio.run(sot("Explain how DNS works")))
```

Two API calls in sequence (skeleton then expansion), but the expansion phase makes N calls in parallel. For a 5-point answer, you get 5 parallel calls instead of one long sequential generation.

### Head-to-Head Comparison

```bash
python examples/sot_vs_cot.py "What are the key differences between microservices and monolithic architecture?"
```

Runs the same question through both CoT and SoT. Measures latency. Prints the speedup ratio.

### Batch Evaluation

```bash
python examples/sot_batch.py
```

Runs 5 questions through both approaches and prints a comparison table.

## Running the Examples

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Minimal example
python examples/sot_basic.py "Explain how DNS works"

# Head-to-head with timing
python examples/sot_vs_cot.py

# Batch comparison table
python examples/sot_batch.py
```

## The Tradeoff

SoT is not free. It makes more API calls (1 skeleton + N expansions vs 1 CoT call). Total token usage may be higher. The individual expansions lack cross-point context — point 3 does not know what point 2 said.

For latency-sensitive applications — AI Agents, real-time assistants, interactive tools — the tradeoff is often worth it. Wall-clock time drops. The user gets a structured answer faster.

For applications where coherence across points matters more than speed, CoT remains the better choice.

## The Principle

CoT treats reasoning as a serial process. SoT treats it as a parallelisable one. The insight is that many answers are naturally decomposable — they have structure, and that structure can be exploited to break the sequential bottleneck.

Outline first. Expand in parallel. Assemble.

---

**Reference:** Ning, X., et al. (2023). Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation. arXiv:2307.15337.

---

*Chief Evangelist @ Kore.ai | I'm passionate about exploring the intersection of AI and language. Language Models, AI Agents, Agentic Apps, Dev Frameworks & Data-Driven Tools shaping tomorrow.*
