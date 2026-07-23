"""Hand-maintained protection corpus for the Gemma 4 child tokenizer.

The child vocabulary is selected primarily from ClimbMix token frequency.  This
module supplies the small, deterministic domain prior that frequency alone
cannot provide: chat/tool control strings, programming syntax, Python names,
and common mathematical notation.

These strings are *not* automatically made into one token.  The builder encodes
them with the original Gemma tokenizer and protects the Gemma pieces that occur.
That preserves exact student-token -> Gemma-token correspondence.
"""

from __future__ import annotations

import builtins
import keyword
import sys


GEMMA_NATIVE_SPECIAL_TOKENS = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    "<|think|>",
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
)


# These are ordinary BPE pieces (not AddedTokens) used by Gemma's canonical
# prompt format.  Protecting them explicitly avoids relying on their frequency
# in an English pretraining sample.
GEMMA_PROMPT_PIECES = (
    "system",
    "user",
    "model",
    "thought",
    "declaration",
    "call",
)


GEMMA_PROMPT_FIXTURE = (
    "<bos><|turn>user\nSolve the problem.<|channel>analysis\n"
    "<|think|>First reason.<channel|><|channel>final\nThe answer.<channel|><turn|>"
)


CUSTOM_SPECIAL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
)


RESERVED_SPECIAL_TOKENS = tuple(f"<|reserved_{i}|>" for i in range(16))


PYTHON_PACKAGES = (
    "aiohttp argparse asyncio collections contextlib copy csv dataclasses datetime enum functools glob hashlib "
    "http importlib inspect io itertools json logging math multiprocessing operator os pathlib pickle queue random "
    "re shutil signal socket sqlite3 statistics string subprocess sys tempfile threading time traceback typing "
    "unittest urllib uuid warnings weakref xml zipfile "
    "numpy pandas scipy sklearn matplotlib seaborn torch torchvision torchaudio transformers tokenizers datasets "
    "accelerate safetensors sentencepiece einops triton pytest requests pydantic fastapi flask django sqlalchemy"
)


GENERAL_CODE_TERMS = """
function const let var interface type namespace export default async await promise
public private protected static final abstract extends implements package instanceof
fn mut impl trait pub crate mod use match enum struct loop while unsafe Result Option
func defer go chan select map range interface package import
SELECT FROM WHERE JOIN LEFT RIGHT INNER OUTER GROUP BY ORDER BY HAVING INSERT UPDATE DELETE CREATE TABLE INDEX
git clone add commit push pull fetch merge rebase checkout branch diff status log
docker kubernetes kubectl cmake makefile cargo rustc npm pnpm yarn node bash shell powershell
json yaml toml xml html css markdown README LICENSE pyproject requirements
"""


CODE_SNIPPETS = r'''
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar
import asyncio
import json
import numpy as np
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 4096
    layers: int = 35
    dtype: torch.dtype = torch.bfloat16

    def validate(self) -> None:
        assert self.hidden_size % 256 == 0, "hidden_size must be aligned"

async def fetch(url: str, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch and decode one JSON response."""
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.get(url, timeout=timeout)
            response.raise_for_status()
            return await response.json()
    except Exception as error:
        raise RuntimeError(f"request failed: {url=}") from error

def fibonacci(n: int) -> list[int]:
    values: list[int] = []
    a, b = 0, 1
    for _ in range(n):
        values.append(a)
        a, b = b, a + b
    return values

if __name__ == "__main__":
    tensor = torch.arange(16, device="cuda").reshape(4, 4)
    print(f"{tensor.shape=}, {tensor.mean().item():.3f}")

// C / C++ / Java / JavaScript / TypeScript / Rust / Go syntax
#include <stdio.h>
#include <vector>
template <typename T> class Vector { public: T* data; };
int main(int argc, char** argv) { return argc > 1 ? 0 : 1; }
const add = (a: number, b: number): number => a + b;
async function main(): Promise<void> { console.log(await fetch(url)); }
pub fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T { a + b }
func main() { fmt.Println("hello, world") }

```python
with open("config.json", "r", encoding="utf-8") as handle:
    config = json.load(handle)
```

{
  "name": "example",
  "enabled": true,
  "items": [1, 2, 3],
  "metadata": null
}

SELECT users.id, users.name
FROM users
LEFT JOIN orders ON orders.user_id = users.id
WHERE users.active = TRUE
ORDER BY users.name ASC;

#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q tests/
git status --short
'''


MATH_SNIPPETS = r'''
Let x, y \in \mathbb{R} and suppose that f(x) = x^2 + 2x + 1.
Therefore f'(x) = 2x + 2 and \int_0^1 f(x)\,dx = \frac{7}{3}.
For every \epsilon > 0 there exists \delta > 0 such that |x-a| < \delta.
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}, \quad
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad
\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{x \sim p(x)}[\nabla_\theta \log p_\theta(x)].
α β γ δ ε θ λ μ π σ φ ψ ω ∀ ∃ ∈ ∉ ⊂ ⊆ ∪ ∩ ∅ ≠ ≤ ≥ ≈ ∞ √ ± × ÷ ∂ ∇
Theorem Lemma Corollary Proof Assume Suppose Therefore Hence Contradiction QED
'''


ROLE_AND_REASONING_TEXT = """
system user model assistant thought analysis final
step by step therefore suppose assume derive verify calculate reason critique revise
tool call tool response function arguments result error output
"""


def protection_texts() -> list[str]:
    """Return deterministic strings whose original Gemma pieces must survive."""

    python_words = sorted(
        set(keyword.kwlist)
        | set(getattr(keyword, "softkwlist", ()))
        | {name for name in dir(builtins) if not name.startswith("__")}
        | set(getattr(sys, "stdlib_module_names", ()))
    )
    return [
        " ".join(python_words),
        PYTHON_PACKAGES,
        GENERAL_CODE_TERMS,
        CODE_SNIPPETS,
        MATH_SNIPPETS,
        ROLE_AND_REASONING_TEXT,
    ]


def all_custom_special_tokens() -> tuple[str, ...]:
    return CUSTOM_SPECIAL_TOKENS + RESERVED_SPECIAL_TOKENS
