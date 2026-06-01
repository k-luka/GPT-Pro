"""
CORE-style multiple-choice eval tasks (nanochat/DCLM-style aggregate).

Each task is normalized to {"context": str, "choices": [str, ...], "label": int}
and scored the same way as HellaSwag: append each choice to the context, take
the length-normalized LM loss over the choice tokens, and predict the choice
with the lowest loss. The mean accuracy across tasks is the "CORE" score.

Tasks (loaded via HuggingFace `datasets`, cached under ~/.cache/huggingface so
later/offline runs reuse them). All three share the ARC-style schema
(question + choices{text,label} + answerKey), so one parser handles them:
  - arc_easy       (allenai/ai2_arc, ARC-Easy)
  - arc_challenge  (allenai/ai2_arc, ARC-Challenge)
  - openbookqa     (allenai/openbookqa, main)
HellaSwag stays in its own module (already wired into the trainer).
"""

import torch

from scripts.data_prep.hellaswag import _get_enc, EOT_ID


def render_mc(context, choices, label):
    """Render a multiple-choice example to (tokens, mask, label).

    tokens/mask are (n_choices, max_len); mask is 1 over the choice tokens
    (where we score the LM loss), 0 over the shared context. Mirrors
    hellaswag.render_example but for an arbitrary number of choices.
    """
    enc = _get_enc()
    ctx_tokens = [EOT_ID] + enc.encode(context).ids
    tok_rows, mask_rows = [], []
    for choice in choices:
        # leading space for byte-level BPE word boundary, matching HellaSwag
        end_tokens = enc.encode(" " + str(choice)).ids
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    n = len(choices)
    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((n, max_len), dtype=torch.long)
    mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, (tr, mr) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tr)] = torch.tensor(tr)
        mask[i, : len(mr)] = torch.tensor(mr)
    return tokens, mask, label


def _iterate_arc_style(path, subset):
    """ARC / OpenBookQA share schema: question(_stem) + choices{text,label} +
    answerKey. All parquet-backed (no loading script)."""
    from datasets import load_dataset

    ds = load_dataset(path, subset, split="validation")
    for ex in ds:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        key = ex["answerKey"]
        if key not in labels:  # a few malformed rows
            continue
        context = ex.get("question", ex.get("question_stem"))
        yield {"context": context, "choices": texts, "label": labels.index(key)}


# task name -> zero-arg generator yielding normalized examples
TASKS = {
    "arc_easy": lambda: _iterate_arc_style("allenai/ai2_arc", "ARC-Easy"),
    "arc_challenge": lambda: _iterate_arc_style("allenai/ai2_arc", "ARC-Challenge"),
    "openbookqa": lambda: _iterate_arc_style("allenai/openbookqa", "main"),
}
