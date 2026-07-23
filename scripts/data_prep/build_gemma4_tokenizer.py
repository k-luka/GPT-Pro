#!/usr/bin/env python3
"""Build a frequency-ranked, merge-closed child of the Gemma 4 tokenizer.

The ordinary vocabulary is a strict subset of the original Gemma vocabulary.
This preserves an exact student-token -> teacher-token mapping for later logit
distillation.  A small number of explicitly declared custom special tokens have
no teacher ID and are marked ``null`` in the mapping.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import string
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

try:  # Support both direct CLI execution and package imports in tests.
    from .gemma4_protected_tokens import (
        GEMMA_NATIVE_SPECIAL_TOKENS,
        GEMMA_PROMPT_FIXTURE,
        GEMMA_PROMPT_PIECES,
        all_custom_special_tokens,
        protection_texts,
    )
except ImportError:
    from gemma4_protected_tokens import (
        GEMMA_NATIVE_SPECIAL_TOKENS,
        GEMMA_PROMPT_FIXTURE,
        GEMMA_PROMPT_PIECES,
        all_custom_special_tokens,
        protection_texts,
    )


BYTE_TOKEN_RE = re.compile(r"<0x[0-9A-F]{2}>")
UNUSED_TOKEN_RE = re.compile(r"<unused\d+>")

DROP_EXACT = {
    "<mask>",
    "[multimodal]",
    "<|image>",
    "<image|>",
    "<|image|>",
    "<|audio>",
    "<audio|>",
    "<|audio|>",
    "<|video|>",
}

# Scripts deliberately excluded from efficient merged representation.  Byte
# fallback still makes every Unicode string losslessly representable.
DROP_SCRIPT_RANGES = (
    (0x0400, 0x052F),  # Cyrillic
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x08FF),  # Arabic and extensions
    (0x0900, 0x0DFF),  # major Indic blocks
    (0x0E00, 0x0E7F),  # Thai
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana / Katakana
    (0x3130, 0x318F),  # Hangul compatibility Jamo
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-tokenizer", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--output-dir", default="tokenizers/gemma4_80k")
    parser.add_argument("--target-vocab-size", type=int, default=81_920)
    parser.add_argument(
        "--allow-dropped-scripts",
        action="store_true",
        help="Allow frequency-ranked tokens from scripts excluded by default.",
    )
    return parser.parse_args()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contains_dropped_script(token: str) -> bool:
    return any(
        lower <= ord(character) <= upper
        for character in token
        for lower, upper in DROP_SCRIPT_RANGES
    )


def is_byte_token(token: str) -> bool:
    return BYTE_TOKEN_RE.fullmatch(token) is not None


def is_base_atom(token: str) -> bool:
    # BPE begins with Unicode scalars; missing scalars are handled by the 256
    # byte-fallback pieces.  Added/special tokens are handled separately.
    return len(token) == 1 or is_byte_token(token)


def decoded_piece_bytes(token: str) -> int:
    if is_byte_token(token):
        return 1
    # Gemma normalizes literal spaces to the SentencePiece-style marker.
    return len(token.replace("▁", " ").encode("utf-8"))


def token_utility(count: int, token: str) -> float:
    # Frequency is the primary signal.  The weak length multiplier rewards a
    # long piece without allowing rare long artifacts to outrank common words.
    saved = max(1, decoded_piece_bytes(token) - 1)
    return float(count) * (1.0 + math.log2(saved))


def load_counts(path: str, expected_size: int) -> tuple[np.ndarray, dict[str, int]]:
    with np.load(path) as archive:
        counts = np.asarray(archive["counts"], dtype=np.uint64)
        scalars = {
            name: int(archive[name])
            for name in ("documents", "source_tokens", "gemma_tokens", "utf8_bytes")
            if name in archive
        }
    if counts.shape != (expected_size,):
        raise ValueError(
            f"count vector has shape {counts.shape}; expected ({expected_size},)"
        )
    return counts, scalars


def make_parent_graph(
    merges: list[list[str]], vocab: dict[str, int]
) -> tuple[dict[str, list[tuple[int, str, str]]], set[str]]:
    parents: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    outputs: set[str] = set()
    for rank, pair in enumerate(merges):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"unexpected merge record at rank {rank}: {pair!r}")
        left, right = pair
        output = left + right
        if left not in vocab or right not in vocab or output not in vocab:
            raise ValueError(
                f"merge rank {rank} references token outside vocabulary: {pair!r}"
            )
        parents[output].append((rank, left, right))
        outputs.add(output)
    return parents, outputs


def whitespace_tokens(vocab: dict[str, int]) -> set[str]:
    protected = {"▁", "\n", "\t"}
    for token in vocab:
        if token and set(token) == {"▁"}:
            protected.add(token)
        elif token and set(token) == {"\n"}:
            protected.add(token)
        elif token and set(token) == {"\t"}:
            protected.add(token)
    return protected


def native_protected_tokens(tokenizer: Tokenizer, vocab: dict[str, int]) -> set[str]:
    protected = set(GEMMA_NATIVE_SPECIAL_TOKENS)
    missing = protected - vocab.keys()
    if missing:
        raise ValueError(f"Gemma source is missing required specials: {sorted(missing)}")

    protected.update(token for token in vocab if is_byte_token(token))
    if sum(is_byte_token(token) for token in vocab) != 256:
        raise ValueError("Gemma tokenizer does not contain all 256 byte pieces")

    protected.update(whitespace_tokens(vocab))
    for character in string.printable:
        normalized = "▁" if character == " " else character
        if normalized in vocab:
            protected.add(normalized)

    missing_prompt_pieces = set(GEMMA_PROMPT_PIECES) - vocab.keys()
    if missing_prompt_pieces:
        raise ValueError(
            f"Gemma source is missing prompt pieces: {sorted(missing_prompt_pieces)}"
        )
    protected.update(GEMMA_PROMPT_PIECES)

    for text in protection_texts():
        encoding = tokenizer.encode(text, add_special_tokens=False)
        protected.update(encoding.tokens)
    protected.update(
        tokenizer.encode(GEMMA_PROMPT_FIXTURE, add_special_tokens=False).tokens
    )

    # Keeping only a final token plus one possible dependency tree can still
    # alter BPE segmentation: Gemma may reach that token through a different
    # sequence of ranked intermediate merges.  For the relatively small hand-
    # protected corpus, retain every in-vocabulary substring of each emitted
    # piece.  That preserves the complete within-piece merge search space and
    # makes canonical code/prompt fixtures tokenize identically to Gemma.
    emitted_pieces = tuple(protected)
    for piece in emitted_pieces:
        if piece in GEMMA_NATIVE_SPECIAL_TOKENS or is_byte_token(piece):
            continue
        for start in range(len(piece)):
            for stop in range(start + 1, len(piece) + 1):
                substring = piece[start:stop]
                if substring in vocab:
                    protected.add(substring)
    return protected


def choose_vocabulary(
    *,
    vocab: dict[str, int],
    inverse_vocab: list[str],
    parents: dict[str, list[tuple[int, str, str]]],
    counts: np.ndarray,
    protected: set[str],
    native_budget: int,
    allow_dropped_scripts: bool,
) -> tuple[set[str], dict[str, object]]:
    native_specials = set(GEMMA_NATIVE_SPECIAL_TOKENS)

    def eligible(token: str) -> bool:
        if token in protected:
            return True
        if token in DROP_EXACT or UNUSED_TOKEN_RE.fullmatch(token):
            return False
        if token.startswith("<|") or token.endswith("|>"):
            # Unknown control/multimodal entries are never admitted by frequency.
            return False
        if not allow_dropped_scripts and contains_dropped_script(token):
            return False
        return True

    @lru_cache(maxsize=None)
    def minimal_closure(token: str) -> frozenset[str] | None:
        if token not in vocab or not eligible(token):
            return None
        if token in native_specials or is_base_atom(token):
            return frozenset((token,))

        best: frozenset[str] | None = None
        best_rank = math.inf
        for rank, left, right in parents.get(token, ()):  # alternate BPE paths
            left_closure = minimal_closure(left)
            right_closure = minimal_closure(right)
            if left_closure is None or right_closure is None:
                continue
            candidate = left_closure | right_closure | {token}
            if best is None or (len(candidate), rank) < (len(best), best_rank):
                best = frozenset(candidate)
                best_rank = rank
        return best

    selected: set[str] = set()
    for token in sorted(protected, key=lambda item: vocab[item]):
        closure = minimal_closure(token)
        if closure is None:
            raise ValueError(f"protected token is not derivable: {token!r}")
        selected.update(closure)
    mandatory_size = len(selected)
    if mandatory_size > native_budget:
        raise ValueError(
            f"protected token closure needs {mandatory_size:,} native entries, "
            f"exceeding budget {native_budget:,}"
        )

    candidates = [
        token
        for token in inverse_vocab
        if token not in selected and eligible(token) and int(counts[vocab[token]]) > 0
    ]
    candidates.sort(
        key=lambda token: (
            token_utility(int(counts[vocab[token]]), token),
            int(counts[vocab[token]]),
            -vocab[token],
        ),
        reverse=True,
    )

    skipped_for_cost = 0
    for token in candidates:
        if len(selected) >= native_budget:
            break
        closure = minimal_closure(token)
        if closure is None:
            continue
        additions = closure - selected
        if len(selected) + len(additions) <= native_budget:
            selected.update(additions)
        else:
            skipped_for_cost += 1

    # A second pass fills small holes after high-value dependencies have made
    # other closures cheaper.  Zero-count tokens are deliberately excluded.
    changed = True
    while len(selected) < native_budget and changed:
        changed = False
        for token in candidates:
            if token in selected:
                continue
            closure = minimal_closure(token)
            if closure is None:
                continue
            additions = closure - selected
            if additions and len(selected) + len(additions) <= native_budget:
                selected.update(additions)
                changed = True
                if len(selected) == native_budget:
                    break

    if len(selected) != native_budget:
        raise RuntimeError(
            f"could only select {len(selected):,}/{native_budget:,} native "
            "tokens with nonzero corpus frequency and valid merge closure"
        )

    selected_ids = np.fromiter((vocab[token] for token in selected), dtype=np.int64)
    selected_mass = int(counts[selected_ids].sum())
    total_mass = int(counts.sum())
    report = {
        "mandatory_native_closure": mandatory_size,
        "selected_native_tokens": len(selected),
        "selected_original_token_mass": selected_mass,
        "total_original_token_mass": total_mass,
        "selected_original_token_fraction": selected_mass / max(total_mass, 1),
        "skipped_for_closure_budget": skipped_for_cost,
        "minimal_closure_cache": minimal_closure.cache_info()._asdict(),
    }
    return selected, report


def reachable_tokens(
    selected: set[str], merges: list[list[str]], special_tokens: set[str]
) -> set[str]:
    reachable = {
        token
        for token in selected
        if is_base_atom(token) or token in special_tokens
    }
    dependents: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for left, right in merges:
        output = left + right
        dependents[left].append((left, right, output))
        if right != left:
            dependents[right].append((left, right, output))

    queue = deque(reachable)
    while queue:
        token = queue.popleft()
        for left, right, output in dependents.get(token, ()):  # pragma: no branch
            if output not in reachable and left in reachable and right in reachable:
                reachable.add(output)
                queue.append(output)
    return reachable


def rewrite_tokenizer(
    source_json: dict[str, object],
    source_vocab: dict[str, int],
    selected_native: set[str],
    custom_specials: tuple[str, ...],
) -> tuple[dict[str, object], list[int | None], list[int], int]:
    for token in custom_specials:
        if token in source_vocab:
            raise ValueError(f"custom special already exists in Gemma vocab: {token}")

    leading_native = list(GEMMA_NATIVE_SPECIAL_TOKENS)
    remaining_native = sorted(
        selected_native - set(leading_native), key=lambda token: source_vocab[token]
    )
    ordered = leading_native + list(custom_specials) + remaining_native
    if len(ordered) != len(set(ordered)):
        raise AssertionError("duplicate output token")

    new_vocab = {token: index for index, token in enumerate(ordered)}
    new_merges = [
        pair
        for pair in source_json["model"]["merges"]
        if pair[0] in selected_native
        and pair[1] in selected_native
        and pair[0] + pair[1] in selected_native
    ]

    output_json = copy.deepcopy(source_json)
    output_json["model"]["vocab"] = new_vocab
    output_json["model"]["merges"] = new_merges

    original_added = {
        record["content"]: record for record in source_json.get("added_tokens", [])
    }
    special_set = set(GEMMA_NATIVE_SPECIAL_TOKENS) | set(custom_specials)
    added_tokens = []
    for token in ordered:
        if token not in special_set:
            continue
        record = copy.deepcopy(original_added.get(token, {}))
        record.update(
            {
                "id": new_vocab[token],
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        added_tokens.append(record)
    output_json["added_tokens"] = sorted(added_tokens, key=lambda record: record["id"])

    # IDs 0/1/2/3 are intentionally identical to Gemma.  Rebuild the template
    # explicitly so future source-tokenizer changes cannot leave a stale BOS ID.
    output_json["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<bos>", "type_id": 1}},
            {"Sequence": {"id": "B", "type_id": 1}},
        ],
        "special_tokens": {
            "<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}
        },
    }

    student_to_gemma: list[int | None] = [
        source_vocab.get(token) for token in ordered
    ]
    gemma_to_student = [-1] * len(source_vocab)
    for student_id, gemma_id in enumerate(student_to_gemma):
        if gemma_id is not None:
            gemma_to_student[gemma_id] = student_id
    return output_json, student_to_gemma, gemma_to_student, len(new_merges)


def validate_output(
    *,
    tokenizer_path: Path,
    gemma_tokenizer_path: str,
    expected_size: int,
    student_to_gemma: list[int | None],
    selected_native: set[str],
    output_json: dict[str, object],
    custom_specials: tuple[str, ...],
) -> dict[str, object]:
    child = Tokenizer.from_file(str(tokenizer_path))
    teacher = Tokenizer.from_file(gemma_tokenizer_path)
    vocab = child.get_vocab()
    if child.get_vocab_size() != expected_size:
        raise AssertionError(
            f"child size {child.get_vocab_size()} != expected {expected_size}"
        )
    if sorted(vocab.values()) != list(range(expected_size)):
        raise AssertionError("child token IDs are not dense")
    if [vocab[token] for token in ("<pad>", "<eos>", "<bos>", "<unk>")] != [0, 1, 2, 3]:
        raise AssertionError("core special IDs changed")
    if sum(is_byte_token(token) for token in vocab) != 256:
        raise AssertionError("child lost byte fallback coverage")
    if "▁" not in vocab:
        raise AssertionError("child lost Gemma's normalized-space atom")

    filtered_merges = output_json["model"]["merges"]
    reachable = reachable_tokens(
        selected_native, filtered_merges, set(GEMMA_NATIVE_SPECIAL_TOKENS)
    )
    unreachable = selected_native - reachable
    if unreachable:
        examples = sorted(unreachable)[:20]
        raise AssertionError(
            f"{len(unreachable):,} selected native tokens are unreachable; "
            f"examples={examples!r}"
        )

    general_fuzz_strings = [
        "Hello world!  Two spaces.\n    Python indentation\n\tand a tab.",
        "Café naïve — π ≈ 3.14159; emoji: 🧠🚀",
        "中文 العربية हिन्दी русский 한국어 日本語",
        "path/to/file.py __name__ == '__main__' {x: [1, 2, 3]}",
        "\x00\x01 control bytes and trailing space ",
    ]
    protected_strings = protection_texts()
    fuzz_strings = general_fuzz_strings + protected_strings
    mapped_equivalence = 0
    exact_protected_segmentations = 0
    unk_id = vocab["<unk>"]
    for text in fuzz_strings:
        encoding = child.encode(text, add_special_tokens=False)
        if unk_id in encoding.ids:
            raise AssertionError(f"valid UTF-8 text emitted <unk>: {text!r}")
        decoded = child.decode(encoding.ids, skip_special_tokens=False)
        if decoded != text:
            raise AssertionError(
                f"child round trip failed: expected {text!r}, got {decoded!r}"
            )
        mapped = [student_to_gemma[token_id] for token_id in encoding.ids]
        if any(token_id is None for token_id in mapped):
            # Ordinary text should never trigger custom special tokens.
            raise AssertionError("ordinary text emitted a custom special token")
        teacher_decoded = teacher.decode(mapped, skip_special_tokens=False)
        if teacher_decoded != text:
            raise AssertionError(
                f"mapped teacher round trip failed: {text!r} -> {teacher_decoded!r}"
            )
        mapped_equivalence += 1
        if text in protected_strings:
            teacher_ids = teacher.encode(text, add_special_tokens=False).ids
            if mapped != teacher_ids:
                raise AssertionError(
                    "protected code/math text no longer has exact Gemma segmentation"
                )
            exact_protected_segmentations += 1

    for student_id, gemma_id in enumerate(student_to_gemma):
        child_piece = child.id_to_token(student_id)
        if gemma_id is None:
            continue
        if child_piece != teacher.id_to_token(gemma_id):
            raise AssertionError(
                f"mapping token mismatch at student ID {student_id}: "
                f"{child_piece!r} != {teacher.id_to_token(gemma_id)!r}"
            )

    for token in GEMMA_NATIVE_SPECIAL_TOKENS:
        encoded = child.encode(token, add_special_tokens=False).ids
        if encoded != [vocab[token]]:
            raise AssertionError(f"native special is not atomic: {token!r} -> {encoded}")
        student_id = encoded[0]
        if student_to_gemma[student_id] != teacher.token_to_id(token):
            raise AssertionError(f"native special mapping changed: {token!r}")

    for token in custom_specials:
        encoded = child.encode(token, add_special_tokens=False).ids
        if encoded != [vocab[token]]:
            raise AssertionError(f"custom special is not atomic: {token!r} -> {encoded}")
        if student_to_gemma[encoded[0]] is not None:
            raise AssertionError(f"custom special unexpectedly maps to Gemma: {token!r}")

    child_prompt = child.encode(GEMMA_PROMPT_FIXTURE, add_special_tokens=False).ids
    teacher_prompt = teacher.encode(
        GEMMA_PROMPT_FIXTURE, add_special_tokens=False
    ).ids
    mapped_prompt = [student_to_gemma[token_id] for token_id in child_prompt]
    if mapped_prompt != teacher_prompt:
        raise AssertionError("canonical prompt IDs do not map exactly to Gemma IDs")

    bos_encoding = child.encode("hello", add_special_tokens=True)
    if not bos_encoding.ids or bos_encoding.ids[0] != 2 or bos_encoding.ids.count(2) != 1:
        raise AssertionError("post-processor did not insert exactly one BOS")
    if 2 in child.encode("hello", add_special_tokens=False).ids:
        raise AssertionError("BOS appeared when special-token processing was disabled")

    return {
        "dense_ids": True,
        "byte_tokens": 256,
        "reachable_native_tokens": len(reachable),
        "round_trip_cases": len(fuzz_strings),
        "teacher_mapping_round_trip_cases": mapped_equivalence,
        "exact_protected_segmentations": exact_protected_segmentations,
        "native_specials_atomic": len(GEMMA_NATIVE_SPECIAL_TOKENS),
        "custom_specials_atomic": len(custom_specials),
        "canonical_prompt_mapping_exact": True,
        "bos_once": True,
    }


def main() -> None:
    args = parse_args()
    source_path = Path(args.gemma_tokenizer)
    with source_path.open(encoding="utf-8") as handle:
        source_json = json.load(handle)
    source_vocab: dict[str, int] = source_json["model"]["vocab"]
    inverse_vocab = [""] * len(source_vocab)
    for token, token_id in source_vocab.items():
        inverse_vocab[token_id] = token
    if not all(inverse_vocab):
        raise ValueError("source vocabulary IDs are not dense")

    counts, count_metadata = load_counts(args.counts, len(source_vocab))
    parents, _ = make_parent_graph(source_json["model"]["merges"], source_vocab)
    source_tokenizer = Tokenizer.from_file(str(source_path))
    protected = native_protected_tokens(source_tokenizer, source_vocab)
    custom_specials = all_custom_special_tokens()
    native_budget = args.target_vocab_size - len(custom_specials)
    if native_budget <= 0:
        raise ValueError("target vocabulary is smaller than custom special set")

    selected, selection_report = choose_vocabulary(
        vocab=source_vocab,
        inverse_vocab=inverse_vocab,
        parents=parents,
        counts=counts,
        protected=protected,
        native_budget=native_budget,
        allow_dropped_scripts=args.allow_dropped_scripts,
    )
    output_json, student_to_gemma, gemma_to_student, merge_count = rewrite_tokenizer(
        source_json, source_vocab, selected, custom_specials
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(output_json, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (output_dir / "student_to_gemma_id.json").write_text(
        json.dumps(student_to_gemma, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gemma_to_student_id.json").write_text(
        json.dumps(gemma_to_student, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    new_vocab: dict[str, int] = output_json["model"]["vocab"]
    with (output_dir / "vocab.txt").open("w", encoding="utf-8") as handle:
        for token, token_id in sorted(new_vocab.items(), key=lambda item: item[1]):
            teacher_id = student_to_gemma[token_id]
            handle.write(f"{token_id}\t{teacher_id}\t{token!r}\t{int(counts[teacher_id]) if teacher_id is not None else 0}\n")

    validation = validate_output(
        tokenizer_path=tokenizer_path,
        gemma_tokenizer_path=str(source_path),
        expected_size=args.target_vocab_size,
        student_to_gemma=student_to_gemma,
        selected_native=selected,
        output_json=output_json,
        custom_specials=custom_specials,
    )
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 1_048_576,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": list(GEMMA_NATIVE_SPECIAL_TOKENS[4:])
        + list(custom_specials),
    }
    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": tokenizer_config["additional_special_tokens"],
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "name": f"gemma4-en-code-{args.target_vocab_size}",
        "target_vocab_size": args.target_vocab_size,
        "native_gemma_tokens": len(selected),
        "custom_special_tokens": list(custom_specials),
        "gemma_native_special_tokens": list(GEMMA_NATIVE_SPECIAL_TOKENS),
        "protected_native_tokens_before_closure": len(protected),
        "retained_merges": merge_count,
        "source_gemma_tokenizer_sha256": sha256_file(source_path),
        "counts_sha256": sha256_file(args.counts),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "count_metadata": count_metadata,
        "selection": selection_report,
        "validation": validation,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved tokenizer: {tokenizer_path}")


if __name__ == "__main__":
    main()
