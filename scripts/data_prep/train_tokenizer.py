"""
Train a custom 64k BPE tokenizer on karpathy/climbmix-400b-shuffle.

Design follows Karpathy's minbpe / nanochat (https://github.com/karpathy/minbpe):
  - Byte-level BPE: all 256 raw bytes are the initial alphabet, no <unk>
  - GPT-4 regex pre-tokenization: splits text into chunks before merging,
    preventing merges across word/number/punctuation boundaries
  - Special tokens added after BPE training at the top of the vocab

Uses the HuggingFace tokenizers (Rust) backend for speed.

Output (default):
  data/tokenizer/tokenizer.json   — load with tokenizers.Tokenizer.from_file()
  data/tokenizer/vocab.txt        — human-readable vocab for inspection

Resulting vocab: 65527 BPE + 9 special = 65536 → set vocab_size: 65536 in config
(65536 = 2^16 is already a multiple of 256, max id 65535 fits in uint16 exactly)
"""

import argparse
import os
import time

import tiktoken
from datasets import load_dataset
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.trainers import BpeTrainer

# ── constants ─────────────────────────────────────────────────────────────────

DATASET = "karpathy/climbmix-400b-shuffle"
DEFAULT_VOCAB_SIZE = (
    65527  # BPE tokens; 9 special tokens bring total to 65536 (= 2^16, fits uint16)
)
DEFAULT_SAMPLE_DOCS = 1_000_000

# GPT-4 split pattern (from Karpathy's minbpe / tiktoken):
# Prevents BPE merges across word/number/punctuation/whitespace category boundaries.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Special tokens — IDs assigned after BPE vocab (starting at vocab_size).
# Layout follows Karpathy's nanochat: dedicated <|bos|> for document boundaries
# (pretraining) + sequence start, separate chat-turn tokens for SFT, and
# python/output pairs reserved for tool-use SFT.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]
BOS_TOKEN = "<|bos|>"


# ── helpers ───────────────────────────────────────────────────────────────────


def text_iterator(n_docs: int):
    """Stream raw text from ClimbMix, yielding one document at a time."""
    ds = load_dataset(DATASET, split="train", streaming=True)
    for i, example in enumerate(ds):
        if i >= n_docs:
            break
        yield example["text"]


def compression_benchmark(tokenizer: Tokenizer, n_docs: int = 500) -> None:
    """Compare tokens/byte vs cl100k_base on a sample of ClimbMix."""
    print("\nRunning compression benchmark (500 docs)...")
    cl100k = tiktoken.get_encoding("cl100k_base")

    total_bytes, custom_tokens, cl_tokens = 0, 0, 0
    ds = load_dataset(DATASET, split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n_docs:
            break
        text = ex["text"]
        total_bytes += len(text.encode("utf-8"))
        custom_tokens += len(tokenizer.encode(text).ids)
        cl_tokens += len(cl100k.encode_ordinary(text))

    print(f"  Bytes sampled       : {total_bytes:>12,}")
    print(
        f"  Custom 32k  tokens  : {custom_tokens:>12,}  ({custom_tokens/total_bytes:.4f} tok/byte)"
    )
    print(
        f"  cl100k_base tokens  : {cl_tokens:>12,}  ({cl_tokens/total_bytes:.4f} tok/byte)"
    )
    ratio = cl_tokens / custom_tokens
    print(
        f"  Compression ratio   : {ratio:.3f}x  ({'better' if ratio > 1 else 'worse'} than cl100k)"
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 64k BPE tokenizer on ClimbMix-400B"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="BPE vocab size (special tokens added on top)",
    )
    parser.add_argument(
        "--sample_docs",
        type=int,
        default=DEFAULT_SAMPLE_DOCS,
        help="Number of ClimbMix documents to sample for training",
    )
    parser.add_argument("--output_dir", type=str, default="data/tokenizer")
    parser.add_argument(
        "--skip_benchmark",
        action="store_true",
        help="Skip compression benchmark after training",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    padded_vocab = ((args.vocab_size + len(SPECIAL_TOKENS) + 255) // 256) * 256

    print("=" * 60)
    print("Custom BPE Tokenizer Training")
    print("=" * 60)
    print(f"  Dataset      : {DATASET}")
    print(f"  Sample docs  : {args.sample_docs:,}")
    print(f"  BPE vocab    : {args.vocab_size:,}")
    print(f"  Special toks : {SPECIAL_TOKENS}")
    print(
        f"  Total vocab  : {args.vocab_size + len(SPECIAL_TOKENS):,}  "
        f"(padded to {padded_vocab:,} for tensor cores)"
    )
    print(f"  Output dir   : {args.output_dir}")
    print("=" * 60)

    # ── build tokenizer ───────────────────────────────────────────────────────
    tokenizer = Tokenizer(BPE(unk_token=None))

    # Pre-tokenizer: GPT-4 regex split → then byte-level encoding within each chunk.
    # This matches Karpathy's minbpe RegexTokenizer design exactly.
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(Regex(GPT4_SPLIT_PATTERN), behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
        # special_tokens added after training so they get IDs at the top of vocab
    )

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\nStreaming {args.sample_docs:,} documents from HuggingFace...")
    t0 = time.time()
    tokenizer.train_from_iterator(
        text_iterator(args.sample_docs),
        trainer=trainer,
        length=args.sample_docs,
    )
    elapsed = time.time() - t0
    print(f"\nBPE training done in {elapsed/60:.1f} min")

    # ── add special tokens at top of vocab ────────────────────────────────────
    tokenizer.add_special_tokens(
        [AddedToken(tok, single_word=False, special=True) for tok in SPECIAL_TOKENS]
    )

    # ── report IDs ────────────────────────────────────────────────────────────
    print("\nSpecial token IDs:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok:<20} → {tokenizer.token_to_id(tok)}")
    print(
        f"\nTotal vocab size: {tokenizer.get_vocab_size():,}  (set vocab_size: {padded_vocab} in config)"
    )

    # ── save ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(json_path)
    print(f"\nSaved: {json_path}")

    # human-readable vocab sorted by ID
    vocab_path = os.path.join(args.output_dir, "vocab.txt")
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, idx in vocab:
            f.write(f"{idx}\t{repr(token)}\n")
    print(f"Saved: {vocab_path}")

    # ── sanity check ──────────────────────────────────────────────────────────
    test = "Hello world! The quick brown fox.\nThis is line 2. x=3.14, y=42."
    enc = tokenizer.encode(test)
    dec = tokenizer.decode(enc.ids)
    print(f"\nSanity check:")
    print(f"  Input  : {test!r}")
    print(f"  Tokens : {enc.tokens[:15]}...")
    print(f"  IDs    : {enc.ids[:15]}...")
    print(f"  Decode : {dec!r}")
    print(f"  Lossless: {test == dec}")

    # ── compression benchmark ─────────────────────────────────────────────────
    if not args.skip_benchmark:
        compression_benchmark(tokenizer)

    print("\nDone. To use in training pipeline:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tok = Tokenizer.from_file('{json_path}')")
    print(
        f"  bos_id = tok.token_to_id('<|bos|>')  # = {tokenizer.token_to_id(BOS_TOKEN)}"
    )


if __name__ == "__main__":
    main()
