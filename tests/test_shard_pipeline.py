import tempfile
import unittest
import warnings
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from tokenizers import Tokenizer

from scripts.data_prep.core_tasks import render_mc
from scripts.data_prep.hellaswag import render_example
from scripts.data_prep.prepare_climbmix import (
    _apply_resume_offset,
    _init_worker,
    _load_checkpoint,
    _save_checkpoint,
    _tokenize,
    main as prepare_main,
)
from src.datasets.dataloader import DataLoader
from src.datasets.shard_format import (
    assert_metadata_matches,
    build_metadata,
    dtype_for_vocab_size,
    write_metadata,
)


TOKENIZER_PATH = Path("tokenizers/gemma4_80k/tokenizer.json")
TOKENIZER_SHA = "d3fe07a051ce0bbe1c111f7a4c88638451b175da2282338d90f978b00e39bf97"


def metadata(dtype="uint32", vocab_size=81920, shard_size=16):
    return build_metadata(
        dtype=np.dtype(dtype),
        vocab_size=vocab_size,
        max_token_id=vocab_size - 1,
        bos_token="<bos>",
        bos_id=2,
        block_size=4,
        shard_size=shard_size,
        tokenizer_sha256=TOKENIZER_SHA,
        dataset="test/dataset",
        val_shards=0,
    )


class ShardFormatTests(unittest.TestCase):
    def test_dtype_boundary(self):
        self.assertEqual(dtype_for_vocab_size(65536), np.dtype("uint16"))
        self.assertEqual(dtype_for_vocab_size(65537), np.dtype("uint32"))

    def test_metadata_mismatch_refuses_resume(self):
        actual = metadata()
        expected = {**actual, "block_size": 8}
        with self.assertRaisesRegex(ValueError, "block_size"):
            assert_metadata_matches(actual, expected)

    def test_mid_document_resume_reconstructs_exact_tail(self):
        document = np.arange(20, dtype=np.uint32)
        persisted = document[:7].copy()
        tail, consumed = _apply_resume_offset(
            document,
            doc_index=12,
            completed_docs=12,
            resume_offset=7,
        )
        self.assertEqual(consumed, 7)
        np.testing.assert_array_equal(np.concatenate((persisted, tail)), document)

        with tempfile.TemporaryDirectory() as directory:
            info = metadata()
            _save_checkpoint(
                directory,
                next_shard=3,
                completed_docs=12,
                doc_token_offset=7,
                metadata=info,
            )
            state = _load_checkpoint(directory, info)
            self.assertEqual(state["doc_token_offset"], 7)
            with self.assertRaises(ValueError):
                _load_checkpoint(directory, {**info, "dtype": "uint16"})

    @unittest.skipUnless(TOKENIZER_PATH.exists(), "generated tokenizer is absent")
    def test_manual_bos_and_uint32_tokenization(self):
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        _init_worker(str(TOKENIZER_PATH), 2, "uint32")
        result = _tokenize({"text": "hello world"})
        expected_body = tokenizer.encode(
            "hello world", add_special_tokens=False
        ).ids
        self.assertEqual(result.dtype, np.dtype("uint32"))
        self.assertEqual(result.tolist(), [2, *expected_body])
        self.assertNotEqual(expected_body[0], 2)

    @unittest.skipUnless(TOKENIZER_PATH.exists(), "generated tokenizer is absent")
    def test_end_to_end_resume_does_not_drop_mid_document_tail(self):
        class FakeDataset(list):
            def skip(self, count):
                return FakeDataset(self[count:])

        class FakePool:
            def __init__(self, _workers, initializer, initargs):
                initializer(*initargs)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def imap(self, function, iterable, chunksize):
                del chunksize
                return map(function, iterable)

        text = "hello world " * 30
        dataset = FakeDataset([{"text": text}])
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        expected = np.asarray(
            [
                2,
                *tokenizer.encode(text, add_special_tokens=False).ids,
            ],
            dtype=np.uint32,
        )
        self.assertGreater(len(expected), 16)

        with tempfile.TemporaryDirectory() as directory:
            common_args = [
                "prepare_climbmix.py",
                "--tokenizer",
                str(TOKENIZER_PATH),
                "--output_dir",
                directory,
                "--block_size",
                "4",
                "--shard_size",
                "8",
                "--val_shards",
                "0",
                "--num_workers",
                "1",
                "--max_shards",
                "1",
            ]
            with (
                mock.patch(
                    "scripts.data_prep.prepare_climbmix.load_dataset",
                    return_value=dataset,
                ),
                mock.patch("scripts.data_prep.prepare_climbmix.mp.Pool", FakePool),
                mock.patch.object(sys, "argv", common_args),
            ):
                prepare_main()
            with (
                mock.patch(
                    "scripts.data_prep.prepare_climbmix.load_dataset",
                    return_value=dataset,
                ),
                mock.patch("scripts.data_prep.prepare_climbmix.mp.Pool", FakePool),
                mock.patch.object(sys, "argv", [*common_args, "--resume"]),
            ):
                prepare_main()

            root = Path(directory)
            actual = np.concatenate(
                [
                    np.fromfile(path, dtype=np.uint32)
                    for path in sorted(root.glob("*.bin"))
                ]
            )
            np.testing.assert_array_equal(actual, expected[:16])


class DataLoaderTests(unittest.TestCase):
    def test_uint32_ids_and_rank_lockstep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_metadata(root, metadata(shard_size=8))
            first = np.array([70000, 1, 2, 3, 4], dtype=np.uint32)
            second = np.array([71000, 11, 12, 13, 14], dtype=np.uint32)
            first.tofile(root / "climbmix_train_000000.bin")
            second.tofile(root / "climbmix_train_000001.bin")

            rank0 = DataLoader(root, 1, 2, "train", rank=0, world_size=2)
            rank1 = DataLoader(root, 1, 2, "train", rank=1, world_size=2)
            with mock.patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
                x0, _ = next(rank0)
                x1, _ = next(rank1)
                self.assertEqual(x0.tolist(), [[70000, 1]])
                self.assertEqual(x1.tolist(), [[2, 3]])
                next(rank0)
                next(rank1)
            self.assertEqual(rank0.current_shard, rank1.current_shard)
            self.assertEqual(rank0.current_shard, 1)

    def test_dtype_aware_set_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_metadata(root, metadata(shard_size=16))
            np.arange(9, dtype=np.uint32).tofile(
                root / "climbmix_train_000000.bin"
            )
            (np.arange(9, dtype=np.uint32) + 100).tofile(
                root / "climbmix_train_000001.bin"
            )
            loader = DataLoader(root, 1, 2, "train", rank=0, world_size=2)
            loader.set_step(2, 1)
            self.assertEqual(loader.current_shard, 1)
            self.assertEqual(loader.current_position, 0)

    def test_legacy_uint16_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.arange(9, dtype=np.uint16).tofile(
                root / "climbmix_train_000000.bin"
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loader = DataLoader(root, 1, 2, "train")
            self.assertEqual(loader.dtype, np.dtype("uint16"))
            self.assertTrue(any("legacy uint16" in str(item.message) for item in caught))


@unittest.skipUnless(TOKENIZER_PATH.exists(), "generated tokenizer is absent")
class ConfiguredEvaluationTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    def test_hellaswag_uses_configured_tokenizer_and_one_bos(self):
        example = {
            "ctx": "A short context",
            "label": 0,
            "endings": ["one", "two", "three", "four"],
        }
        _, tokens, _, _ = render_example(example, tokenizer=self.tokenizer)
        self.assertTrue(torch.all(tokens[:, 0] == 2))
        expected = self.tokenizer.encode(
            example["ctx"], add_special_tokens=False
        ).ids
        self.assertEqual(tokens[0, 1 : 1 + len(expected)].tolist(), expected)

    def test_core_uses_configured_tokenizer(self):
        tokens, _, _ = render_mc(
            "Question?", ["A", "B"], 0, tokenizer=self.tokenizer
        )
        self.assertTrue(torch.all(tokens[:, 0] == 2))
        self.assertLess(int(tokens.max()), self.tokenizer.get_vocab_size())


if __name__ == "__main__":
    unittest.main()
