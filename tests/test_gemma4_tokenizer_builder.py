import unittest

from scripts.data_prep.build_gemma4_tokenizer import (
    contains_dropped_script,
    decoded_piece_bytes,
    is_byte_token,
    make_parent_graph,
    reachable_tokens,
)
from scripts.data_prep.gemma4_protected_tokens import (
    CUSTOM_SPECIAL_TOKENS,
    GEMMA_NATIVE_SPECIAL_TOKENS,
    RESERVED_SPECIAL_TOKENS,
)


class GemmaChildTokenizerBuilderTests(unittest.TestCase):
    def test_byte_piece_detection_is_exact(self) -> None:
        self.assertTrue(is_byte_token("<0x00>"))
        self.assertTrue(is_byte_token("<0xFF>"))
        self.assertFalse(is_byte_token("<0xff>"))
        self.assertFalse(is_byte_token("x<0x00>"))

    def test_piece_byte_utility_uses_decoded_space_marker(self) -> None:
        self.assertEqual(decoded_piece_bytes("▁hello"), 6)
        self.assertEqual(decoded_piece_bytes("<0xF0>"), 1)

    def test_script_filter_keeps_english_code_and_math(self) -> None:
        self.assertFalse(contains_dropped_script("torch.nn.Linear"))
        self.assertFalse(contains_dropped_script("λ = 3.0"))
        self.assertTrue(contains_dropped_script("中文"))
        self.assertTrue(contains_dropped_script("العربية"))

    def test_parent_graph_rejects_invalid_merges(self) -> None:
        with self.assertRaises(ValueError):
            make_parent_graph([["a", "missing"]], {"a": 0, "amissing": 1})

    def test_reachability_handles_alternate_merge_activation(self) -> None:
        # Gemma contains cases where an earlier-ranked merge becomes possible
        # only after a later-ranked intermediate merge is formed.
        selected = {"a", "aa", "aaa"}
        merges = [["aa", "a"], ["a", "a"]]
        self.assertEqual(reachable_tokens(selected, merges, set()), selected)

    def test_special_token_manifest_has_no_duplicates(self) -> None:
        tokens = (
            GEMMA_NATIVE_SPECIAL_TOKENS
            + CUSTOM_SPECIAL_TOKENS
            + RESERVED_SPECIAL_TOKENS
        )
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertEqual(GEMMA_NATIVE_SPECIAL_TOKENS[:4], (
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
        ))


if __name__ == "__main__":
    unittest.main()
