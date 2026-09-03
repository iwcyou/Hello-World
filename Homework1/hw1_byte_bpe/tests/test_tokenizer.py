import tempfile
import unittest
from pathlib import Path

from byte_bpe_tokenizer import ByteBPETokenizer, pretokenize


class ByteBPETokenizerTest(unittest.TestCase):
    def setUp(self):
        # a+b -> 256, 256+c -> 257
        self.tokenizer = ByteBPETokenizer(merges=[(97, 98), (256, 99)])

    def test_pretokenization_is_lossless(self):
        text = "Gene TP53：中文 🧬\nnext_line!"
        self.assertEqual("".join(pretokenize(text)), text)

    def test_multilingual_roundtrip(self):
        for text in ["abc abc", "BRCA1/2", "中文与English", "emoji 🧬🙂", "\n\t  "]:
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_merge_order(self):
        ids = self.tokenizer.encode("abc")
        self.assertEqual(ids, [self.tokenizer.offset + 257])

    def test_special_tokens(self):
        ids = self.tokenizer.encode("abc", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], self.tokenizer.bos_token_id)
        self.assertEqual(ids[-1], self.tokenizer.eos_token_id)
        self.assertEqual(self.tokenizer.decode(ids), "abc")

    def test_invalid_id_is_rejected(self):
        for token_id in (-1, self.tokenizer.vocab_size):
            with self.assertRaises(ValueError):
                self.tokenizer.decode([token_id])

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tokenizer.save_pretrained(directory)
            loaded = ByteBPETokenizer.from_pretrained(directory)
            self.assertEqual(loaded.encode("abc 中文"), self.tokenizer.encode("abc 中文"))

    def test_batch_padding(self):
        result = self.tokenizer(["aa", "a"], padding=True)
        self.assertEqual(len(result["input_ids"][0]), len(result["input_ids"][1]))
        self.assertEqual(result["attention_mask"][1][-1], 0)


if __name__ == "__main__":
    unittest.main()
