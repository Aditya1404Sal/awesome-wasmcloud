#!/usr/bin/env python3
"""Write a tiny, randomly initialized Qwen3 model for testing the plugin.

The model has the real Qwen3 GGUF layout (tensor names, metadata keys, chat
special tokens) at a few hundred kilobytes, so the whole pipeline — loading,
prompting, streaming, stopping — runs without downloading real weights. Its
output is gibberish by construction. For real answers, use fetch-models.sh.

    pip install gguf numpy tokenizers
    python3 scripts/make-test-model.py models/tiny
"""

import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import quantize
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
INTERMEDIATE = 128
LAYERS = 2
CONTEXT = 512

CORPUS = [
    "wasmCloud runs WebAssembly components across clouds, Kubernetes, and edge devices.",
    "A host component plugin serves a capability from its own sandboxed store.",
    "The quick brown fox jumps over the lazy dog. 0123456789 ,.;:!?'\"()[]{}",
]


def make_tokenizer(out: Path) -> int:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    tokenizer.save(str(out / "tokenizer.json"))
    return tokenizer.get_vocab_size(with_added_tokens=True)


def make_model(out: Path, vocab: int) -> None:
    rng = np.random.default_rng(0)

    def weights(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * 0.05).astype(np.float32)

    writer = GGUFWriter(str(out / "model.gguf"), "qwen3")
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(INTERMEDIATE)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_key_length(HEAD_DIM)
    writer.add_value_length(HEAD_DIM)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(1_000_000.0)

    def matmul(name: str, rows: int, cols: int) -> None:
        # Q8_0, so the test exercises the quantized kernels a real model uses.
        data = quantize(weights(rows, cols), GGMLQuantizationType.Q8_0)
        writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType.Q8_0)

    def norm(name: str, size: int) -> None:
        writer.add_tensor(name, np.ones(size, dtype=np.float32))

    writer.add_tensor("token_embd.weight", weights(vocab, HIDDEN))
    for i in range(LAYERS):
        p = f"blk.{i}"
        matmul(f"{p}.attn_q.weight", HEADS * HEAD_DIM, HIDDEN)
        matmul(f"{p}.attn_k.weight", KV_HEADS * HEAD_DIM, HIDDEN)
        matmul(f"{p}.attn_v.weight", KV_HEADS * HEAD_DIM, HIDDEN)
        matmul(f"{p}.attn_output.weight", HIDDEN, HEADS * HEAD_DIM)
        norm(f"{p}.attn_q_norm.weight", HEAD_DIM)
        norm(f"{p}.attn_k_norm.weight", HEAD_DIM)
        norm(f"{p}.attn_norm.weight", HIDDEN)
        norm(f"{p}.ffn_norm.weight", HIDDEN)
        matmul(f"{p}.ffn_gate.weight", INTERMEDIATE, HIDDEN)
        matmul(f"{p}.ffn_up.weight", INTERMEDIATE, HIDDEN)
        matmul(f"{p}.ffn_down.weight", HIDDEN, INTERMEDIATE)
    norm("output_norm.weight", HIDDEN)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "models/tiny")
    out.mkdir(parents=True, exist_ok=True)
    vocab = make_tokenizer(out)
    make_model(out, vocab)
    print(f"wrote {out}/model.gguf and {out}/tokenizer.json ({vocab} tokens)")


if __name__ == "__main__":
    main()
