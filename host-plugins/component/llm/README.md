# llm

A wasmCloud **host component plugin** that serves the draft `wasmcloud:llm` interface from a model on the host's disk. It runs a quantized Qwen3 GGUF model on the CPU with [Candle](https://github.com/huggingface/candle), inside the plugin's own sandbox. A workload imports `wasmcloud:llm/inference` and gets completions, streamed or whole, without ever seeing the model files, the runtime, or where they live.

> **Work in progress.** `wasmcloud:llm` is a draft interface, and this plugin needs host-component-plugin `volumes`, which are not in a wasmCloud release yet (see [Requirements](#requirements)). Expect both to change. The full path (HTTP workload → plugin → Candle → streamed answer) is verified with the tiny test model below; answers and speed with the real Qwen3-0.6B weights are not measured yet.

```
workload ──wasmcloud:llm/inference──▶ llm plugin (own store) ──▶ Candle ──▶ /models/*.gguf
           (implements ..) label                  │                         (plugin volume)
                                                  └─ binding config picks the model
```

## What it serves

| Interface | Notes |
|---|---|
| `wasmcloud:llm/inference` | `complete` returns the whole answer. `complete-streaming` returns the text as a `stream<string>`, plus a `generation` resource whose `finish` reports finish reason, usage, and any reasoning. Dropping the stream cancels generation. |
| `wasmcloud:llm/models` | `list-models` reports the binding's model. |
| `wasmcloud:host/workload-lifecycle` | Validates each binding when a workload deploys, so a wrong path fails the deploy rather than the first request. |

It does not export `wasmcloud:llm/embeddings`: the chat models it runs are not embedding models.

## Requirements

- **A `wash` built with host component plugin volumes.** Host component plugins cannot mount host directories in wasmCloud v2.10.x. This plugin needs the `volumes` field from the `feat/host-plugin-volumes` branch of [Aditya1404Sal/wasmCloud](https://github.com/Aditya1404Sal/wasmCloud/tree/feat/host-plugin-volumes), built with the `host-component-plugins` feature:

  ```console
  cargo build --release -p wash --features host-component-plugins
  ```

  Tested with that branch on top of wasmCloud `main` at v2.10.3.
- Rust with the `wasm32-wasip1` target (`rustup target add wasm32-wasip1`).
- A Qwen3 GGUF model and its `tokenizer.json` (see [Get a model](#get-a-model)).

## Build

```console
wash build              # the plugin: target/wasm32-wasip1/release/llm_plugin.wasm
cd example && wash build # the example workload
```

The plugin builds as a `wasm32-wasip1` core module that `wash build` wraps into a component, the way wasmCloud builds its own P3 fixtures. It is not built for `wasm32-wasip2` because Candle reaches `std::os::wasi` (through `zip`), which is still unstable on that target.

## Get a model

```console
./fetch-models.sh   # Qwen3-0.6B Q8_0 + tokenizer.json into models/qwen3-0.6b/
```

The weights are roughly 600 MB and are not in git. Override `MODEL_URL` and `TOKENIZER_URL` to fetch another Qwen3 GGUF, then update `model-path` in the binding config to match.

Without network access to Hugging Face, `scripts/make-test-model.py` writes a tiny, randomly initialized model with the same layout (`pip install gguf numpy tokenizers`). It exercises the whole pipeline, but its output is gibberish.

## Run the example

```console
cd example
wash dev
curl -N -d 'What is WebAssembly?' http://127.0.0.1:8000/
```

`example/.wash/config.yaml` loads the plugin, mounts `../models/qwen3-0.6b` at `/models` in the plugin's store, and binds the workload's unlabeled `wasmcloud:llm` import to that model. The response streams as it is generated and ends with a usage line like:

```
[qwen3-0.6b · FinishReason::Stop · 14 prompt + 87 generated tokens]
```

## Binding configuration

Each binding is configured by the operator, under the plugin's `host.plugins` entry or the workload's `host_interfaces` entry. The paths are inside the plugin's volumes.

| Key | Required | Meaning |
|---|---|---|
| `model-path` | yes | The GGUF weights. Only `qwen3` architecture models are supported. |
| `tokenizer-path` | yes | The `tokenizer.json` matching the weights. |
| `model` | no | The id requests name the model by. Defaults to the GGUF file name. |
| `allowed-models` | no | Comma-separated ceiling on what a request may name. |
| `context-length` | no | Overrides the context window the GGUF metadata declares. |
| `max-tokens` | no | Default ceiling on generated tokens (512). |
| `temperature`, `top-p`, `top-k` | no | Default sampling. Unset uses Qwen3's recommended values for the mode. |
| `repeat-penalty` | no | Default 1.1 over the last 64 tokens. |
| `thinking` | no | `true` lets Qwen3 reason before answering. Off by default; a request can set `extra: [("enable_thinking", "true")]`. |
| `backend` | no | Must be `local` if set. |

Unknown keys are refused at bind, so a typo like `model_path` fails the deploy instead of being ignored.

A `wash host` operator config serving a binding named `local`, which a workload imports as `(implements local)`:

```yaml
host:
  plugins:
    - id: wasmcloud-llm
      file: ./llm_plugin.wasm
      volumes:
        - hostPath: /var/lib/models/qwen3-0.6b
          mountPath: /models
          readOnly: true
      workloadConfig: deny
      hostOwnedKeys: [model-path, tokenizer-path, allowed-models]
      bindings:
        local:
          config:
            model: qwen3-0.6b
            model-path: /models/Qwen3-0.6B-Q8_0.gguf
            tokenizer-path: /models/tokenizer.json
```

## How it works

- **One store, many workloads.** The plugin is a single long-lived store. Each call is routed to its binding by the caller's workload id and `(implements ..)` label (`wasmcloud:host/identity`).
- **Models load once.** A model loads on its first request and is shared by every binding naming the same files. It is dropped when no binding names it anymore.
- **One generation per model at a time.** A model has one KV cache, so generations on it are serialized. The cache is cleared between conversations.
- **Cooperative.** Generation yields between tokens, so one long answer does not stall the plugin's other callers. Each token is still computed synchronously on one core.
- **Cancellation.** If the consumer drops the text stream, generation stops at the next token.
- **Reasoning.** Qwen3's `<think>` block goes to `summary.reasoning`, never into the text stream.
- **Prompt safety.** Chat control markers (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) are stripped from message content, so user text cannot forge a turn.

## Limitations

- CPU only, single-threaded, with WASM SIMD. A 0.6B model is the practical size.
- Qwen3 GGUF models only.
- No tool calling, structured output, media input, or embeddings. Requests asking for them are refused with `invalid-request`.
- Text streams as `string`, not as a variant of text, reasoning, and tool calls. The host relays only streams of scalars and strings between a plugin's store and a workload's.
- Candle is pinned to a commit on its `main` branch, because the 0.11.0 release does not compile for wasm32 with SIMD. Move to the next release when it ships.

## Test

```console
cargo test                                                        # unit tests
python3 scripts/make-test-model.py models/tiny && cargo test --release -- --include-ignored  # plus model tests
```

The model tests load the tiny model and check that generation respects `max-tokens`, that repeated runs match exactly (so nothing leaks between conversations), and that dropping the consumer stops generation.

## License

Apache-2.0. See [LICENSE](LICENSE).
