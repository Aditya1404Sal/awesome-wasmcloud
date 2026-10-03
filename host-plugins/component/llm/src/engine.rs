//! The local backend: a quantized Qwen3 model run on the CPU with Candle,
//! inside the plugin's own sandbox.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use tokenizers::Tokenizer;

use crate::config::BindingConfig;
use crate::decode::IncrementalDecoder;
use crate::stop::StopFilter;

/// Context window assumed when neither the binding nor the GGUF metadata says.
const FALLBACK_CONTEXT_LENGTH: u32 = 32_768;

/// A loaded model: weights, tokenizer, and the special tokens generation
/// steers by. Loading reads the whole GGUF file, so one is shared by every
/// binding and workload naming the same files.
pub struct LocalModel {
    weights: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    /// Tokens that end the assistant's turn.
    eos: Vec<u32>,
    /// Qwen3's `<think>` / `</think>`, which bracket reasoning.
    think: Option<(u32, u32)>,
    pub context_length: u32,
}

/// One increment of output.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Text(String),
    Reasoning(String),
}

/// How to sample one generation.
pub struct Params {
    pub max_tokens: u32,
    pub sampling: Sampling,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub stops: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// End of turn, or a stop sequence.
    Stop,
    /// `max-tokens` or the context window.
    Length,
    /// The consumer stopped listening.
    Cancelled,
}

pub struct Outcome {
    pub finish: Finish,
    pub output_tokens: u32,
}

/// Why releasing a piece ended generation, if it did.
enum Released {
    Continue,
    Stopped,
    Cancelled,
}

impl LocalModel {
    pub fn load(cfg: &BindingConfig) -> Result<Self, String> {
        let device = Device::Cpu;
        // Buffered: GGUF metadata is many small values (Qwen3's carries its
        // whole vocabulary), and unbuffered, every one is a WASI host call.
        let file =
            std::fs::File::open(&cfg.model_path).map_err(|e| format!("{}: {e}", cfg.model_path))?;
        let mut file = std::io::BufReader::with_capacity(1 << 20, file);
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| format!("{} is not a readable GGUF file: {e}", cfg.model_path))?;
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned();
        if arch.as_deref() != Some("qwen3") {
            return Err(format!(
                "{} has architecture {arch:?}; this backend runs qwen3 GGUF models",
                cfg.model_path
            ));
        }
        let gguf_context = content
            .metadata
            .get("qwen3.context_length")
            .and_then(|v| v.to_u32().ok());
        let weights = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| format!("failed to load {}: {e}", cfg.model_path))?;

        let tokenizer = Tokenizer::from_file(&cfg.tokenizer_path)
            .map_err(|e| format!("failed to load {}: {e}", cfg.tokenizer_path))?;
        let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();
        if eos.is_empty() {
            return Err(format!(
                "{} has no `<|im_end|>` token; is it the tokenizer for this Qwen3 model?",
                cfg.tokenizer_path
            ));
        }
        let think = tokenizer
            .token_to_id("<think>")
            .zip(tokenizer.token_to_id("</think>"));

        Ok(Self {
            weights,
            tokenizer,
            device,
            eos,
            think,
            context_length: cfg
                .context_length
                .or(gguf_context)
                .unwrap_or(FALLBACK_CONTEXT_LENGTH),
        })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(prompt, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| format!("failed to tokenize the prompt: {e}"))
    }

    /// Generate from `prompt`, handing each piece to `emit` as it is
    /// produced. `emit` returns `false` once its consumer has gone, which ends
    /// generation so no further compute is spent on it.
    pub async fn generate(
        &mut self,
        prompt: &[u32],
        params: &Params,
        mut emit: impl AsyncFnMut(Piece) -> bool,
    ) -> Result<Outcome, String> {
        // One model serves conversation after conversation; nothing from the
        // last one may leak into this one.
        self.weights.clear_kv_cache();
        let budget = (self.context_length as usize).saturating_sub(prompt.len());
        let max_tokens = (params.max_tokens as usize).min(budget);

        let mut sampler = LogitsProcessor::from_sampling(params.seed, params.sampling.clone());
        let mut decoder = IncrementalDecoder::default();
        let mut stop = StopFilter::new(params.stops.clone());
        let mut reasoning = false;
        let mut at_mode_start = true;
        let mut generated: Vec<u32> = Vec::new();
        let mut stop_sequence = false;
        let mut logits = self.forward(prompt, 0)?;

        let finish = 'generate: loop {
            if generated.len() >= max_tokens {
                break Finish::Length;
            }
            let next = self.sample(&mut sampler, &logits, &generated, params)?;
            if self.eos.contains(&next) {
                break Finish::Stop;
            }
            generated.push(next);

            let mut pieces = Vec::with_capacity(1);
            match self.think {
                Some((open, close)) if next == open || next == close => {
                    if let Some(text) = decoder.flush(&self.tokenizer)? {
                        pieces.push((reasoning, text));
                    }
                    reasoning = next == open;
                    at_mode_start = true;
                }
                _ => {
                    if let Some(text) = decoder.push(&self.tokenizer, next)? {
                        pieces.push((reasoning, text));
                    }
                }
            }
            for (is_reasoning, text) in pieces {
                match release(is_reasoning, text, &mut stop, &mut at_mode_start, &mut emit).await {
                    Released::Continue => {}
                    Released::Stopped => {
                        stop_sequence = true;
                        break 'generate Finish::Stop;
                    }
                    Released::Cancelled => break 'generate Finish::Cancelled,
                }
            }

            // The plugin's store serves every workload; a generation that
            // never awaited would hold all of them up until it finished.
            #[cfg(target_arch = "wasm32")]
            wit_bindgen::yield_async().await;
            logits = self.forward(&[next], prompt.len() + generated.len() - 1)?;
        };

        // A turn that ended on its own still owes whatever was held back; one
        // cut by a stop sequence or a departed consumer owes nothing.
        if finish == Finish::Length || (finish == Finish::Stop && !stop_sequence) {
            if let Some(text) = decoder.flush(&self.tokenizer)?
                && let Released::Cancelled =
                    release(reasoning, text, &mut stop, &mut at_mode_start, &mut emit).await
            {
                return Ok(outcome(Finish::Cancelled, &generated));
            }
            let tail = stop.finish();
            if !tail.is_empty() && !emit(Piece::Text(tail)).await {
                return Ok(outcome(Finish::Cancelled, &generated));
            }
        }
        Ok(outcome(finish, &generated))
    }

    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Tensor, String> {
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| e.to_string())?;
        self.weights
            .forward(&input, offset)
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| format!("inference failed: {e}"))
    }

    fn sample(
        &self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
        generated: &[u32],
        params: &Params,
    ) -> Result<u32, String> {
        let penalized;
        let logits = if params.repeat_penalty == 1.0 || generated.is_empty() {
            logits
        } else {
            let from = generated.len().saturating_sub(params.repeat_last_n);
            penalized = candle_transformers::utils::apply_repeat_penalty(
                logits,
                params.repeat_penalty,
                &generated[from..],
            )
            .map_err(|e| e.to_string())?;
            &penalized
        };
        sampler.sample(logits).map_err(|e| e.to_string())
    }
}

fn outcome(finish: Finish, generated: &[u32]) -> Outcome {
    Outcome {
        finish,
        output_tokens: generated.len() as u32,
    }
}

async fn release(
    reasoning: bool,
    text: String,
    stop: &mut StopFilter,
    at_mode_start: &mut bool,
    emit: &mut impl AsyncFnMut(Piece) -> bool,
) -> Released {
    // Qwen3 opens each mode with newlines that carry nothing for a reader.
    let text = if *at_mode_start {
        let trimmed = text.trim_start_matches('\n');
        if trimmed.is_empty() {
            return Released::Continue;
        }
        *at_mode_start = false;
        trimmed.to_string()
    } else {
        text
    };
    if reasoning {
        return match emit(Piece::Reasoning(text)).await {
            true => Released::Continue,
            false => Released::Cancelled,
        };
    }
    let filtered = stop.push(&text);
    if !filtered.text.is_empty() && !emit(Piece::Text(filtered.text)).await {
        return Released::Cancelled;
    }
    if filtered.stopped {
        Released::Stopped
    } else {
        Released::Continue
    }
}

#[cfg(test)]
mod tests {
    //! Against the tiny random model `scripts/make-test-model.py` writes:
    //! its output is gibberish, so these check the mechanics, not the text.

    use super::*;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/tiny");

    fn load() -> LocalModel {
        let cfg = BindingConfig::parse(&[
            ("model-path".into(), format!("{DIR}/model.gguf")),
            ("tokenizer-path".into(), format!("{DIR}/tokenizer.json")),
        ])
        .unwrap();
        LocalModel::load(&cfg).expect("tiny model loads")
    }

    fn params(max_tokens: u32) -> Params {
        Params {
            max_tokens,
            sampling: Sampling::ArgMax,
            seed: 7,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            stops: vec![],
        }
    }

    fn run(model: &mut LocalModel, prompt: &[u32], params: &Params) -> (Outcome, Vec<Piece>) {
        let mut pieces = Vec::new();
        let outcome = futures::executor::block_on(model.generate(prompt, params, async |p| {
            pieces.push(p);
            true
        }))
        .unwrap();
        (outcome, pieces)
    }

    #[test]
    #[ignore = "needs models/tiny: python3 scripts/make-test-model.py models/tiny"]
    fn generates_within_its_budget_and_repeats_exactly() {
        let mut model = load();
        assert_eq!(
            model.context_length, 512,
            "context comes from GGUF metadata"
        );
        let prompt = model
            .encode("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")
            .unwrap();

        let (first, pieces) = run(&mut model, &prompt, &params(16));
        assert!(first.output_tokens >= 1 && first.output_tokens <= 16);
        assert!(!pieces.is_empty());

        // Same seed, same prompt: identical output, which only holds if the
        // first conversation left nothing behind in the KV cache.
        let (second, again) = run(&mut model, &prompt, &params(16));
        assert_eq!(first.output_tokens, second.output_tokens);
        assert_eq!(pieces, again);
    }

    #[test]
    #[ignore = "needs models/tiny: python3 scripts/make-test-model.py models/tiny"]
    fn stops_spending_once_the_consumer_leaves() {
        let mut model = load();
        let prompt = model.encode("<|im_start|>user\nhi<|im_end|>\n").unwrap();
        let mut seen = 0;
        let outcome =
            futures::executor::block_on(model.generate(&prompt, &params(64), async |_| {
                seen += 1;
                false
            }))
            .unwrap();
        assert_eq!(outcome.finish, Finish::Cancelled);
        assert_eq!(seen, 1);
        assert!(outcome.output_tokens < 64);
    }
}
