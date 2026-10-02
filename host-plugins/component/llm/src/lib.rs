//! A wasmCloud host component plugin serving `wasmcloud:llm` from a GGUF
//! model on the host's disk, run in the plugin's own sandbox with Candle.
//!
//! The plugin is one long-lived store shared by every workload that imports
//! `wasmcloud:llm`. Each workload's bindings are validated when it deploys
//! (`on-workload-bind`) and looked up on every call by the caller's workload
//! id and `(implements ..)` label. Models load on first use and stay loaded,
//! shared by every binding naming the same files, until no binding does.

mod bindings {
    wit_bindgen::generate!({ world: "llm-plugin", generate_all });
}

mod chat;
mod config;
mod decode;
mod engine;
mod stop;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::Shared;
use futures::lock::{Mutex, OwnedMutexGuard};
use wit_bindgen::StreamReader;

use bindings::exports::wasmcloud::host::workload_lifecycle::{
    Guest as LifecycleGuest, WorkloadInfo,
};
use bindings::exports::wasmcloud::llm::inference::{
    Completion, Generation, Guest as InferenceGuest, GuestGeneration, Summary,
};
use bindings::exports::wasmcloud::llm::models::{Guest as ModelsGuest, ModelInfo};
use bindings::exports::wasmcloud::llm::types::{
    AssistantMessage, ContentPart, Error, FinishReason, GenerationOptions, Message, Usage,
};
use bindings::wasmcloud::host::identity;
use config::BindingConfig;
use engine::{Finish, LocalModel, Piece};

/// A model slot: empty until first use, then the loaded model. The lock is
/// held for a whole generation, because a model has one KV cache.
type ModelSlot = Arc<Mutex<Option<LocalModel>>>;

/// A model is identified by the files it loads from.
type ModelKey = (String, String);

thread_local! {
    /// Workload id → binding label (`None` for a plain import) → config.
    static BINDINGS: RefCell<HashMap<String, HashMap<Option<String>, Rc<BindingConfig>>>> =
        RefCell::new(HashMap::new());
    static MODELS: RefCell<HashMap<ModelKey, ModelSlot>> = RefCell::new(HashMap::new());
}

struct Component;

impl LifecycleGuest for Component {
    async fn on_workload_bind(workload: WorkloadInfo) -> Result<(), String> {
        // Entries sharing a label are one binding; their config merges in
        // order. An entry naming only `types` carries no capability of its own.
        let mut merged: HashMap<Option<String>, Vec<(String, String)>> = HashMap::new();
        for binding in workload.interfaces.iter().filter(|b| {
            b.namespace == "wasmcloud"
                && b.package == "llm"
                && (b.interfaces.is_empty() || b.interfaces.iter().any(|i| i != "types"))
        }) {
            merged
                .entry(binding.name.clone())
                .or_default()
                .extend(binding.config.iter().cloned());
        }

        let mut parsed = HashMap::with_capacity(merged.len());
        for (name, config) in merged {
            let label = name.as_deref().unwrap_or("(unlabeled)");
            let config = BindingConfig::parse(&config)
                .and_then(|c| c.check_files().map(|()| c))
                .map_err(|e| format!("wasmcloud:llm binding {label}: {e}"))?;
            parsed.insert(name, Rc::new(config));
        }
        BINDINGS.with(|b| b.borrow_mut().insert(workload.id, parsed));
        Ok(())
    }

    async fn on_workload_unbind(id: String) {
        BINDINGS.with(|b| b.borrow_mut().remove(&id));
        release_unused_models();
    }
}

impl InferenceGuest for Component {
    type Generation = StreamedGeneration;

    async fn complete(
        messages: Vec<Message>,
        options: GenerationOptions,
    ) -> Result<Completion, Error> {
        let cfg = current_binding()?;
        let plan = chat::plan(&cfg, &messages, &options)?;
        let (mut model, prompt) = prepare(&cfg, &plan.prompt).await?;
        let model_ref = model.as_mut().expect("prepare loads the model");

        let mut text = String::new();
        let outcome = model_ref
            .generate(&prompt, &plan.params, async |piece| {
                if let Piece::Text(t) = piece {
                    text.push_str(&t);
                }
                true
            })
            .await
            .map_err(Error::BackendFailed)?;

        Ok(Completion {
            model: cfg.model_id.clone(),
            message: AssistantMessage {
                content: vec![ContentPart::Text(text)],
                tool_calls: vec![],
            },
            finish_reason: finish_reason(outcome.finish),
            usage: Some(usage(&prompt, outcome.output_tokens)),
        })
    }

    async fn complete_streaming(
        messages: Vec<Message>,
        options: GenerationOptions,
    ) -> Result<(StreamReader<String>, Generation), Error> {
        let cfg = current_binding()?;
        let plan = chat::plan(&cfg, &messages, &options)?;
        // Load and tokenize before answering, so a missing model or an
        // oversized conversation is the outer error rather than a stream that
        // ends at once.
        let (mut model, prompt) = prepare(&cfg, &plan.prompt).await?;

        let (mut text, text_rx) = bindings::wit_stream::new::<String>();
        let (done, done_rx) = oneshot::channel();
        wit_bindgen::spawn_local(async move {
            let model_ref = model.as_mut().expect("prepare loads the model");
            let mut reasoning = String::new();
            let result = model_ref
                .generate(&prompt, &plan.params, async |piece| match piece {
                    // `Some` hands the value back: the reader is gone.
                    Piece::Text(t) => text.write_one(t).await.is_none(),
                    Piece::Reasoning(t) => {
                        reasoning.push_str(&t);
                        true
                    }
                })
                .await;
            // Close the stream and free the model before reporting, so a
            // caller woken by `finish` finds both already done.
            drop(text);
            drop(model);
            let _ = done.send(
                result
                    .map(|outcome| Summary {
                        model: cfg.model_id.clone(),
                        finish_reason: finish_reason(outcome.finish),
                        usage: Some(usage(&prompt, outcome.output_tokens)),
                        reasoning: (!reasoning.is_empty()).then_some(reasoning),
                        tool_calls: vec![],
                    })
                    .map_err(Error::BackendFailed),
            );
        });
        Ok((
            text_rx,
            Generation::new(StreamedGeneration {
                done: done_rx.shared(),
            }),
        ))
    }
}

/// The `generation` resource: how a streamed generation ended, once it has.
pub struct StreamedGeneration {
    done: Shared<oneshot::Receiver<Result<Summary, Error>>>,
}

impl GuestGeneration for StreamedGeneration {
    async fn finish(&self) -> Result<Summary, Error> {
        self.done.clone().await.unwrap_or_else(|_| {
            Err(Error::Other("generation ended without reporting".into()))
        })
    }
}

impl ModelsGuest for Component {
    async fn list_models() -> Result<Vec<ModelInfo>, Error> {
        let cfg = current_binding()?;
        // The window is known exactly once the model has loaded; until then,
        // only what the binding declares.
        let loaded = MODELS.with(|m| m.borrow().get(&model_key(&cfg)).cloned());
        let context_length = match loaded {
            Some(slot) => slot
                .try_lock()
                .and_then(|m| m.as_ref().map(|m| m.context_length))
                .or(cfg.context_length),
            None => cfg.context_length,
        };
        Ok(vec![ModelInfo {
            id: cfg.model_id.clone(),
            context_length,
            supports_tools: false,
            supports_media: false,
        }])
    }
}

/// The binding the in-flight call arrived on.
fn current_binding() -> Result<Rc<BindingConfig>, Error> {
    let workload = identity::get_workload_id();
    let label = identity::get_binding_name();
    BINDINGS
        .with(|b| {
            b.borrow()
                .get(&workload)
                .and_then(|bindings| bindings.get(&label))
                .cloned()
        })
        .ok_or_else(|| {
            Error::InvalidRequest(format!(
                "no wasmcloud:llm binding {} is configured for this workload",
                label.as_deref().unwrap_or("(unlabeled)")
            ))
        })
}

/// Take the binding's model for one generation — loading it on first use —
/// and tokenize the prompt, refusing one that leaves no room to answer.
async fn prepare(
    cfg: &BindingConfig,
    prompt: &str,
) -> Result<(OwnedMutexGuard<Option<LocalModel>>, Vec<u32>), Error> {
    let slot = MODELS.with(|m| {
        Arc::clone(
            m.borrow_mut()
                .entry(model_key(cfg))
                .or_insert_with(|| Arc::new(Mutex::new(None))),
        )
    });
    let mut model = slot.lock_owned().await;
    if model.is_none() {
        let started = std::time::Instant::now();
        let loaded = LocalModel::load(cfg).map_err(Error::ModelUnavailable)?;
        eprintln!(
            "llm-plugin: loaded {} ({} token context) in {:.1}s",
            cfg.model_id,
            loaded.context_length,
            started.elapsed().as_secs_f32()
        );
        *model = Some(loaded);
    }
    let loaded = model.as_ref().expect("loaded above");
    let tokens = loaded.encode(prompt).map_err(Error::BackendFailed)?;
    if tokens.len() as u32 >= loaded.context_length {
        return Err(Error::ContextLengthExceeded(format!(
            "the conversation is {} tokens; the model's context is {}",
            tokens.len(),
            loaded.context_length
        )));
    }
    Ok((model, tokens))
}

/// Drop models no binding names any more. A generation still holding one
/// keeps it alive until it finishes.
fn release_unused_models() {
    let live: Vec<ModelKey> = BINDINGS.with(|b| {
        b.borrow()
            .values()
            .flat_map(|bindings| bindings.values().map(|c| model_key(c)))
            .collect()
    });
    MODELS.with(|m| m.borrow_mut().retain(|key, _| live.contains(key)));
}

fn model_key(cfg: &BindingConfig) -> ModelKey {
    (cfg.model_path.clone(), cfg.tokenizer_path.clone())
}

fn finish_reason(finish: Finish) -> FinishReason {
    match finish {
        Finish::Stop => FinishReason::Stop,
        Finish::Length => FinishReason::Length,
        Finish::Cancelled => FinishReason::Other,
    }
}

fn usage(prompt: &[u32], output_tokens: u32) -> Usage {
    Usage {
        input_tokens: prompt.len() as u32,
        output_tokens,
    }
}

bindings::export!(Component with_types_in bindings);
