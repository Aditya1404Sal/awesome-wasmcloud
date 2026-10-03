//! Example workload for the llm plugin: an HTTP endpoint that sends the request
//! body to `wasmcloud:llm/inference` as a user message and streams the model's
//! answer back as the response body, ending with a line of usage.
//!
//! It knows nothing about the model — no path, no weights, no endpoint. Which
//! model answers is the binding's business, configured on the host.

mod bindings {
    wit_bindgen::generate!({ generate_all });
}

use bindings::exports::wasi::http::handler::Guest;
use bindings::wasi::http::types::{ErrorCode, Fields, Request, Response};
use bindings::wasmcloud::llm::inference::{self, Summary};
use bindings::wasmcloud::llm::types::{ContentPart, Error, GenerationOptions, Message};
use wit_bindgen::StreamResult;

/// Prompts larger than this are cut off rather than read into memory.
const MAX_PROMPT_BYTES: usize = 16 * 1024;

struct Component;

impl Guest for Component {
    async fn handle(request: Request) -> Result<Response, ErrorCode> {
        let prompt = read_prompt(request).await;
        if prompt.is_empty() {
            return Ok(text_response(
                400,
                "send a prompt as the request body\n".into(),
            ));
        }

        let messages = vec![Message::User(vec![ContentPart::Text(prompt)])];
        let options = GenerationOptions {
            model: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: vec![],
            seed: None,
            tools: vec![],
            tool_choice: None,
            response_format: None,
            extra: vec![],
        };
        let (mut text, generation) = match inference::complete_streaming(messages, options).await {
            Ok(streaming) => streaming,
            Err(err) => return Ok(text_response(502, format!("{}\n", describe(&err)))),
        };

        let (mut body, body_rx) = bindings::wit_stream::new::<u8>();
        let (trailers, trailers_rx) = bindings::wit_future::new(|| Ok(None));
        wit_bindgen::spawn_local(async move {
            while let Some(piece) = text.next().await {
                // A client that hung up gets nothing more; dropping `text` on
                // return tells the plugin to stop generating.
                if !body.write_all(piece.into_bytes()).await.is_empty() {
                    return;
                }
            }
            let line = match generation.finish().await {
                Ok(summary) => usage_line(&summary),
                Err(err) => format!("\n\n[{}]\n", describe(&err)),
            };
            body.write_all(line.into_bytes()).await;
            drop(body);
            let _ = trailers.write(Ok(None)).await;
        });

        let (response, _) = Response::new(Fields::new(), Some(body_rx), trailers_rx);
        Ok(response)
    }
}

async fn read_prompt(request: Request) -> String {
    let (done, done_rx) = bindings::wit_future::new(|| Ok(()));
    let (mut body, _trailers) = Request::consume_body(request, done_rx);
    let mut bytes = Vec::new();
    while bytes.len() < MAX_PROMPT_BYTES {
        let (status, chunk) = body.read(Vec::with_capacity(4096)).await;
        bytes.extend_from_slice(&chunk);
        if !matches!(status, StreamResult::Complete(_)) {
            break;
        }
    }
    drop(body);
    let _ = done.write(Ok(())).await;
    bytes.truncate(MAX_PROMPT_BYTES);
    String::from_utf8_lossy(&bytes).trim().to_string()
}

fn usage_line(summary: &Summary) -> String {
    match &summary.usage {
        Some(usage) => format!(
            "\n\n[{} · {:?} · {} prompt + {} generated tokens]\n",
            summary.model, summary.finish_reason, usage.input_tokens, usage.output_tokens
        ),
        None => format!("\n\n[{} · {:?}]\n", summary.model, summary.finish_reason),
    }
}

fn describe(err: &Error) -> String {
    format!("wasmcloud:llm error: {err:?}")
}

fn text_response(status: u16, text: String) -> Response {
    let (mut body, body_rx) = bindings::wit_stream::new::<u8>();
    let (trailers, trailers_rx) = bindings::wit_future::new(|| Ok(None));
    wit_bindgen::spawn_local(async move {
        body.write_all(text.into_bytes()).await;
        drop(body);
        let _ = trailers.write(Ok(None)).await;
    });
    let (response, _) = Response::new(Fields::new(), Some(body_rx), trailers_rx);
    let _ = response.set_status_code(status);
    response
}

bindings::export!(Component with_types_in bindings);
