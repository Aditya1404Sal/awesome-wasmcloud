//! Per-binding configuration: what the operator's `host.plugins` binding (and,
//! under `workloadConfig: allow`, the workload) says about the model it serves.

/// Generated-token ceiling when neither the request nor the binding sets one.
const DEFAULT_MAX_TOKENS: u32 = 512;
/// Tokens of history the penalty looks back over.
const DEFAULT_REPEAT_LAST_N: usize = 64;

/// Every key this backend reads. Anything else is refused at bind, so a typo
/// (`model_path`) fails the deploy instead of silently falling back.
const KNOWN_KEYS: &[&str] = &[
    "backend",
    "model",
    "model-path",
    "tokenizer-path",
    "allowed-models",
    "context-length",
    "max-tokens",
    "temperature",
    "top-p",
    "top-k",
    "repeat-penalty",
    "thinking",
];

/// One binding's model and its generation defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct BindingConfig {
    /// The id requests name this model by, and `list-models` reports.
    pub model_id: String,
    /// GGUF weights, as a path inside the plugin's volumes.
    pub model_path: String,
    /// `tokenizer.json` matching the weights.
    pub tokenizer_path: String,
    /// Ceiling on what a request may name. Empty means only `model_id`.
    pub allowed_models: Vec<String>,
    /// Overrides the context window the GGUF metadata declares.
    pub context_length: Option<u32>,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// Whether the model reasons before answering (Qwen3's thinking mode).
    pub thinking: bool,
}

impl BindingConfig {
    /// Parse a binding's resolved `(key, value)` config.
    pub fn parse(config: &[(String, String)]) -> Result<Self, String> {
        let get = |key: &str| {
            config
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.trim())
        };

        if let Some((key, _)) = config
            .iter()
            .find(|(k, _)| !KNOWN_KEYS.contains(&k.as_str()))
        {
            return Err(format!(
                "unknown config key `{key}`; this backend reads {}",
                KNOWN_KEYS.join(", ")
            ));
        }
        if let Some(backend) = get("backend")
            && backend != "local"
        {
            return Err(format!(
                "backend `{backend}` is not served by this plugin, which runs `local` models only"
            ));
        }

        let model_path = get("model-path")
            .filter(|v| !v.is_empty())
            .ok_or("`model-path` is required: the GGUF file inside the plugin's volume")?
            .to_string();
        let tokenizer_path = get("tokenizer-path")
            .filter(|v| !v.is_empty())
            .ok_or("`tokenizer-path` is required: the tokenizer.json matching the model")?
            .to_string();
        let model_id = match get("model").filter(|v| !v.is_empty()) {
            Some(id) => id.to_string(),
            None => std::path::Path::new(&model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("`model` is required when `model-path` names no file")?
                .to_string(),
        };
        let allowed_models: Vec<String> = get("allowed-models")
            .map(|v| {
                v.split(',')
                    .map(str::trim)
                    .filter(|m| !m.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();
        if !allowed_models.is_empty() && !allowed_models.contains(&model_id) {
            return Err(format!(
                "`allowed-models` does not include this binding's own model `{model_id}`"
            ));
        }

        let max_tokens = parse_or(get("max-tokens"), "max-tokens", DEFAULT_MAX_TOKENS)?;
        if max_tokens == 0 {
            return Err("`max-tokens` must be at least 1".into());
        }

        Ok(Self {
            model_id,
            model_path,
            tokenizer_path,
            allowed_models,
            context_length: parse_opt(get("context-length"), "context-length")?,
            max_tokens,
            temperature: parse_opt(get("temperature"), "temperature")?,
            top_p: parse_opt(get("top-p"), "top-p")?,
            top_k: parse_opt(get("top-k"), "top-k")?,
            repeat_penalty: parse_or(get("repeat-penalty"), "repeat-penalty", 1.1)?,
            repeat_last_n: DEFAULT_REPEAT_LAST_N,
            thinking: parse_or(get("thinking"), "thinking", false)?,
        })
    }

    /// Fail the bind, rather than the first request, when a file is missing.
    /// Cheap: it reads metadata only, never the weights.
    pub fn check_files(&self) -> Result<(), String> {
        for (key, path) in [
            ("model-path", &self.model_path),
            ("tokenizer-path", &self.tokenizer_path),
        ] {
            let meta = std::fs::metadata(path).map_err(|e| {
                format!(
                    "`{key}` {path}: {e}; is the directory mounted under the plugin's `volumes`?"
                )
            })?;
            if !meta.is_file() {
                return Err(format!("`{key}` {path} is not a file"));
            }
        }
        Ok(())
    }
}

fn parse_opt<T: std::str::FromStr>(value: Option<&str>, key: &str) -> Result<Option<T>, String> {
    value
        .map(|v| {
            v.parse()
                .map_err(|_| format!("`{key}` has an invalid value {v:?}"))
        })
        .transpose()
}

fn parse_or<T: std::str::FromStr>(value: Option<&str>, key: &str, default: T) -> Result<T, String> {
    Ok(parse_opt(value, key)?.unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parses_a_minimal_binding_and_derives_the_model_id() {
        let c = BindingConfig::parse(&cfg(&[
            ("model-path", "/models/Qwen3-0.6B-Q4_K_M.gguf"),
            ("tokenizer-path", "/models/tokenizer.json"),
        ]))
        .unwrap();
        assert_eq!(c.model_id, "Qwen3-0.6B-Q4_K_M");
        assert_eq!(c.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(!c.thinking);
        assert!(c.allowed_models.is_empty());
    }

    #[test]
    fn later_keys_win_like_layered_config() {
        let c = BindingConfig::parse(&cfg(&[
            ("model-path", "/models/a.gguf"),
            ("tokenizer-path", "/models/tokenizer.json"),
            ("max-tokens", "64"),
            ("max-tokens", "128"),
        ]))
        .unwrap();
        assert_eq!(c.max_tokens, 128);
    }

    #[test]
    fn refuses_what_would_otherwise_be_silently_ignored() {
        for (pairs, expected) in [
            (
                vec![("tokenizer-path", "/t.json")],
                "`model-path` is required",
            ),
            (
                vec![
                    ("model-path", "/m.gguf"),
                    ("tokenizer-path", "/t.json"),
                    ("model_path", "/x"),
                ],
                "unknown config key `model_path`",
            ),
            (
                vec![
                    ("backend", "openai"),
                    ("model-path", "/m.gguf"),
                    ("tokenizer-path", "/t.json"),
                ],
                "runs `local` models only",
            ),
            (
                vec![
                    ("model-path", "/m.gguf"),
                    ("tokenizer-path", "/t.json"),
                    ("max-tokens", "lots"),
                ],
                "`max-tokens` has an invalid value",
            ),
            (
                vec![
                    ("model-path", "/m.gguf"),
                    ("tokenizer-path", "/t.json"),
                    ("allowed-models", "other"),
                ],
                "does not include this binding's own model",
            ),
        ] {
            let err = BindingConfig::parse(&cfg(&pairs)).unwrap_err();
            assert!(err.contains(expected), "expected {expected:?}, got {err:?}");
        }
    }
}
