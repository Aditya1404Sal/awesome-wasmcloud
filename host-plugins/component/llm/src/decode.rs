//! Incremental detokenization.
//!
//! Decoding one token at a time would split multi-byte characters and drop the
//! leading spaces some tokenizers only render in context, so this decodes the
//! window since the last release and releases only what it adds — and only
//! once that no longer ends in a partial character. The approach follows
//! Candle's `TokenOutputStream` example.

use tokenizers::Tokenizer;

const REPLACEMENT: char = '\u{FFFD}';

#[derive(Default)]
pub struct IncrementalDecoder {
    tokens: Vec<u32>,
    /// Start of the window decoded for context.
    prev: usize,
    /// End of what has already been released.
    current: usize,
}

impl IncrementalDecoder {
    /// Feed one token; returns the text it completes, if any.
    pub fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> Result<Option<String>, String> {
        let before = self.decode(tokenizer, self.prev, self.current)?;
        self.tokens.push(token);
        let after = self.decode(tokenizer, self.prev, self.tokens.len())?;
        if after.len() > before.len()
            && !after.ends_with(REPLACEMENT)
            && after.is_char_boundary(before.len())
            && after.starts_with(&before)
        {
            self.prev = self.current;
            self.current = self.tokens.len();
            Ok(Some(after[before.len()..].to_string()))
        } else {
            Ok(None)
        }
    }

    /// Release whatever is still pending, e.g. before switching modes or at
    /// the end of generation.
    pub fn flush(&mut self, tokenizer: &Tokenizer) -> Result<Option<String>, String> {
        let before = self.decode(tokenizer, self.prev, self.current)?;
        let after = self.decode(tokenizer, self.prev, self.tokens.len())?;
        self.tokens.clear();
        self.prev = 0;
        self.current = 0;
        if after.len() > before.len() && after.is_char_boundary(before.len()) {
            Ok(Some(after[before.len()..].to_string()))
        } else {
            Ok(None)
        }
    }

    fn decode(&self, tokenizer: &Tokenizer, from: usize, to: usize) -> Result<String, String> {
        if from >= to {
            return Ok(String::new());
        }
        tokenizer
            .decode(&self.tokens[from..to], true)
            .map_err(|e| format!("failed to decode tokens: {e}"))
    }
}
