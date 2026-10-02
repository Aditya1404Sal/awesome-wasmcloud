//! Stop sequences over streamed text.
//!
//! A stop sequence can arrive split across tokens, so text that might be the
//! start of one is held back until the next piece settles it. Without that, a
//! caller would already have received `"EN"` by the time `"D"` completed the
//! stop sequence `"END"`.

pub struct StopFilter {
    stops: Vec<String>,
    /// Text received but not yet released.
    pending: String,
}

/// What a piece of text did to the filter.
pub struct Filtered {
    /// Text now safe to release.
    pub text: String,
    /// Whether a stop sequence was reached; nothing after it is released.
    pub stopped: bool,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            pending: String::new(),
        }
    }

    pub fn push(&mut self, piece: &str) -> Filtered {
        self.pending.push_str(piece);
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min()
        {
            let text = self.pending[..at].to_string();
            self.pending.clear();
            return Filtered {
                text,
                stopped: true,
            };
        }
        let keep = self.held_back();
        let release = self.pending.len() - keep;
        let text = self.pending[..release].to_string();
        self.pending.drain(..release);
        Filtered {
            text,
            stopped: false,
        }
    }

    /// Whatever is still held back, once no more text is coming.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Length of the longest suffix of `pending` that is a proper prefix of
    /// some stop sequence, on a char boundary.
    fn held_back(&self) -> usize {
        let mut longest = 0;
        for stop in &self.stops {
            for (len, _) in stop.char_indices().skip(1) {
                if len > longest
                    && self.pending.len() >= len
                    && self.pending.is_char_boundary(self.pending.len() - len)
                    && self.pending.ends_with(&stop[..len])
                {
                    longest = len;
                }
            }
        }
        longest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut f = StopFilter::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for piece in pieces {
            let r = f.push(piece);
            out.push_str(&r.text);
            if r.stopped {
                return (out, true);
            }
        }
        out.push_str(&f.finish());
        (out, false)
    }

    #[test]
    fn passes_text_through_without_stops() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), false));
    }

    #[test]
    fn stops_on_a_sequence_split_across_pieces_without_leaking_it() {
        let mut f = StopFilter::new(vec!["END".into()]);
        assert_eq!(f.push("hello EN").text, "hello ");
        let r = f.push("D and more");
        assert_eq!(r.text, "");
        assert!(r.stopped);
    }

    #[test]
    fn releases_held_text_that_turns_out_not_to_be_a_stop() {
        assert_eq!(run(&["END"], &["E", "N", "o"]), ("ENo".into(), false));
        assert_eq!(run(&["END"], &["the E"]), ("the E".into(), false));
    }

    #[test]
    fn earliest_stop_wins() {
        assert_eq!(run(&["b", "a"], &["xab"]), ("x".into(), true));
    }

    #[test]
    fn handles_multibyte_text() {
        assert_eq!(run(&["éé"], &["café", "é!"]), ("caf".into(), true));
        assert_eq!(run(&["→x"], &["a→", "b"]), ("a→b".into(), false));
    }
}
