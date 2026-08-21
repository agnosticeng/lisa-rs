/// Telemetry for the serving UI. The Model records per-request speed/acceptance
/// data into an `Arc<MetricsStore>`; the TUI polls it for rendering. Writers
/// (the completeness path, serialized under the Model mutex) and the UI reader
/// coordinate through one shared mutex.
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub type SharedMetrics = Arc<Mutex<Metrics>>;

/// One completed (or in-flight) request.
pub struct QueryRecord {
    pub seq: u64,
    pub started_unix_ms: u64,
    pub prompt: String, // truncated preview, single line
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub prefill_tok_s: f64,
    pub decode_ms: f64,
    pub decode_tok_s: f64,
    pub acceptance: f64,   // accepted_drafts / drafted_tokens (0..1)
    pub draft_ratio: f64,  // drafted_tokens / completion_tokens (0..1)
    pub target_forwards: usize,
    pub done: bool,
}

#[derive(Default, Clone, Copy)]
pub struct Aggregates {
    pub total_queries: u64,
    pub total_completion_tokens: u64,
    pub total_prompt_tokens: u64,
    pub total_drafted_tokens: u64,
    pub total_accepted_drafts: u64,
    pub total_target_forwards: u64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

pub struct Metrics {
    start: Instant,
    next_seq: u64,
    pub aggregates: Aggregates,
    /// Most recent first. Bounded; older entries are dropped.
    pub queries: Vec<QueryRecord>,
    /// Snapshot of the currently-running query, if any.
    pub active: Option<QueryRecord>,
    /// Fs for the active query (decode start, decode token counter).
    active_decode_start: Option<Instant>,
    active_decode_tokens: usize,
    active_prefill_start: Option<Instant>,
}

const MAX_QUERIES: usize = 40;

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            next_seq: 0,
            aggregates: Aggregates::default(),
            queries: Vec::new(),
            active: None,
            active_decode_start: None,
            active_decode_tokens: 0,
            active_prefill_start: None,
        }
    }

    pub fn shared() -> SharedMetrics {
        Arc::new(Mutex::new(Self::new()))
    }

    pub fn uptime_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub fn now_unix_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Begin tracking a request.
    pub fn begin(&mut self, prompt_preview: String, prompt_tokens: usize) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let record = QueryRecord {
            seq,
            started_unix_ms: self.now_unix_ms(),
            prompt: prompt_preview,
            prompt_tokens,
            completion_tokens: 0,
            prefill_ms: 0.0,
            prefill_tok_s: 0.0,
            decode_ms: 0.0,
            decode_tok_s: 0.0,
            acceptance: 0.0,
            draft_ratio: 0.0,
            target_forwards: 0,
            done: false,
        };
        self.active = Some(record);
        self.active_decode_start = None;
        self.active_decode_tokens = 0;
        self.active_prefill_start = Some(Instant::now());
        seq
    }

    /// Called when prefill completes (before decode begins). Records the prefill
    /// duration so the UI can show a live prefill speed while tokens stream.
    pub fn prefill_done(&mut self, prefill_ms: f64) {
        self.active_prefill_start = None;
        if let Some(active) = self.active.as_mut() {
            active.prefill_ms = prefill_ms;
            active.prefill_tok_s = if prefill_ms > 0.0 {
                active.prompt_tokens as f64 / (prefill_ms / 1_000.0)
            } else {
                0.0
            };
        }
    }

    /// Update the in-progress record's completed-token count as tokens stream.
    /// Gliding token speed from the last tick window.
    pub fn tick(&mut self, completion_tokens: usize) {
        if let Some(active) = self.active.as_mut() {
            active.completion_tokens = completion_tokens;
        }
        match (self.active_decode_start, self.active_decode_tokens) {
            (Some(start), prev) if completion_tokens > prev => {
                let secs = start.elapsed().as_secs_f64();
                if secs > 0.0 {
                    let per_second = (completion_tokens - prev) as f64 / secs;
                    if let Some(active) = self.active.as_mut() {
                        active.decode_tok_s = per_second;
                    }
                }
                self.active_decode_start = Some(Instant::now());
                self.active_decode_tokens = completion_tokens;
            }
            (None, _) if completion_tokens > 0 => {
                self.active_decode_start = Some(Instant::now());
                self.active_decode_tokens = completion_tokens;
            }
            _ => {}
        }
    }

    /// Finalize a request with measured timings and speculative stats.
    pub fn finish(
        &mut self,
        prefill_ms: f64,
        decode_ms: f64,
        target_forwards: usize,
        drafted_tokens: usize,
        accepted_drafts: usize,
    ) {
        let mut active = match self.active.take() {
            Some(a) => a,
            None => return,
        };
        active.prefill_ms = prefill_ms;
        active.prefill_tok_s = if prefill_ms > 0.0 {
            active.prompt_tokens as f64 / (prefill_ms / 1_000.0)
        } else {
            0.0
        };
        active.decode_ms = decode_ms;
        // Use the measured decode time for the final speed, but prefer the live
        // gliding speed when decode was very short (avoids a spiky reading).
        active.decode_tok_s = if decode_ms >= 100.0 {
            active.completion_tokens as f64 / (decode_ms / 1_000.0)
        } else {
            active.decode_tok_s
        };
        active.acceptance = if drafted_tokens > 0 {
            accepted_drafts as f64 / drafted_tokens as f64
        } else {
            0.0
        };
        active.draft_ratio = if active.completion_tokens > 0 {
            drafted_tokens as f64 / active.completion_tokens as f64
        } else {
            0.0
        };
        active.target_forwards = target_forwards;
        active.done = true;

        // Reset the decode-window bookkeeping for the next query.
        self.active_decode_start = None;
        self.active_decode_tokens = 0;

        let a = &mut self.aggregates;
        a.total_queries += 1;
        a.total_completion_tokens += active.completion_tokens as u64;
        a.total_prompt_tokens += active.prompt_tokens as u64;
        a.total_drafted_tokens += drafted_tokens as u64;
        a.total_accepted_drafts += accepted_drafts as u64;
        a.total_target_forwards += target_forwards as u64;
        a.prefill_ms += prefill_ms;
        a.decode_ms += decode_ms;

        self.queries.insert(0, active);
        self.queries.truncate(MAX_QUERIES);
    }

    /// Mark a request as failed (records prompt tokens only, count included).
    pub fn fail(&mut self, prompt_tokens: usize) {
        let mut active = match self.active.take() {
            Some(a) => a,
            None => return,
        };
        active.done = true;
        self.aggregates.total_queries += 1;
        self.aggregates.total_prompt_tokens += prompt_tokens as u64;
        self.queries.insert(0, active);
        self.queries.truncate(MAX_QUERIES);
        self.active_decode_start = None;
        self.active_decode_tokens = 0;
        self.active_prefill_start = None;
    }
}