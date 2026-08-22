// Multi-token prediction (MTP) speculative decode. P4.
// The Qwen MTP drafter borrows target embed/lm_head and target hidden states,
// but owns a separate KV cache because its attention projections are learned.

use crate::model::runner::{MtpRunner, QwenRunner};

#[derive(Debug, Eq, PartialEq)]
pub struct GreedyAcceptance {
    pub accepted_drafts: usize,
    pub emitted: Vec<u32>,
    pub target_trim: usize,
}

/// Accept the matching draft prefix, then emit the target token at the first
/// mismatch (or the target bonus when every draft matched).
pub fn accept_greedy(drafts: &[u32], target_tokens: &[u32]) -> GreedyAcceptance {
    assert_eq!(target_tokens.len(), drafts.len() + 1);
    let accepted = drafts
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count();
    let mut emitted = drafts[..accepted].to_vec();
    emitted.push(target_tokens[accepted]);
    let verified_rows = target_tokens.len();
    let target_trim = verified_rows - emitted.len();
    GreedyAcceptance {
        accepted_drafts: accepted,
        emitted,
        target_trim,
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum MtpReplay {
    Draft(usize),
    Replacement,
}

#[derive(Debug, Eq, PartialEq)]
pub struct MtpReconcilePlan {
    pub trim: usize,
    pub replay: Vec<MtpReplay>,
}

/// Mirrors Qwen3.5 MTP's `accept_verified_tokens`: retain drafts already
/// appended during this round, replay accepted drafts not in cache, then append
/// the verifier's replacement/bonus to seed the next round.
pub fn mtp_reconcile_plan(
    draft_count: usize,
    round_appended: usize,
    accepted: usize,
) -> MtpReconcilePlan {
    assert!(accepted <= draft_count);
    assert!(round_appended <= draft_count);
    let keep_appended = accepted.min(round_appended);
    let mut replay = (keep_appended..accepted)
        .map(MtpReplay::Draft)
        .collect::<Vec<_>>();
    replay.push(MtpReplay::Replacement);
    MtpReconcilePlan {
        trim: round_appended - keep_appended,
        replay,
    }
}

#[derive(Debug)]
pub struct GreedyBlock3Result {
    pub tokens: Vec<u32>,
    pub rounds: usize,
    pub accepted_drafts: usize,
    pub drafted_tokens: usize,
    pub target_forwards: usize,
}

impl GreedyBlock3Result {
    pub fn acceptance(&self) -> f64 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_drafts as f64 / self.drafted_tokens as f64
        }
    }
}

struct DraftSeed {
    token: u32,
    hidden: Vec<f32>,
}

fn new_result(max_tokens: usize) -> GreedyBlock3Result {
    GreedyBlock3Result {
        tokens: Vec::with_capacity(max_tokens),
        rounds: 0,
        accepted_drafts: 0,
        drafted_tokens: 0,
        target_forwards: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_block3_loop(
    target: &mut QwenRunner,
    mtp: &mut MtpRunner,
    mut bonus: u32,
    target_hidden: Vec<f32>,
    seed: Option<DraftSeed>,
    max_tokens: usize,
    eos: Option<u32>,
    on_token: &mut dyn FnMut(u32),
    result: &mut GreedyBlock3Result,
) -> Result<(), String> {
    // Carry both the target and MTP hidden states on-device so the verify/MTP
    // chain does not round-trip (bf16->f32 readback + f32->bf16 re-upload +
    // host sync) between every draft forward. Upload the initial host states
    // once; from then on drafts are conditioned by device buffer offsets.
    target.upload_hidden_device(0, &target_hidden);
    // `seed` = the MTP drafter's own (draft token, hidden) from prefill.
    let mut seed_dev = seed.map(|s| {
        mtp.upload_normalized_device(&s.hidden);
        (s.token, mtp.normalized_offset())
    });

    while result.tokens.len() < max_tokens {
        let remaining = max_tokens - result.tokens.len();
        if remaining < 3 {
            for _ in 0..remaining {
                let (next, _) = target.forward_token_decode(bonus)?;
                result.target_forwards += 1;
                bonus = next;
                if eos == Some(next) {
                    return Ok(());
                }
                result.tokens.push(next);
                on_token(next);
            }
            break;
        }

        // ---- generate the two drafts, condition state on-device ----
        let (d0, round_appended) = if let Some(first) = seed_dev.take() {
            // first draft token is already known; derive the second draft d1
            // from the drafter's own hidden (draft chaining, cond=None -> own).
            let (d1, _) = mtp.forward_position_decode_device(first.0, None, first.1)?;
            (vec![first.0, d1], 1)
        } else {
            // first round: d0 from the bonus conditioned on the target hidden
            // (row 0 of the just-uploaded/verified target hidden).
            let (d0, d0_off) =
                mtp.forward_position_decode_device(bonus, Some(target.scratch()), target.hidden_offset(0))?;
            let (d1, _) = mtp.forward_position_decode_device(d0, None, d0_off)?;
            (vec![d0, d1], 2)
        };

        // ---- verify the three tokens against the target ----
        let verified = target.verify_block3_decode_device([bonus, d0[0], d0[1]])?;
        result.target_forwards += 1;
        let target_tokens = verified;
        let acceptance = accept_greedy(&d0, &target_tokens);
        let accepted_rows = acceptance.accepted_drafts + 1;
        target.commit_verified_prefix(accepted_rows)?;

        // The accepted prefix is already in the target KV cache (rows 0..accepted
        // verified). The MTP drafter was appended `round_appended` drafts; keep
        // the accepted ones and seed the next round from the replacement.
        let plan = mtp_reconcile_plan(d0.len(), round_appended, acceptance.accepted_drafts);
        mtp.trim_state(plan.trim)?;
        let mut next_seed = None;
        if let Some(replay) = plan.replay.last() {
            let (token, cond_row) = match replay {
                MtpReplay::Draft(index) => (d0[*index], *index),
                MtpReplay::Replacement => (
                    *acceptance.emitted.last().expect("acceptance emits a token"),
                    acceptance.accepted_drafts,
                ),
            };
            let (argmax, hidden_off) = mtp.forward_position_decode_device(
                token,
                Some(target.scratch()),
                target.hidden_offset(cond_row),
            )?;
            next_seed = Some((argmax, hidden_off));
        }
        seed_dev = next_seed;
        bonus = *acceptance.emitted.last().expect("acceptance emits a token");
        for &token in &acceptance.emitted {
            if eos == Some(token) {
                return Ok(());
            }
            result.tokens.push(token);
            on_token(token);
        }
        result.rounds += 1;
        result.accepted_drafts += acceptance.accepted_drafts;
        result.drafted_tokens += d0.len();
    }
    Ok(())
}

/// A drafter seed: `(draft_token, draft_hidden)`.
pub type MtpSeed = (u32, Vec<f32>);

/// Prefill a prompt token-by-token and collect the state needed to continue
/// speculative generation. Mirrors MLX `prefill_from_target_hidden`: the drafter
/// replays the shifted tokens (`prompt[1..]` + the bonus) conditioned on the
/// target's per-position hidden state, then seeds itself from the final draft.
/// Returns `(bonus, target_hidden, mtp_seed)` — `bonus` is the first generated
/// token, `target_hidden` is the target state at the last prompt position, and
/// `mtp_seed` is `(draft_token, draft_hidden)` for the drafter.
pub fn prefill_prompt(
    target: &mut QwenRunner,
    mtp: &mut MtpRunner,
    prompt: &[u32],
) -> Result<(u32, Vec<f32>, MtpSeed), String> {
    prefill_prompt_from(target, mtp, prompt, 0)
}

/// Session-cache variant of `prefill_prompt`. `target`/`mtp` have already been
/// positioned at `start` (restored from a checkpoint taken at a messages-only
/// boundary). Only `prompt[start..]` is prefilled — the messages delta plus the
/// generation-prompt suffix — so a conversation prefix is not re-tokenized/re-run
/// every turn. Produces the same `(bonus, target_hidden, mtp_seed)` as a full
/// `prefill_prompt(prompt)` when `start == 0`, and positions the runners at
/// `prompt.len()`.
pub fn prefill_prompt_from(
    target: &mut QwenRunner,
    mtp: &mut MtpRunner,
    prompt: &[u32],
    start: usize,
) -> Result<(u32, Vec<f32>, MtpSeed), String> {
    prefill_prompt_from_with_progress(target, mtp, prompt, start, |_| {})
}

/// `prefill_prompt_from` with a progress callback: `on_progress(processed)`
/// fires after each fused prefill chunk with the cumulative number of prompt
/// tokens processed so far (target + MTP), used to stream live prefill speed.
pub fn prefill_prompt_from_with_progress(
    target: &mut QwenRunner,
    mtp: &mut MtpRunner,
    prompt: &[u32],
    start: usize,
    mut on_progress: impl FnMut(usize),
) -> Result<(u32, Vec<f32>, MtpSeed), String> {
    if prompt.is_empty() {
        return Err("cannot prefill an empty prompt".into());
    }
    if start > prompt.len() {
        return Err(format!(
            "prefill start {start} exceeds prompt length {}",
            prompt.len()
        ));
    }
    let seg = &prompt[start..];
    let mut hidden_states = Vec::with_capacity(seg.len());
    let mut bonus = 0u32;
    if seg.len() >= 4 {
        // Batched prefill: process the segment in fused multi-token passes over
        // the GEMM/attention/GDN prefill path (much faster than N sequential
        // decodes for long deltas), then MTP-prefill the shifted tokens.
        for (argmax, hidden) in
            target.forward_prefill_with_progress(seg, |done| on_progress(start + done))?
        {
            hidden_states.push(hidden);
            bonus = argmax;
        }
    } else {
        for (t, &token) in seg.iter().enumerate() {
            let (argmax, hidden) = target.forward_token_decode(token)?;
            hidden_states.push(hidden);
            bonus = argmax;
            on_progress(start + t + 1);
        }
    }
    let mut last_mtp: Option<(u32, Vec<f32>)> = None;
    let full = prompt.len();
    let mtp_positions: Vec<(u32, &[f32])> = hidden_states
        .iter()
        .enumerate()
        .map(|(t, hidden)| {
            let global = start + t;
            let token = if global + 1 < full {
                prompt[global + 1]
            } else {
                bonus
            };
            (token, hidden.as_slice())
        })
        .collect();
    // Batched MTP prefill for longer segments (mirrors the target path); falls
    // back to sequential for tiny ones.
    if mtp_positions.len() >= 4 {
        let mtp_outputs = mtp.forward_position_batch_with_progress(&mtp_positions, |done| {
            on_progress(start + done)
        })?;
        last_mtp = mtp_outputs.into_iter().last();
    } else {
        for (t, hidden) in hidden_states.iter().enumerate() {
            let global = start + t;
            let token = if global + 1 < full {
                prompt[global + 1]
            } else {
                bonus
            };
            let output = mtp.forward_position_decode(token, hidden)?;
            last_mtp = Some(output);
        }
    }
    let target_hidden = hidden_states.into_iter().last().expect("non-empty prompt");
    let mtp_seed = last_mtp.expect("non-empty prompt");
    Ok((bonus, target_hidden, mtp_seed))
}

/// Prefill a strict prefix `prompt[start..end]` on target + MTP and leave both
/// positioned at `end`, WITHOUT returning a generation seed. Used to advance to
/// and re-checkpoint a messages-only boundary for the next turn. `end` must be
/// less than `prompt.len()` (this is a prefix segment, never the final
/// generation segment, so every drafter token is a real prompt token).
pub fn prefill_prefix_until(
    target: &mut QwenRunner,
    mtp: &mut MtpRunner,
    prompt: &[u32],
    start: usize,
    end: usize,
) -> Result<(), String> {
    prefill_prefix_until_with_progress(target, mtp, prompt, start, end, |_| {})
}

/// `prefill_prefix_until` with a progress callback firing after each fused
/// chunk with the cumulative processed token count.
pub fn prefill_prefix_until_with_progress(
    target: &mut QwenRunner,
    mtp: &mut MtpRunner,
    prompt: &[u32],
    start: usize,
    end: usize,
    mut on_progress: impl FnMut(usize),
) -> Result<(), String> {
    if start > end || end >= prompt.len() {
        return Err(format!(
            "prefill_prefix_until: invalid range [{start}, {end}] for prompt len {}",
            prompt.len()
        ));
    }
    if start == end {
        return Ok(());
    }
    let seg = &prompt[start..end];
    let mut hidden_states = Vec::with_capacity(seg.len());
    if seg.len() >= 4 {
        for (_, hidden) in target.forward_prefill_with_progress(seg, |done| {
            on_progress(start + done)
        })? {
            hidden_states.push(hidden);
        }
    } else {
        for (t, &token) in seg.iter().enumerate() {
            let (_, hidden) = target.forward_token_decode(token)?;
            hidden_states.push(hidden);
            on_progress(start + t + 1);
        }
    }
    // MTP positions for this prefix: hidden at target position start+t conditions
    // the drafter predicting prompt[start+t+1] (guaranteed < prompt.len()).
    let mtp_positions: Vec<(u32, &[f32])> = hidden_states
        .iter()
        .enumerate()
        .map(|(t, hidden)| (prompt[start + t + 1], hidden.as_slice()))
        .collect();
    if mtp_positions.len() >= 4 {
        let _ = mtp.forward_position_batch_with_progress(&mtp_positions, |done| {
            on_progress(start + done)
        })?;
    } else {
        for (token, hidden) in mtp_positions {
            let _ = mtp.forward_position_decode(token, hidden)?;
        }
    }
    Ok(())
}

/// Speculative generation after `prefill_prompt`. `bonus` is the first generated
/// token (already produced during prefill); the loop emits the remaining tokens
/// up to `max_tokens` (including `bonus`), so `max_tokens - 1` tokens are decoded here.
pub fn generate_greedy_block3_prefilled(
    target: &mut QwenRunner,
    mtp: &mut MtpRunner,
    bonus: u32,
    target_hidden: Vec<f32>,
    mtp_seed: (u32, Vec<f32>),
    max_tokens: usize,
) -> Result<GreedyBlock3Result, String> {
    let mut result = new_result(max_tokens);
    if max_tokens == 0 {
        return Ok(result);
    }
    result.tokens.push(bonus);
    let seed = Some(DraftSeed {
        token: mtp_seed.0,
        hidden: mtp_seed.1,
    });
    run_block3_loop(
        target,
        mtp,
        bonus,
        target_hidden,
        seed,
        max_tokens,
        None,
        &mut |_| {},
        &mut result,
    )?;
    Ok(result)
}

/// Streaming variant of `generate_greedy_block3_prefilled`: identical token
/// stream, but every generated token (including `bonus`) is passed to
/// `on_token` as it is produced, and generation stops early as soon as `eos` is
/// produced (the stop token is not emitted). Useful for the serving layer,
/// which yields text deltas without buffering the whole completion.
#[allow(clippy::too_many_arguments)]
pub fn generate_greedy_block3_prefilled_streaming(
    target: &mut QwenRunner,
    mtp: &mut MtpRunner,
    bonus: u32,
    target_hidden: Vec<f32>,
    mtp_seed: (u32, Vec<f32>),
    max_tokens: usize,
    eos: Option<u32>,
    on_token: &mut dyn FnMut(u32),
) -> Result<GreedyBlock3Result, String> {
    let mut result = new_result(max_tokens);
    if max_tokens == 0 {
        return Ok(result);
    }
    result.tokens.push(bonus);
    if eos == Some(bonus) {
        return Ok(result);
    }
    on_token(bonus);
    let seed = Some(DraftSeed {
        token: mtp_seed.0,
        hidden: mtp_seed.1,
    });
    run_block3_loop(
        target,
        mtp,
        bonus,
        target_hidden,
        seed,
        max_tokens,
        eos,
        on_token,
        &mut result,
    )?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_first_draft() {
        assert_eq!(
            accept_greedy(&[10, 11], &[20, 21, 22]),
            GreedyAcceptance {
                accepted_drafts: 0,
                emitted: vec![20],
                target_trim: 2,
            }
        );
    }

    #[test]
    fn accepts_one_draft() {
        assert_eq!(
            accept_greedy(&[10, 11], &[10, 21, 22]),
            GreedyAcceptance {
                accepted_drafts: 1,
                emitted: vec![10, 21],
                target_trim: 1,
            }
        );
    }

    #[test]
    fn accepts_all_drafts_and_bonus() {
        assert_eq!(
            accept_greedy(&[10, 11], &[10, 11, 22]),
            GreedyAcceptance {
                accepted_drafts: 2,
                emitted: vec![10, 11, 22],
                target_trim: 0,
            }
        );
    }

    #[test]
    fn reconciles_first_round_transitions() {
        assert_eq!(
            mtp_reconcile_plan(2, 2, 0),
            MtpReconcilePlan {
                trim: 2,
                replay: vec![MtpReplay::Replacement],
            }
        );
        assert_eq!(
            mtp_reconcile_plan(2, 2, 1),
            MtpReconcilePlan {
                trim: 1,
                replay: vec![MtpReplay::Replacement],
            }
        );
        assert_eq!(
            mtp_reconcile_plan(2, 2, 2),
            MtpReconcilePlan {
                trim: 0,
                replay: vec![MtpReplay::Replacement],
            }
        );
    }

    #[test]
    fn reconciles_seeded_round_transitions() {
        assert_eq!(
            mtp_reconcile_plan(2, 1, 0),
            MtpReconcilePlan {
                trim: 1,
                replay: vec![MtpReplay::Replacement],
            }
        );
        assert_eq!(
            mtp_reconcile_plan(2, 1, 1),
            MtpReconcilePlan {
                trim: 0,
                replay: vec![MtpReplay::Replacement],
            }
        );
        assert_eq!(
            mtp_reconcile_plan(2, 1, 2),
            MtpReconcilePlan {
                trim: 0,
                replay: vec![MtpReplay::Draft(1), MtpReplay::Replacement],
            }
        );
    }
}
