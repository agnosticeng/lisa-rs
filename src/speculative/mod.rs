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
    mut target_hidden: Vec<f32>,
    mut seed: Option<DraftSeed>,
    max_tokens: usize,
    eos: Option<u32>,
    on_token: &mut dyn FnMut(u32),
    result: &mut GreedyBlock3Result,
) -> Result<(), String> {
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

        let (drafts, round_appended) = if let Some(first_draft) = seed.take() {
            let d0 = first_draft.token;
            let (second_argmax, _) = mtp.forward_position_decode(d0, &first_draft.hidden)?;
            (vec![d0, second_argmax], 1)
        } else {
            let (d0, first_hidden) = mtp.forward_position_decode(bonus, &target_hidden)?;
            let (second_argmax, _) = mtp.forward_position_decode(d0, &first_hidden)?;
            (vec![d0, second_argmax], 2)
        };

        let verified = target.verify_block3_decode([bonus, drafts[0], drafts[1]])?;
        result.target_forwards += 1;
        let target_tokens: Vec<u32> = verified.iter().map(|(argmax, _)| *argmax).collect();
        let acceptance = accept_greedy(&drafts, &target_tokens);
        target.commit_verified_prefix(acceptance.accepted_drafts + 1)?;

        let plan = mtp_reconcile_plan(drafts.len(), round_appended, acceptance.accepted_drafts);
        mtp.trim_state(plan.trim)?;
        let mut next_seed = None;
        for replay in plan.replay {
            let (token, hidden_index) = match replay {
                MtpReplay::Draft(index) => (drafts[index], index),
                MtpReplay::Replacement => (
                    *acceptance.emitted.last().expect("acceptance emits a token"),
                    acceptance.accepted_drafts,
                ),
            };
            let (argmax, hidden) =
                mtp.forward_position_decode(token, &verified[hidden_index].1)?;
            next_seed = Some(DraftSeed {
                token: argmax,
                hidden,
            });
        }
        seed = next_seed;
        target_hidden = verified[acceptance.accepted_drafts].1.clone();
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
        result.drafted_tokens += drafts.len();
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
    if prompt.is_empty() {
        return Err("cannot prefill an empty prompt".into());
    }
    let mut hidden_states = Vec::with_capacity(prompt.len());
    let mut bonus = 0u32;
    if prompt.len() >= 4 {
        // Batched prefill: process the prompt in fused multi-token passes over
        // the GEMM/attention/GDN prefill path (much faster than N sequential
        // decodes for long prompts), then MTP-prefill the shifted tokens.
        for (argmax, hidden) in target.forward_prefill(prompt)? {
            hidden_states.push(hidden);
            bonus = argmax;
        }
    } else {
        for &token in prompt {
            let (argmax, hidden) = target.forward_token_decode(token)?;
            hidden_states.push(hidden);
            bonus = argmax;
        }
    }
    let mut last_mtp: Option<(u32, Vec<f32>)> = None;
    let mtp_positions: Vec<(u32, &[f32])> = hidden_states
        .iter()
        .enumerate()
        .map(|(t, hidden)| {
            let token = if t + 1 < prompt.len() {
                prompt[t + 1]
            } else {
                bonus
            };
            (token, hidden.as_slice())
        })
        .collect();
    // Batched MTP prefill for longer prompts (mirrors the target batched path);
    // falls back to sequential for tiny ones.
    if mtp_positions.len() >= 4 {
        let mtp_outputs = mtp.forward_position_batch(&mtp_positions)?;
        last_mtp = mtp_outputs.into_iter().last();
    } else {
        for (t, hidden) in hidden_states.iter().enumerate() {
            let token = if t + 1 < prompt.len() {
                prompt[t + 1]
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
