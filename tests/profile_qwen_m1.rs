// Synchronized per-family diagnostic profiler for the M=1 target forward.
// Renders the per-family wall-clock breakdown and checks argmax consistency
// against a normal forward. Needs real cached weights.
use lisa_rs::model::LayerKind;
use lisa_rs::model::runner::QwenRunner;
use serde_json::{Value, json};

#[test]
#[ignore = "loads real weights + profiles; diagnostic-only"]
fn profile_qwen_m1_diagnostic() -> Result<(), String> {
    let Ok(snapshot) = lisa_rs::cache::resolve_snapshot("mlx-community/Qwen3.8-27B-4bit") else {
        eprintln!("target model not in HF cache; skipping");
        return Ok(());
    };
    let token = 2005u32;

    let mut runner = QwenRunner::load(&snapshot, 1)?;
    let profile = runner.profile_token_m1_diagnostic(token)?;
    let profiled_position = runner.position() - 1;
    runner.reset_state();
    let normal = runner.forward_token_with_hidden(token)?;
    if profile.output.argmax != normal.argmax {
        return Err(format!(
            "diagnostic argmax {} differs from normal M=1 argmax {}",
            profile.output.argmax, normal.argmax
        ));
    }
    let logits_bit_exact = profile.output.logits == normal.logits;
    let hidden_bit_exact = profile.output.hidden == normal.hidden;
    let max_abs_diff = |left: &[f32], right: &[f32]| {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    };
    let logits_max_abs_diff = max_abs_diff(&profile.output.logits, &normal.logits);
    let hidden_max_abs_diff = max_abs_diff(&profile.output.hidden, &normal.hidden);
    let milliseconds = |d: std::time::Duration| d.as_secs_f64() * 1_000.0;
    let embedding_ms = milliseconds(profile.embedding_wall);
    let gdn_ms = milliseconds(profile.gdn_wall);
    let attention_ms = milliseconds(profile.attention_wall);
    let final_ms = milliseconds(profile.final_norm_lm_head_wall);
    let cpu_ms = milliseconds(profile.cpu_readback_argmax_wall);
    let accounted_ms = embedding_ms + gdn_ms + attention_ms + final_ms + cpu_ms;
    let proportion = |milliseconds: f64| milliseconds / accounted_ms;
    let layers: Vec<Value> = profile
        .layers
        .iter()
        .map(|layer| {
            json!({
                "index": layer.layer,
                "family": match layer.kind {
                    LayerKind::Gdn => "gdn",
                    LayerKind::Attention => "attention",
                },
                "wall_ms": milliseconds(layer.wall),
            })
        })
        .collect();

    let report = json!({
        "mode": "diagnostic_synchronized_m1",
        "token": token,
        "position": profiled_position,
        "argmax": profile.output.argmax,
        "normal_forward_comparison": {
            "argmax_match": true,
            "logits_bit_exact": logits_bit_exact,
            "logits_max_abs_diff": logits_max_abs_diff,
            "hidden_bit_exact": hidden_bit_exact,
            "hidden_max_abs_diff": hidden_max_abs_diff,
        },
        "sections": {
            "embedding": { "wall_ms": embedding_ms, "proportion": proportion(embedding_ms) },
            "gdn_48_total": { "wall_ms": gdn_ms, "proportion": proportion(gdn_ms) },
            "attention_16_total": { "wall_ms": attention_ms, "proportion": proportion(attention_ms) },
            "final_norm_lm_head": { "wall_ms": final_ms, "proportion": proportion(final_ms) },
            "cpu_readback_argmax": { "wall_ms": cpu_ms, "proportion": proportion(cpu_ms) },
        },
        "layers": layers,
        "accounted_wall_ms": accounted_ms,
        "total_wall_ms": milliseconds(profile.total_wall),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| format!("serialize profile JSON: {e}"))?
    );
    Ok(())
}