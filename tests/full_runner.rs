use std::path::{Path, PathBuf};

use lisa_rs::model::VOCAB;
use lisa_rs::model::runner::QwenRunner;

fn local_snapshot() -> Option<PathBuf> {
    let root = std::env::home_dir()?
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.is_dir()
                && std::fs::read_dir(path).is_ok_and(|entries| {
                    entries.filter_map(Result::ok).any(|entry| {
                        entry
                            .path()
                            .extension()
                            .is_some_and(|extension| extension == "safetensors")
                    })
                })
        })
}

fn reference(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    assert_eq!(bytes.len(), VOCAB * 4, "full-model truth size");
    Some(
        bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
            .collect(),
    )
}

#[test]
#[ignore = "loads the 16 GB local checkpoint onto Metal"]
fn token_2005_full_model_matches_mlx() {
    let Some(snapshot) = local_snapshot() else {
        eprintln!("skip: Qwen3.8-27B-4bit snapshot not cached");
        return;
    };
    let mut runner = QwenRunner::load(&snapshot, 16).expect("load native runner");
    let logits = runner.forward_token(2005).expect("forward token 2005");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(runner.position(), 1);

    if let Some(expected) = reference(Path::new("/tmp/mlx_full_token_2005.raw")) {
        let mut max_diff = 0.0f32;
        let mut actual_argmax = (0usize, f32::NEG_INFINITY);
        let mut expected_argmax = (0usize, f32::NEG_INFINITY);
        for (index, (&actual, &reference)) in logits.iter().zip(&expected).enumerate() {
            max_diff = max_diff.max((actual - reference).abs());
            if actual > actual_argmax.1 {
                actual_argmax = (index, actual);
            }
            if reference > expected_argmax.1 {
                expected_argmax = (index, reference);
            }
        }
        eprintln!(
            "full token 2005: max diff={max_diff}, argmax native={} mlx={}",
            actual_argmax.0, expected_argmax.0
        );
        assert!(max_diff <= 1.0, "full logits drifted by {max_diff}");
        assert_eq!(actual_argmax.0, expected_argmax.0, "argmax differs");
    } else {
        eprintln!("skip comparison: run scripts/mlx_full_token_truth.py first");
    }

    runner.reset_state();
    assert_eq!(runner.position(), 0);
}
