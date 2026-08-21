// Probe the real HF Qwen3.8-27B-4bit shards: enumerate the tensor table and
// confirm the affine-q4 linear layout (weight U[out,in/8], scales/biases
// bf16[out,in/64]) matches what the NAX kernel expects. mmap-only, no copy.
use std::fs::read_dir;
use std::path::{Path, PathBuf};

use lisa_rs::format::safetensor as st;

#[test]
#[ignore = "reads real cached weights; diagnostic-only"]
fn probe_layer3_projection_layout() {
    let Ok(model_dir) = lisa_rs::cache::resolve_snapshot("mlx-community/Qwen3.8-27B-4bit") else {
        eprintln!("target model not in HF cache; skipping");
        return;
    };
    let mut paths = Vec::new();
    collect_shards(&model_dir, &mut paths);
    assert!(!paths.is_empty(), "no .safetensors under {model_dir:?}");
    eprintln!("found {} shard file(s)", paths.len());

    let mut found = 0usize;
    for (idx, p) in paths.iter().enumerate() {
        let s = st::open(p);
        eprintln!(
            "shard {}: {} tensors, {} MiB",
            idx,
            s.tensors.len(),
            s.data_len / 1048576
        );
        for te in &s.tensors {
            let n = te.name.as_str();
            if n.starts_with("language_model.model.layers.3.") && n.ends_with("_proj.weight") {
                found += 1;
                let out = te.shape[0];
                let cols = te.shape[1];
                assert_eq!(
                    cols % 64,
                    0,
                    "{} cols must divide 64 for gs64 affine (got {cols})",
                    te.name
                );
                eprintln!(
                    "  # {n} shape=[{out},{cols}] -> in={} groups={}",
                    cols * 8,
                    cols / 8
                );
            }
        }
    }
    assert!(
        found >= 6,
        "expected at least the layer-3 MLP/attention projections, found {found}"
    );
}

fn collect_shards(dir: &Path, out: &mut Vec<PathBuf>) {
    let it = read_dir(dir).expect("read_directory");
    for entry in it {
        let p = entry.expect("dir_entry").path();
        if p.is_dir() {
            collect_shards(&p, out);
        } else if p
            .file_name()
            .expect("fn")
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            out.push(PathBuf::from(&p));
        }
    }
}