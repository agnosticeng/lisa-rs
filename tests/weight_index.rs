use std::fs::read_dir;
use std::path::{Path, PathBuf};

use lisa_rs::format::dtype::Dtype;
use lisa_rs::model::weights::WeightIndex;

fn collect_shards(dir: &Path, paths: &mut Vec<PathBuf>) {
    for entry in read_dir(dir).expect("read cache") {
        let path = entry.expect("cache entry").path();
        if path.is_dir() {
            collect_shards(&path, paths);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            paths.push(path);
        }
    }
}

#[test]
fn indexes_endpoint_weights_without_copying_shards() {
    let root = std::env::home_dir()
        .expect("home")
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let mut paths = Vec::new();
    collect_shards(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("skip: model not cached");
        return;
    }
    let index = WeightIndex::open(&paths);
    assert!(!index.is_empty());

    let embed = index
        .slot("language_model.model.embed_tokens.weight")
        .expect("embed weight");
    assert_eq!(embed.dtype, Dtype::U32);
    assert_eq!(embed.shape, [248320, 640]);
    assert_eq!(
        index
            .rows("language_model.model.embed_tokens.weight", 2005, 1)
            .expect("embed row")
            .len(),
        5120 / 8 * 4
    );

    let norm = index
        .slot("language_model.model.norm.weight")
        .expect("final norm");
    assert_eq!(norm.dtype, Dtype::BF16);
    assert_eq!(norm.shape, [5120]);

    let head = index
        .slot("language_model.lm_head.weight")
        .expect("lm head");
    assert_eq!(head.dtype, Dtype::U32);
    assert_eq!(head.shape, [248320, 640]);
    assert!(index.len() > 1000);
}
