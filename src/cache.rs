// Hugging Face cache resolution: turn a model id (`mlx-community/Qwen3.8-27B-4bit`)
// or an explicit snapshot path into a local snapshot directory.
use std::path::{Path, PathBuf};

/// Resolve a model argument to a local snapshot directory.
///
/// Accepts either a filesystem path to a snapshot directory, or a Hugging Face
/// model id (`org/name`) which is looked up in the HF cache
/// (`~/.cache/huggingface/hub`, or `$HF_HOME/hub` when set). The snapshot is
/// resolved via `refs/main` when present, falling back to any snapshot
/// directory that contains `.safetensors` files.
pub fn resolve_snapshot(model: &str) -> Result<PathBuf, String> {
    let as_path = Path::new(model);
    if as_path.is_dir() {
        return Ok(as_path.to_path_buf());
    }

    let cache_root = hf_cache_root()?;
    let cache_dir = format!("models--{}", model.replace('/', "--"));
    let snapshots = cache_root.join(&cache_dir).join("snapshots");
    if !snapshots.is_dir() {
        return Err(format!(
            "model {model:?} is neither an existing directory nor cached under {}",
            cache_root.display()
        ));
    }

    // Prefer the snapshot referenced by refs/main (the currently-resolved hash).
    let refs_main = cache_root.join(&cache_dir).join("refs").join("main");
    if let Ok(hash) = std::fs::read_to_string(&refs_main) {
        let candidate = snapshots.join(hash.trim());
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }

    let mut found = None;
    if let Ok(entries) = std::fs::read_dir(&snapshots) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() && has_safetensors(&path) {
                found = Some(path);
                break;
            }
        }
    }
    found.ok_or_else(|| {
        format!(
            "no cached snapshot found for {model:?} under {}",
            snapshots.display()
        )
    })
}

fn hf_cache_root() -> Result<PathBuf, String> {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(hf_home).join("hub"));
    }
    let home = std::env::var("HOME").map_err(|_| "HOME is not set".to_string())?;
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub"))
}

fn has_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_cached_model_id() {
        let model = "mlx-community/Qwen3.8-27B-4bit";
        match resolve_snapshot(model) {
            Ok(path) => {
                assert!(path.is_dir());
                assert!(path.join("tokenizer.json").exists());
            }
            Err(e) => eprintln!("skip: {e}"),
        }
    }

    #[test]
    fn resolves_explicit_directory() {
        let dir = std::env::temp_dir();
        assert_eq!(resolve_snapshot(dir.to_str().unwrap()).unwrap(), dir);
    }
}
