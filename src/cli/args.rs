//! Minimal named-argument parser shared by the `serve` and `run` subcommands.

/// A parsed option, mapping both `--name value` and `--name=value` forms.
pub fn value(args: &[String], name: &str) -> Result<Option<String>, String> {
    let flag = format!("--{name}");
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == &flag {
            let value = iter
                .next()
                .ok_or_else(|| format!("--{name} requires a value"))?;
            return Ok(Some(value.clone()));
        }
        if let Some(value) = arg.strip_prefix(&format!("--{name}=")) {
            return Ok(Some(value.to_string()));
        }
    }
    Ok(None)
}

/// A boolean flag (`--name`) — present or absent, never consumes a value.
pub fn has_flag(args: &[String], name: &str) -> bool {
    let flag = format!("--{name}");
    let flag_eq = format!("--{name}=");
    args.iter()
        .any(|arg| arg == &flag || arg.starts_with(&flag_eq))
}

/// Boolean flag with an optional explicit value (`--no-ui` or `--no-ui=true`).
/// Returns `None` when absent, `Some(true/false)` when present.
pub fn flag(args: &[String], name: &str) -> Option<bool> {
    let flag = format!("--{name}");
    let flag_eq = format!("--{name}=");
    for arg in args {
        if arg == &flag {
            return Some(true);
        }
        if let Some(value) = arg.strip_prefix(&flag_eq) {
            return Some(value.parse::<bool>().unwrap_or(true));
        }
    }
    None
}

/// Positional (non-flag) arguments, in order.
pub fn positional(args: &[String]) -> Vec<String> {
    args.iter()
        .filter(|arg| !arg.starts_with('-'))
        .cloned()
        .collect()
}

/// First positional argument matching an alias, or the positional at `index`.
pub fn request(
    args: &[String],
    name: &str,
    positionals: &[String],
    index: usize,
) -> Result<Option<String>, String> {
    if let Some(value) = value(args, name)? {
        return Ok(Some(value));
    }
    Ok(positionals.get(index).cloned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(input: &[&str]) -> Vec<String> {
        input.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn flag_at_end_does_not_consume_positionals() {
        let args = s(&[
            "mlx-community/Qwen3.8-27B-4bit",
            "mlx-community/Qwen3.8-27B-MTP-4bit",
            "--port",
            "8000",
            "--no-ui",
        ]);
        assert!(has_flag(&args, "no-ui"));
        let positionals = positional(&args);
        assert_eq!(
            positionals,
            s(&[
                "mlx-community/Qwen3.8-27B-4bit",
                "mlx-community/Qwen3.8-27B-MTP-4bit",
                "8000",
            ])
        );
    }

    #[test]
    fn flag_at_start_keeps_following_args() {
        let args = s(&["--no-ui", "A", "B"]);
        assert!(has_flag(&args, "no-ui"));
        assert_eq!(positional(&args), s(&["A", "B"]));
    }

    #[test]
    fn bool_flag_value_forms() {
        assert_eq!(flag(&s(&["--no-ui"]), "no-ui"), Some(true));
        assert_eq!(flag(&s(&["--no-ui=true"]), "no-ui"), Some(true));
        assert_eq!(flag(&s(&["--no-ui=false"]), "no-ui"), Some(false));
        assert_eq!(flag(&s(&[]), "no-ui"), None);
    }

    #[test]
    fn value_both_forms() {
        assert_eq!(
            value(&s(&["--port", "8123"]), "port").unwrap(),
            Some("8123".into())
        );
        assert_eq!(
            value(&s(&["--port=8123"]), "port").unwrap(),
            Some("8123".into())
        );
        assert!(value(&s(&["--port"]), "port").is_err());
    }

    #[test]
    fn request_prefers_named_over_positional() {
        let args = s(&["--target", "T", "--mtp", "M"]);
        let positionals = positional(&args);
        assert_eq!(
            request(&args, "target", &positionals, 0).unwrap(),
            Some("T".into())
        );
    }
}