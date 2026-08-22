//! Single CLI binary for lisa-rs.
//!
//! Subcommands:
//!   cli serve <TARGET> <MTP> [options]  run the OpenAI-compatible HTTP server
//!   cli run   <TARGET> <MTP> [options]  run the native target vs MTP benchmark
//!
//! <TARGET>/<MTP> are snapshot directories or Hugging Face ids.
use std::process::ExitCode;

use lisa_rs::cli::{run, serve};

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().collect();
    let prog = args
        .first()
        .and_then(|a| std::path::Path::new(a).file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "cli".to_string());
    args.remove(0);
    let subcommand = args.first().map(String::as_str).unwrap_or("serve");
    match subcommand {
        "serve" => {
            if args.get(1).is_some_and(|a| a == "--help" || a == "-h") {
                println!("{}", serve::USAGE);
                return ExitCode::SUCCESS;
            }
            serve::main()
        }
        "run" => {
            if args.get(1).is_some_and(|a| a == "--help" || a == "-h") {
                println!("{}", run::USAGE);
                return ExitCode::SUCCESS;
            }
            run::main()
        }
        "help" | "-h" | "--help" => {
            println!("usage: {prog} <serve|run> [options]\n\n{}", serve::USAGE);
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("{prog}: unknown subcommand {other:?} (expected 'serve' or 'run')");
            ExitCode::FAILURE
        }
    }
}