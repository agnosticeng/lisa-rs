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
    let args: Vec<String> = std::env::args().collect();
    let subcommand = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("serve");
    match subcommand {
        "serve" => {
            if args.get(2).is_some_and(|a| a == "--help" || a == "-h") {
                println!("{}", serve::USAGE);
                return ExitCode::SUCCESS;
            }
            serve::main()
        }
        "run" => {
            if args.get(2).is_some_and(|a| a == "--help" || a == "-h") {
                println!("{}", run::USAGE);
                return ExitCode::SUCCESS;
            }
            run::main()
        }
        "help" | "-h" | "--help" => {
            println!("usage: cli <serve|run> [options]\n\n{}", serve::USAGE);
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("cli: unknown subcommand {other:?} (expected 'serve' or 'run')");
            ExitCode::FAILURE
        }
    }
}