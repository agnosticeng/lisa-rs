//! `serve` subcommand: run the OpenAI-compatible HTTP server with a live
//! mactop-style terminal dashboard.
use std::process::ExitCode;
use std::sync::{Arc, atomic::AtomicBool, atomic::Ordering};
use std::thread;

use crate::cli::args as cli_args;
use crate::serve::{Model, build_router};

pub const USAGE: &str = "\
serve <TARGET> <MTP> [options]
   or: serve --target <TARGET> --mtp <MTP> [options]

Run the OpenAI-compatible HTTP server. <TARGET>/<MTP> are snapshot
directories or Hugging Face model ids resolved from the local HF cache.

options:
  --port PORT          listen port (default 8000)
  --capacity N         KV/state capacity (default 32768)
  --model-id ID        model id advertised by GET /v1/models (default qwen3.8-27b)
  --no-ui              run without the terminal dashboard
  --help               print this help";

pub struct ServeConfig {
    pub target: String,
    pub mtp: String,
    pub port: u16,
    pub capacity: usize,
    pub model_id: String,
    pub ui: bool,
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(2).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("serve: {e}");
            ExitCode::FAILURE
        }
    }
}

/// True if stdout/stdin are interactive terminals (best guess for "show the UI").
fn wants_ui(ui_flag: bool) -> bool {
    if !ui_flag {
        return false;
    }
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

pub fn run(args: &[String]) -> Result<(), String> {
    let config = parse_args(args)?;

    let target_snapshot = crate::cache::resolve_snapshot(&config.target)?;
    let mtp_snapshot = crate::cache::resolve_snapshot(&config.mtp)?;
    let tokenizer_path = target_snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(format!(
            "tokenizer.json not found under {}",
            target_snapshot.display()
        ));
    }

    // Shared telemetry: the Model records into it; the UI reads it.
    let ui_metrics = crate::serve::metrics::Metrics::shared();

    let model = Model::load_with_metrics(
        &target_snapshot,
        &mtp_snapshot,
        &tokenizer_path,
        config.capacity,
        config.model_id.clone(),
        ui_metrics.clone(),
    )?;
    let router = build_router(model);

    if !wants_ui(config.ui) {
        // No interactive terminal: run the server on the calling thread only.
        let shutdown = Arc::new(AtomicBool::new(false));
        return serve_foreground(config.port, config.model_id, router, &shutdown, false);
    }

    // Interactive: run axum on a background thread, the TUI on the main thread.
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = shutdown.clone();
    let server_model = config.model_id.clone();
    let banner = config.port;
    let server_thread = thread::spawn(move || {
        serve_foreground(banner, server_model, router, &server_shutdown, true)
            .map_err(|e| (banner.to_string(), e))
    });

    // The dashboard takes over the terminal; `metrics` is shared with the Model.
    let dashboard = crate::serve::ui::run_dashboard(&ui_metrics, config.model_id, Some(config.mtp));
    // Whatever the TUI exit status, signal the server to stop, then report it.
    if let Err(e) = dashboard {
        eprintln!("dashboard error: {e}");
    }
    shutdown.store(true, Ordering::Relaxed);

    match server_thread.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err((addr, e))) => Err(format!("server on {addr} failed: {e}")),
        Err(_) => Err("server thread panicked".to_string()),
    }
}

/// Bind the port and run the axum server, honoring `shutdown` for graceful stop.
fn serve_foreground(
    port: u16,
    model_id: String,
    router: axum::Router,
    shutdown: &Arc<AtomicBool>,
    quiet: bool,
) -> Result<(), String> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

    let stop_flag = shutdown.clone();
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
            .await
            .map_err(|e| format!("failed to bind 0.0.0.0:{port}: {e}"))?;
        if !quiet {
            println!("lisa-rs server listening on http://0.0.0.0:{port} (model={model_id})");
        }
        let server = axum::serve(listener, router);
        // Poll the shared flag every 250ms; stop when it flips.
        let flag = stop_flag.clone();
        let stop = async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }
        };
        server
            .with_graceful_shutdown(stop)
            .await
            .map_err(|e| format!("server error: {e}"))
    })
}

fn parse_args(args: &[String]) -> Result<ServeConfig, String> {
    let positionals = cli_args::positional(args);
    let target = cli_args::request(args, "target", &positionals, 0)?
        .ok_or("missing TARGET argument (path or Hugging Face id)")?;
    let mtp = cli_args::request(args, "mtp", &positionals, 1)?
        .ok_or("missing MTP argument (path or Hugging Face id)")?;
    let port = cli_args::value(args, "port")?
        .map_or(Ok(8000u16), |v| {
            v.parse().map_err(|e| format!("invalid --port: {e}"))
        })?;
    let capacity = cli_args::value(args, "capacity")?
        .map_or(Ok(32768usize), |v| {
            v.parse().map_err(|e| format!("invalid --capacity: {e}"))
        })?;
    let model_id = cli_args::value(args, "model-id")?.unwrap_or_else(|| "qwen3.8-27b".into());
    let ui = !cli_args::has_flag(args, "no-ui");
    Ok(ServeConfig {
        target,
        mtp,
        port,
        capacity,
        model_id,
        ui,
    })
}