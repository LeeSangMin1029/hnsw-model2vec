//! v-daemon — Unified daemon for document search (v-hnsw) and code intelligence (v-code).
//!
//! Runs a single persistent daemon that handles both document and code requests,
//! keeping the embedding model and LSP server (rust-analyzer) resident in memory.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(about = "Unified daemon: document search + code intelligence")]
struct Cli {
    /// Database path to preload
    #[arg(long)]
    db: Option<PathBuf>,

    /// TCP port to listen on
    #[arg(long, default_value_t = 19530)]
    port: u16,

    /// Idle timeout in seconds (0 = persistent)
    #[arg(long, default_value_t = 600)]
    timeout: u64,

    /// Run in background
    #[arg(long)]
    background: bool,
}

fn main() {
    #[allow(clippy::expect_used)]
    let directive = "v_hnsw=info".parse().expect("static directive");
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(directive),
        )
        .init();

    v_daemon::interrupt::install_handler();

    let cli = Cli::parse();

    if let Err(err) = v_daemon::server::run(cli.db, cli.port, cli.timeout, cli.background) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
