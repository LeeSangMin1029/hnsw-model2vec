//! v-hnsw — Document-oriented vector database CLI.

use v_hnsw_cli::error::CliError;

fn main() {
    #[allow(clippy::expect_used)]
    let directive = "v_hnsw=info".parse().expect("static directive");
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(directive),
        )
        .init();

    v_hnsw_cli::interrupt::install_handler();

    if let Err(err) = v_hnsw_cli::run_doc() {
        let cli_err = CliError::from(err);
        tracing::error!("{cli_err:#}");
        log_to_file(&cli_err);
        eprintln!("Error: {cli_err}");
        std::process::exit(1);
    }
}

/// Best-effort append error to `~/.v-hnsw/logs/v-hnsw.log`.
fn log_to_file(err: &CliError) {
    use std::io::Write;

    let log_dir = v_hnsw_core::data_dir().join("logs");
    let _ = std::fs::create_dir_all(&log_dir);

    let log_path = log_dir.join("v-hnsw.log");

    if let Ok(meta) = std::fs::metadata(&log_path)
        && meta.len() > 1_000_000
    {
        let _ = std::fs::remove_file(&log_path);
    }

    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    else {
        return;
    };

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let _ = writeln!(f, "[{ts}] {err:?}");
}
