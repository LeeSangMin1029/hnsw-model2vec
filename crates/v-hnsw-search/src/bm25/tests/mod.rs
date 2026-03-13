mod bigram;
mod fieldnorm;
mod fst_storage;
mod index;
mod maxscore;
mod scorer;
mod snapshot;

fn make_temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::create_dir_all(&dir);
    dir
}
