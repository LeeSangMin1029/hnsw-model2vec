// Helper macros available to all test submodules
macro_rules! find_chunk {
    ($chunks:expr, $name:expr) => {
        $chunks
            .iter()
            .find(|c| c.name == $name)
            .unwrap_or_else(|| {
                let names: Vec<&str> = $chunks.iter().map(|c| c.name.as_str()).collect();
                panic!(
                    "expected chunk named {:?}, found: {:?}",
                    $name, names
                );
            })
    };
}

macro_rules! has_chunk {
    ($chunks:expr, $name:expr) => {
        $chunks.iter().any(|c| c.name == $name)
    };
}

mod fixtures;
mod rust;
mod typescript;
mod python;
mod go_lang;
mod java;
mod c_lang;
mod cpp;
mod common;
mod extract;
mod hash;
