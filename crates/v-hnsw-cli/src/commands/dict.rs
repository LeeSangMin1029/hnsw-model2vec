//! Korean dictionary provisioning.
//!
//! Downloads and compiles the ko-dic MeCab dictionary for Lindera on first use.
//! The compiled dictionary is cached at `~/.v-hnsw/dict/ko-dic/`.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use tracing::info;

const KO_DIC_URL: &str = "https://lindera.dev/mecab-ko-dic-2.1.1-20180720.tar.gz";
const KO_DIC_MD5: &str = "b996764e91c96bc89dc32ea208514a96";
const KO_DIC_EXTRACT_DIR: &str = "mecab-ko-dic-2.1.1-20180720";

const KO_DIC_METADATA_JSON: &str = r#"{
  "name": "ko-dic",
  "encoding": "UTF-8",
  "compress_algorithm": "deflate",
  "default_word_cost": -10000,
  "default_left_context_id": 0,
  "default_right_context_id": 0,
  "default_field_value": "*",
  "flexible_csv": false,
  "skip_invalid_cost_or_id": false,
  "normalize_details": false,
  "dictionary_schema": {
    "fields": [
      "surface", "left_context_id", "right_context_id", "cost",
      "part_of_speech_tag", "meaning", "presence_absence", "reading",
      "type", "first_part_of_speech", "last_part_of_speech", "expression"
    ]
  },
  "user_dictionary_schema": {
    "fields": ["surface", "part_of_speech_tag", "reading"]
  }
}"#;

/// Ensure the ko-dic dictionary is available, downloading and building if needed.
///
/// Returns the path to the compiled dictionary directory.
pub fn ensure_ko_dic() -> Result<PathBuf> {
    let dict_dir = v_hnsw_core::ko_dic_dir();

    // Check if already built (dict.da is the compiled dictionary trie)
    if dict_dir.join("dict.da").exists() {
        return Ok(dict_dir);
    }

    info!("Korean dictionary not found, downloading and building...");
    eprintln!("Downloading Korean dictionary (first-time setup)...");

    download_and_build(&dict_dir)?;

    info!("Korean dictionary ready at {}", dict_dir.display());
    Ok(dict_dir)
}


fn download_and_build(output_dir: &Path) -> Result<()> {
    let temp_dir = output_dir.parent().unwrap_or(Path::new(".")).join("_tmp");
    std::fs::create_dir_all(&temp_dir).context("create temp dir")?;

    // Download
    let tar_gz_path = temp_dir.join("mecab-ko-dic.tar.gz");
    download_with_checksum(KO_DIC_URL, &tar_gz_path, KO_DIC_MD5)?;

    // Extract
    let extract_dir = temp_dir.join(KO_DIC_EXTRACT_DIR);
    extract_tar_gz(&tar_gz_path, &temp_dir)?;

    if !extract_dir.exists() {
        bail!(
            "expected directory {} not found after extraction",
            extract_dir.display()
        );
    }

    // Build compiled dictionary
    std::fs::create_dir_all(output_dir).context("create dict output dir")?;
    build_dictionary(&extract_dir, output_dir)?;

    // Cleanup temp files
    let _ = std::fs::remove_dir_all(&temp_dir);

    Ok(())
}

fn download_with_checksum(url: &str, dest: &Path, expected_md5: &str) -> Result<()> {
    let resp = ureq::get(url).call().context("download ko-dic tar.gz")?;

    let mut body = Vec::new();
    resp.into_reader()
        .read_to_end(&mut body)
        .context("read response body")?;

    // Verify MD5
    let actual_md5 = format!("{:x}", md5::compute(&body));
    if actual_md5 != expected_md5 {
        bail!(
            "MD5 mismatch: expected {}, got {} (download may be corrupted)",
            expected_md5,
            actual_md5
        );
    }

    std::fs::write(dest, &body).context("write tar.gz")?;
    Ok(())
}

fn extract_tar_gz(tar_gz_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = std::fs::File::open(tar_gz_path).context("open tar.gz")?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);
    archive.unpack(dest_dir).context("extract tar.gz")?;
    Ok(())
}

fn build_dictionary(input_dir: &Path, output_dir: &Path) -> Result<()> {
    use lindera::dictionary::{DictionaryBuilder, Metadata};

    let metadata: Metadata =
        serde_json::from_str(KO_DIC_METADATA_JSON).context("parse ko-dic metadata")?;

    let builder = DictionaryBuilder::new(metadata);
    builder
        .build_dictionary(input_dir, output_dir)
        .map_err(|e| anyhow::anyhow!("failed to build ko-dic dictionary: {}", e))?;

    Ok(())
}
