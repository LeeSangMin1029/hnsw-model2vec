//! Auto-generate a minimal stub workspace for lightweight rust-analyzer analysis.
//!
//! Strategy:
//! 1. `cargo metadata` → full dependency graph (workspace + external versions)
//! 2. Target crate: junction/symlink src/ to real source
//! 3. Workspace crates (non-target): extract pub signatures via tree-sitter
//! 4. External crates: auto-extract pub signatures from cargo registry source
//! 5. Generate workspace Cargo.toml listing all members

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

/// Metadata for a single crate in the dependency graph.
#[derive(Debug)]
struct CrateMeta {
    name: String,
    _version: String,
    is_workspace: bool,
    /// path dependencies (workspace crates)
    workspace_deps: Vec<String>,
    /// external dependencies (from crates.io)
    external_deps: Vec<ExternalDep>,
    /// source directory (for workspace crates)
    manifest_path: PathBuf,
    /// features used by dependents
    _features: Vec<String>,
}

#[derive(Debug, Clone)]
struct ExternalDep {
    name: String,
    features: Vec<String>,
    /// Source directory from cargo registry (for auto-stub generation).
    source_dir: Option<PathBuf>,
}

/// Result of stub workspace generation.
pub struct StubWorkspace {
    /// Root directory of the generated workspace.
    pub root: PathBuf,
    /// The target crate that has real source (via symlink).
    pub target_crate: String,
}

/// Generate a stub workspace at `output_dir` for analyzing `target_crate`.
///
/// - `project_root`: path to the real project (contains root Cargo.toml)
/// - `target_crate`: name of the crate to analyze (e.g., "v-code-intel")
/// - `output_dir`: where to create the stub workspace
pub fn generate(
    project_root: &Path,
    target_crate: &str,
    output_dir: &Path,
) -> Result<StubWorkspace> {
    // 1. Parse dependency graph
    let meta = parse_cargo_metadata(project_root)?;

    // Verify target crate exists
    anyhow::ensure!(
        meta.contains_key(target_crate),
        "target crate '{target_crate}' not found in workspace"
    );

    // 2. Collect workspace crate names
    let workspace_crates: Vec<&str> = meta
        .iter()
        .filter(|(_, info)| info.is_workspace)
        .map(|(name, _)| name.as_str())
        .collect();

    // 3. Collect unique external deps across all workspace crates
    let mut all_external: HashMap<String, ExternalDep> = HashMap::new();
    for info in meta.values().filter(|i| i.is_workspace) {
        for dep in &info.external_deps {
            all_external
                .entry(dep.name.clone())
                .and_modify(|existing| {
                    for f in &dep.features {
                        if !existing.features.contains(f) {
                            existing.features.push(f.clone());
                        }
                    }
                    if existing.source_dir.is_none() {
                        existing.source_dir.clone_from(&dep.source_dir);
                    }
                })
                .or_insert_with(|| dep.clone());
        }
    }

    // 4. Create output directory structure
    std::fs::create_dir_all(output_dir)?;
    let stubs_dir = output_dir.join("stubs");
    std::fs::create_dir_all(&stubs_dir)?;

    // 5. Generate external crate stubs (auto-extracted from real source)
    for dep in all_external.values() {
        generate_external_stub(&stubs_dir, dep)?;
    }

    // 6. Generate workspace crate stubs (non-target: pub signatures only)
    for &crate_name in &workspace_crates {
        let info = &meta[crate_name];
        let crate_dir = output_dir.join(format!("{crate_name}-stub"));

        if crate_name == target_crate {
            // Target crate: copy Cargo.toml, symlink src/
            generate_target_crate(output_dir, info, &meta, target_crate)?;
        } else {
            // Non-target: extract pub signatures
            generate_workspace_stub(&crate_dir, info, &meta, target_crate)?;
        }
    }

    // 6. Generate root Cargo.toml
    generate_workspace_toml(output_dir, target_crate, &workspace_crates)?;

    Ok(StubWorkspace {
        root: output_dir.to_path_buf(),
        target_crate: target_crate.to_owned(),
    })
}

/// Parse `cargo metadata` output into our simplified format.
///
/// Runs full metadata (without `--no-deps`) to get external crate source paths
/// from the cargo registry, enabling automatic stub generation.
fn parse_cargo_metadata(project_root: &Path) -> Result<HashMap<String, CrateMeta>> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version=1"])
        .current_dir(project_root)
        .output()
        .context("failed to run cargo metadata")?;

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("failed to parse cargo metadata")?;

    let packages = json["packages"]
        .as_array()
        .context("no packages in metadata")?;

    // First pass: build name → source_dir map for external crates.
    // Uses the lib target's src_path to find the actual Rust source directory,
    // since some crates (e.g., tree-sitter) don't use the standard src/ layout.
    let mut ext_source_dirs: HashMap<String, PathBuf> = HashMap::new();
    for pkg in packages {
        if pkg["source"].as_str().is_some() {
            let name = pkg["name"].as_str().unwrap_or("");
            let lib_src_dir = pkg["targets"]
                .as_array()
                .and_then(|targets| {
                    targets.iter().find(|t| {
                        t["kind"]
                            .as_array()
                            .is_some_and(|k| k.iter().any(|v| v.as_str() == Some("lib")))
                    })
                })
                .and_then(|t| t["src_path"].as_str())
                .and_then(|sp| PathBuf::from(sp).parent().map(Path::to_path_buf));

            if let Some(dir) = lib_src_dir {
                if dir.is_dir() {
                    ext_source_dirs.insert(name.to_owned(), dir);
                }
            }
        }
    }

    // Second pass: build workspace crate metadata with resolved external dep source dirs.
    let mut result = HashMap::new();
    for pkg in packages {
        let source = pkg["source"].as_str();
        let is_workspace = source.is_none();
        if !is_workspace {
            continue; // Only workspace crates go into our result map.
        }

        let name = pkg["name"].as_str().unwrap_or("").to_owned();
        let version = pkg["version"].as_str().unwrap_or("0.0.0").to_owned();
        let manifest_path = PathBuf::from(pkg["manifest_path"].as_str().unwrap_or(""));

        let mut workspace_deps = Vec::new();
        let mut external_deps = Vec::new();

        if let Some(deps) = pkg["dependencies"].as_array() {
            for dep in deps {
                let dep_name = dep["name"].as_str().unwrap_or("").to_owned();
                let has_path = dep.get("path").is_some_and(|p| !p.is_null());
                let features: Vec<String> = dep
                    .get("features")
                    .and_then(|f| f.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                // Skip dev-dependencies for stubs
                let kind = dep.get("kind").and_then(|k| k.as_str()).unwrap_or("normal");
                if kind == "dev" {
                    continue;
                }

                if has_path {
                    workspace_deps.push(dep_name);
                } else {
                    let source_dir = ext_source_dirs.get(&dep_name).cloned();
                    external_deps.push(ExternalDep {
                        name: dep_name,
                        features,
                        source_dir,
                    });
                }
            }
        }

        result.insert(
            name.clone(),
            CrateMeta {
                name,
                _version: version,
                is_workspace,
                workspace_deps,
                external_deps,
                manifest_path,
                _features: Vec::new(),
            },
        );
    }

    Ok(result)
}

/// Generate a stub for an external crate by extracting pub signatures from its real source.
///
/// Uses the same `extract_pub_signatures` + `skeletonize_rust` pipeline as workspace
/// stubs. Compilation errors in stubs are expected and acceptable — RA only needs
/// parseable source, not compileable code, for type inference.
fn generate_external_stub(stubs_dir: &Path, dep: &ExternalDep) -> Result<()> {
    let name = &dep.name;
    let crate_dir = stubs_dir.join(name);
    let src_dir = crate_dir.join("src");
    std::fs::create_dir_all(&src_dir)?;

    // Cargo.toml with features declared
    let features_toml = if dep.features.is_empty() {
        String::new()
    } else {
        let mut ft = String::from("\n[features]\ndefault = []\n");
        let mut seen = std::collections::HashSet::new();
        for f in &dep.features {
            if f != "default" && seen.insert(f) {
                ft.push_str(&format!("{f} = []\n"));
            }
        }
        ft
    };

    let cargo_toml = format!(
        "[package]\nname = \"{name}\"\nversion = \"0.0.1\"\nedition = \"2024\"\n{features_toml}"
    );
    std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml)?;

    // Extract pub signatures from real source if available.
    if let Some(real_src) = &dep.source_dir {
        extract_pub_signatures(real_src, &src_dir)?;
    } else {
        std::fs::write(src_dir.join("lib.rs"), format!("//! Stub for {name}\n"))?;
    }

    Ok(())
}

/// Generate target crate directory with symlinked src/.
fn generate_target_crate(
    output_dir: &Path,
    info: &CrateMeta,
    all_meta: &HashMap<String, CrateMeta>,
    target_crate: &str,
) -> Result<()> {
    let crate_dir = output_dir.join(&info.name);
    std::fs::create_dir_all(&crate_dir)?;

    // Generate Cargo.toml pointing to stubs
    let cargo_toml = build_crate_cargo_toml(info, all_meta, target_crate)?;
    std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml)?;

    // Symlink src/ → real source
    let real_src = info
        .manifest_path
        .parent()
        .context("no parent")?
        .join("src");
    let link_path = crate_dir.join("src");

    if link_path.exists() {
        std::fs::remove_dir_all(&link_path).ok();
        std::fs::remove_file(&link_path).ok();
    }

    #[cfg(unix)]
    std::os::unix::fs::symlink(&real_src, &link_path)?;
    #[cfg(windows)]
    {
        // Use junction (mklink /J) instead of symlink — no admin privileges required.
        let link_str = link_path.to_string_lossy().replace('/', "\\");
        let target_str = real_src.to_string_lossy().replace('/', "\\");
        let output = Command::new("cmd")
            .arg("/c")
            .arg("mklink")
            .arg("/J")
            .arg(&*link_str)
            .arg(&*target_str)
            .output()
            .context("failed to run mklink /J")?;
        anyhow::ensure!(
            output.status.success(),
            "mklink /J failed for {link_str}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(())
}

/// Generate workspace crate stub with junction to real source.
/// Uses the same junction approach as target crate so RA can resolve types everywhere.
fn generate_workspace_stub(
    crate_dir: &Path,
    info: &CrateMeta,
    all_meta: &HashMap<String, CrateMeta>,
    target_crate: &str,
) -> Result<()> {
    std::fs::create_dir_all(crate_dir)?;

    // Junction src/ → real source (same as target crate)
    let real_src = info
        .manifest_path
        .parent()
        .context("no parent")?
        .join("src");
    let link_path = crate_dir.join("src");

    if link_path.exists() {
        std::fs::remove_dir_all(&link_path).ok();
        std::fs::remove_file(&link_path).ok();
    }

    #[cfg(unix)]
    std::os::unix::fs::symlink(&real_src, &link_path)?;
    #[cfg(windows)]
    {
        let link_str = link_path.to_string_lossy().replace('/', "\\");
        let target_str = real_src.to_string_lossy().replace('/', "\\");
        let output = Command::new("cmd")
            .arg("/c")
            .arg("mklink")
            .arg("/J")
            .arg(&*link_str)
            .arg(&*target_str)
            .output()
            .context("failed to run mklink /J")?;
        anyhow::ensure!(
            output.status.success(),
            "mklink /J failed for {link_str}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Generate Cargo.toml
    let cargo_toml = build_crate_cargo_toml(info, all_meta, target_crate)?;
    std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml)?;

    Ok(())
}

/// Build a Cargo.toml for a crate, pointing deps to stubs.
fn build_crate_cargo_toml(
    info: &CrateMeta,
    all_meta: &HashMap<String, CrateMeta>,
    target_crate: &str,
) -> Result<String> {
    let mut toml = format!(
        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\n",
        info.name
    );

    // Workspace deps → point to stub dirs
    // Target crate has no -stub suffix; all others do.
    for dep_name in &info.workspace_deps {
        if all_meta.contains_key(dep_name.as_str()) {
            let dir_name = if dep_name == target_crate {
                dep_name.clone()
            } else {
                format!("{dep_name}-stub")
            };
            toml.push_str(&format!("{dep_name} = {{ path = \"../{dir_name}\" }}\n"));
        }
    }

    // External deps → point to stubs/<name>
    for dep in &info.external_deps {
        let rel_path = format!("../stubs/{}", dep.name);
        if dep.features.is_empty() {
            toml.push_str(&format!(
                "{} = {{ path = \"{}\" }}\n", dep.name, rel_path
            ));
        } else {
            let feat_str = dep
                .features
                .iter()
                .map(|f| format!("\"{f}\""))
                .collect::<Vec<_>>()
                .join(", ");
            toml.push_str(&format!(
                "{} = {{ path = \"{}\", features = [{}] }}\n",
                dep.name, rel_path, feat_str
            ));
        }
    }

    Ok(toml)
}

/// Generate root workspace Cargo.toml.
fn generate_workspace_toml(
    output_dir: &Path,
    target_crate: &str,
    workspace_crates: &[&str],
) -> Result<()> {
    let mut members: Vec<String> = Vec::new();
    for &name in workspace_crates {
        if name == target_crate {
            members.push(name.to_owned());
        } else {
            members.push(format!("{name}-stub"));
        }
    }
    let members_str = members
        .iter()
        .map(|m| format!("    \"{m}\""))
        .collect::<Vec<_>>()
        .join(",\n");

    let toml = format!(
        "[workspace]\nmembers = [\n{members_str}\n]\nresolver = \"2\"\n"
    );
    std::fs::write(output_dir.join("Cargo.toml"), toml)?;

    Ok(())
}

/// Extract pub signatures from a Rust source directory into stub files.
///
/// Walks all .rs files, replaces function bodies with `unimplemented!()`,
/// strips `#[derive(...)]` attributes (no proc macros in stubs),
/// and skips `#[cfg(test)]` modules.
fn extract_pub_signatures(real_src: &Path, stub_src: &Path) -> Result<()> {
    for entry in walkdir(real_src)? {
        let rel = entry
            .strip_prefix(real_src)
            .unwrap_or(&entry);

        // Skip test directories
        if rel.components().any(|c| c.as_os_str() == "tests") {
            continue;
        }

        let dest = stub_src.join(rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let source = std::fs::read_to_string(&entry)?;
        let skeleton = skeletonize_rust(&source);
        std::fs::write(&dest, skeleton)?;
    }

    Ok(())
}

/// Simple recursive file walk (no external dep needed).
fn walkdir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !dir.is_dir() {
        return Ok(files);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            files.extend(walkdir(&path)?);
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path);
        }
    }
    Ok(files)
}

/// Replace function bodies with `unimplemented!()` and strip problematic items.
///
/// Strips: derive macros, serde/bincode attributes, extern crate imports,
/// `#[cfg(test)]` blocks. Replaces fn bodies with `unimplemented!()`.
/// Goal: produce code that rust-analyzer can parse without external proc macros.
fn skeletonize_rust(source: &str) -> String {
    let mut result = String::with_capacity(source.len() / 2);
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Skip #[cfg(test)] mod blocks
        if trimmed.starts_with("#[cfg(test)]") {
            i += 1;
            if i < lines.len() && lines[i].trim().starts_with("mod ") {
                skip_brace_block(&lines, &mut i);
            }
            continue;
        }

        // Strip proc-macro derives but keep std derives.
        // External crates are real versions now, so std derives (Clone, Debug, etc.)
        // must remain for workspace stubs to compile.
        if trimmed.starts_with("#[derive(") {
            let filtered = filter_std_derives(trimmed);
            if let Some(kept) = filtered {
                let indent = line.len() - line.trim_start().len();
                result.push_str(&" ".repeat(indent));
                result.push_str(&kept);
                result.push('\n');
            }
            i += 1;
            continue;
        }

        // Strip serde/bincode/thiserror attributes (not available in stubs)
        if trimmed.starts_with("#[serde")
            || trimmed.starts_with("#[error")
            || trimmed.starts_with("#[from]")
            || trimmed.starts_with("#[expect(")
            || trimmed.starts_with("#[allow(")
        {
            i += 1;
            continue;
        }

        // Detect function definition
        let is_fn = trimmed.contains("fn ")
            && !trimmed.starts_with("//")
            && !trimmed.starts_with("*");

        if is_fn && trimmed.ends_with('{') {
            // Single-line fn signature
            result.push_str(line);
            result.push('\n');
            let indent = line.len() - line.trim_start().len();
            result.push_str(&" ".repeat(indent + 4));
            result.push_str("unimplemented!()\n");
            i += 1;
            skip_brace_body(&lines, &mut i);
            // Write closing brace
            if i < lines.len() {
                result.push_str(lines[i]);
                result.push('\n');
            }
            i += 1;
        } else if is_fn && !trimmed.contains(';') && !trimmed.ends_with('}') {
            // Multi-line fn signature: collect until '{'
            let sig_start = i;
            let indent = line.len() - line.trim_start().len();
            while i < lines.len() && !lines[i].contains('{') {
                result.push_str(lines[i]);
                result.push('\n');
                i += 1;
            }
            if i < lines.len() {
                result.push_str(lines[i]);
                result.push('\n');
                i += 1;
            }
            result.push_str(&" ".repeat(indent + 4));
            result.push_str("unimplemented!()\n");
            skip_brace_body(&lines, &mut i);
            if i < lines.len() {
                result.push_str(lines[i]);
                result.push('\n');
            }
            let _ = sig_start; // suppress unused warning
            i += 1;
        } else {
            result.push_str(line);
            result.push('\n');
            i += 1;
        }
    }

    // Strip inline proc-macro attributes that aren't available in stubs
    let result = result.replace("#[from] ", "").replace("#[from]", "");

    // Ensure brace balance — skeletonize may miscount braces inside
    // char/string literals. Append missing closing braces if needed.
    let mut depth: i32 = 0;
    for ch in result.chars() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
    }
    if depth > 0 {
        let mut result = result;
        for _ in 0..depth {
            result.push_str("}\n");
        }
        return result;
    }

    result
}

/// Standard derives that don't need proc macros.
const STD_DERIVES: &[&str] = &[
    "Clone", "Copy", "Debug", "Default", "Hash",
    "PartialEq", "Eq", "PartialOrd", "Ord",
];

/// Filter a `#[derive(...)]` line, keeping only std derives.
/// Returns `None` if no std derives remain.
fn filter_std_derives(line: &str) -> Option<String> {
    let start = line.find('(')? + 1;
    let end = line.rfind(')')?;
    let inner = &line[start..end];

    let kept: Vec<&str> = inner
        .split(',')
        .map(str::trim)
        .filter(|d| STD_DERIVES.contains(d))
        .collect();

    if kept.is_empty() {
        None
    } else {
        Some(format!("#[derive({})]", kept.join(", ")))
    }
}

/// Skip lines until matching closing brace (brace_count goes to 0).
fn skip_brace_body(lines: &[&str], i: &mut usize) {
    let mut depth: i32 = 1;
    while *i < lines.len() && depth > 0 {
        for ch in lines[*i].chars() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        if depth > 0 {
            *i += 1;
        }
    }
    // *i now points to the line with the closing brace
}

/// Skip an entire brace-delimited block (e.g., `mod tests { ... }`).
fn skip_brace_block(lines: &[&str], i: &mut usize) {
    let mut depth: i32 = 0;
    while *i < lines.len() {
        for ch in lines[*i].chars() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        *i += 1;
        if depth > 0 {
            // entered the block
        }
        if depth == 0 && *i > 0 {
            break;
        }
    }
}

/// Staleness check result — tells caller what needs regenerating.
#[derive(Debug)]
pub enum StubStatus {
    /// Everything is current.
    UpToDate,
    /// Dependency graph changed (Cargo.toml modified) — full regeneration needed.
    DepsChanged,
    /// Only workspace crate pub APIs changed — incremental stub update.
    ApiChanged(Vec<String>),
    /// No stub workspace exists yet.
    Missing,
}

/// Check if a stub workspace exists and is up-to-date.
///
/// Checks two layers:
/// 1. **Cargo.toml timestamps** — any workspace crate's Cargo.toml newer than marker
///    means deps may have changed → full regen.
/// 2. **Pub API hashes** — hash of pub signatures per workspace crate, stored in
///    `.stub-apihash`. If only pub API changed (not deps), only affected stubs
///    need regeneration.
pub fn check_status(output_dir: &Path, project_root: &Path) -> StubStatus {
    let marker = output_dir.join(".stub-timestamp");
    let hash_file = output_dir.join(".stub-apihash");

    let Some(marker_time) = std::fs::metadata(&marker)
        .ok()
        .and_then(|m| m.modified().ok())
    else {
        return StubStatus::Missing;
    };

    // Layer 1: Check ALL Cargo.toml files (root + each crate)
    let root_toml = project_root.join("Cargo.toml");
    if is_newer_than(&root_toml, marker_time) {
        return StubStatus::DepsChanged;
    }

    // Find workspace crate directories from root Cargo.toml
    let crate_dirs = find_workspace_crate_dirs(project_root);
    for dir in &crate_dirs {
        let crate_toml = dir.join("Cargo.toml");
        if is_newer_than(&crate_toml, marker_time) {
            return StubStatus::DepsChanged;
        }
    }

    // Layer 2: Check pub API hashes for non-target workspace crates
    let old_hashes = load_api_hashes(&hash_file);
    let mut changed_crates = Vec::new();

    for dir in &crate_dirs {
        let crate_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let src_dir = dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }

        let current_hash = compute_pub_api_hash(&src_dir);
        let old_hash = old_hashes.get(crate_name).copied().unwrap_or(0);

        if current_hash != old_hash {
            changed_crates.push(crate_name.to_owned());
        }
    }

    if changed_crates.is_empty() {
        StubStatus::UpToDate
    } else {
        StubStatus::ApiChanged(changed_crates)
    }
}

/// Ensure stub workspace is up-to-date. Regenerates as needed.
///
/// Returns the stub workspace root path.
pub fn ensure_up_to_date(
    project_root: &Path,
    target_crate: &str,
    output_dir: &Path,
) -> Result<StubWorkspace> {
    match check_status(output_dir, project_root) {
        StubStatus::UpToDate => {
            eprintln!("[stub] workspace up-to-date");
            Ok(StubWorkspace {
                root: output_dir.to_path_buf(),
                target_crate: target_crate.to_owned(),
            })
        }
        StubStatus::Missing | StubStatus::DepsChanged => {
            eprintln!("[stub] full regeneration (deps changed or missing)");
            // Clean and regenerate everything
            if output_dir.exists() {
                // Preserve junction/symlink target by removing contents carefully
                remove_stub_contents(output_dir)?;
            }
            let ws = generate(project_root, target_crate, output_dir)?;
            save_all_api_hashes(output_dir, project_root)?;
            mark_up_to_date(output_dir)?;
            Ok(ws)
        }
        StubStatus::ApiChanged(changed) => {
            eprintln!("[stub] incremental update: {} crates changed", changed.len());
            let meta = parse_cargo_metadata(project_root)?;

            // Only regenerate stubs for changed crates
            for crate_name in &changed {
                if crate_name.as_str() == target_crate {
                    // Target crate uses junction — source changes are automatic
                    continue;
                }
                if let Some(info) = meta.get(crate_name.as_str()) {
                    let crate_dir = output_dir.join(format!("{crate_name}-stub"));
                    let src_dir = crate_dir.join("src");

                    // Remove old stub source and regenerate
                    if src_dir.is_dir() {
                        std::fs::remove_dir_all(&src_dir)?;
                    }
                    let real_src = info
                        .manifest_path
                        .parent()
                        .context("no parent")?
                        .join("src");
                    extract_pub_signatures(&real_src, &src_dir)?;
                    eprintln!("  [stub] updated {crate_name}");
                }
            }

            save_all_api_hashes(output_dir, project_root)?;
            mark_up_to_date(output_dir)?;

            Ok(StubWorkspace {
                root: output_dir.to_path_buf(),
                target_crate: target_crate.to_owned(),
            })
        }
    }
}

/// Touch the timestamp marker after successful generation.
fn mark_up_to_date(output_dir: &Path) -> Result<()> {
    let marker = output_dir.join(".stub-timestamp");
    std::fs::write(&marker, "")?;
    Ok(())
}

/// Check if a file is newer than the given reference time.
fn is_newer_than(path: &Path, reference: std::time::SystemTime) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .is_ok_and(|t| t > reference)
}

/// Find workspace crate directories from root Cargo.toml.
fn find_workspace_crate_dirs(project_root: &Path) -> Vec<PathBuf> {
    let toml_path = project_root.join("Cargo.toml");
    let Ok(content) = std::fs::read_to_string(&toml_path) else {
        return Vec::new();
    };

    // Simple parsing: extract members = ["crates/foo", "crates/bar"]
    let mut dirs = Vec::new();
    let mut in_members = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("members") {
            in_members = true;
        }
        if in_members {
            // Extract quoted paths
            let mut start = 0;
            while let Some(q1) = trimmed[start..].find('"') {
                let after = start + q1 + 1;
                if let Some(q2) = trimmed[after..].find('"') {
                    let path = &trimmed[after..after + q2];
                    dirs.push(project_root.join(path));
                    start = after + q2 + 1;
                } else {
                    break;
                }
            }
            if trimmed.contains(']') {
                in_members = false;
            }
        }
    }
    dirs
}

/// Compute a hash of all pub signatures in a crate's src/ directory.
///
/// This captures: pub fn signatures, pub struct/enum/trait definitions.
/// Body changes are ignored — only API surface matters for stubs.
fn compute_pub_api_hash(src_dir: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    let Ok(files) = walkdir(src_dir) else {
        return 0;
    };

    // Sort for deterministic hashing
    let mut files = files;
    files.sort();

    for file in &files {
        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };

        // Hash only lines containing pub declarations
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("pub ")
                || trimmed.starts_with("pub(crate)")
                || (trimmed.contains("fn ") && !trimmed.starts_with("//"))
            {
                trimmed.hash(&mut hasher);
            }
        }

        // Also hash use statements (affect type resolution)
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("use ") || trimmed.starts_with("pub use ") {
                trimmed.hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

/// Load saved API hashes from disk.
fn load_api_hashes(path: &Path) -> HashMap<String, u64> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return HashMap::new();
    };
    let mut map = HashMap::new();
    for line in content.lines() {
        if let Some((name, hash_str)) = line.split_once('=') {
            if let Ok(hash) = hash_str.trim().parse::<u64>() {
                map.insert(name.trim().to_owned(), hash);
            }
        }
    }
    map
}

/// Save current API hashes for all workspace crates.
fn save_all_api_hashes(output_dir: &Path, project_root: &Path) -> Result<()> {
    let hash_file = output_dir.join(".stub-apihash");
    let crate_dirs = find_workspace_crate_dirs(project_root);

    let mut content = String::new();
    for dir in &crate_dirs {
        let crate_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let src_dir = dir.join("src");
        if src_dir.is_dir() {
            let hash = compute_pub_api_hash(&src_dir);
            content.push_str(&format!("{crate_name}={hash}\n"));
        }
    }

    std::fs::write(&hash_file, content)?;
    Ok(())
}

/// Remove stub workspace contents without removing junction targets.
fn remove_stub_contents(output_dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Skip hidden files (.stub-timestamp, .stub-apihash)
        if name.starts_with('.') {
            continue;
        }

        if path.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}
