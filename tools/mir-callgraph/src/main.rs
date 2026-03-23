#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

use std::env;
use std::process::Command;

use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LOCAL_CRATE;
use serde::Serialize;

// ── Output types ────────────────────────────────────────────────────

#[derive(Serialize)]
struct CallEdge {
    caller: String,
    caller_file: String,
    caller_kind: String,
    callee: String,
    line: usize,
    is_local: bool,
}

/// Chunk definition extracted from MIR — replaces RA file_structure + source parsing.
#[derive(Serialize)]
struct MirChunk {
    name: String,
    file: String,
    kind: String,
    start_line: usize,
    end_line: usize,
    signature: Option<String>,
}

// ── Callbacks ───────────────────────────────────────────────────────

struct MirCallbacks {
    json: bool,
}

impl rustc_driver::Callbacks for MirCallbacks {
    fn after_analysis(
        &mut self,
        _compiler: &rustc_interface::interface::Compiler,
        tcx: TyCtxt<'_>,
    ) -> rustc_driver::Compilation {
        extract_all(tcx, self.json);
        rustc_driver::Compilation::Continue
    }
}

// ── Extraction ──────────────────────────────────────────────────────

fn extract_filename(source_map: &rustc_span::source_map::SourceMap, span: rustc_span::Span) -> String {
    match source_map.span_to_filename(span) {
        rustc_span::FileName::Real(ref name) => {
            let path_str = format!("{name:?}");
            if let Some(start) = path_str.find("name: \"") {
                let rest = &path_str[start + 7..];
                if let Some(end) = rest.find('"') {
                    return rest[..end].replace("\\\\", "/").to_string();
                }
            }
            path_str
        }
        other => format!("{other:?}"),
    }
}

fn extract_all(tcx: TyCtxt<'_>, json: bool) {
    let crate_name = tcx.crate_name(LOCAL_CRATE).to_string();
    let source_map = tcx.sess.source_map();

    let mut edges: Vec<CallEdge> = Vec::new();
    let mut chunks: Vec<MirChunk> = Vec::new();

    for &def_id in tcx.mir_keys(()) {
        let def_kind = tcx.def_kind(def_id);

        // Only functions/methods (skip closures for chunks, keep for edges)
        let is_fn = matches!(def_kind,
            rustc_hir::def::DefKind::Fn
            | rustc_hir::def::DefKind::AssocFn
        );
        let is_closure = matches!(def_kind, rustc_hir::def::DefKind::Closure);

        if !is_fn && !is_closure {
            // Also extract struct/enum/trait as chunks (no MIR body)
            let kind_str = match def_kind {
                rustc_hir::def::DefKind::Struct => "struct",
                rustc_hir::def::DefKind::Enum => "enum",
                rustc_hir::def::DefKind::Trait => "trait",
                rustc_hir::def::DefKind::Impl { .. } => "impl",
                _ => continue,
            };

            let hir_id = tcx.local_def_id_to_hir_id(def_id);
            let span = tcx.hir_span(hir_id);
            let file = extract_filename(source_map, span);
            let start = source_map.lookup_char_pos(span.lo());
            let end = source_map.lookup_char_pos(span.hi());

            chunks.push(MirChunk {
                name: tcx.def_path_str(def_id.to_def_id()),
                file,
                kind: kind_str.to_string(),
                start_line: start.line,
                end_line: end.line,
                signature: None,
            });
            continue;
        }

        let body = tcx.optimized_mir(def_id);
        let caller_name = tcx.def_path_str(def_id.to_def_id());
        let caller_file = extract_filename(source_map, body.span);
        let caller_kind = match def_kind {
            rustc_hir::def::DefKind::Fn => "fn",
            rustc_hir::def::DefKind::AssocFn => "method",
            rustc_hir::def::DefKind::Closure => "closure",
            _ => "other",
        };

        // Emit chunk for functions (not closures)
        if is_fn {
            let start = source_map.lookup_char_pos(body.span.lo());
            let end = source_map.lookup_char_pos(body.span.hi());

            // Extract signature from fn_sig
            let sig = tcx.fn_sig(def_id).skip_binder().skip_binder();
            let sig_str = format!("{sig:?}");

            chunks.push(MirChunk {
                name: caller_name.clone(),
                file: caller_file.clone(),
                kind: caller_kind.to_string(),
                start_line: start.line,
                end_line: end.line,
                signature: Some(sig_str),
            });
        }

        // Extract call edges
        for block in body.basic_blocks.iter() {
            let terminator = block.terminator();
            if let TerminatorKind::Call { ref func, .. } = terminator.kind {
                let func_ty = func.ty(&body.local_decls, tcx);
                let callee_def_id = match func_ty.kind() {
                    rustc_middle::ty::TyKind::FnDef(def_id, _) => *def_id,
                    _ => continue,
                };

                let callee_name = tcx.def_path_str(callee_def_id);
                let call_line = source_map
                    .lookup_char_pos(terminator.source_info.span.lo())
                    .line;

                edges.push(CallEdge {
                    caller: caller_name.clone(),
                    caller_file: caller_file.clone(),
                    caller_kind: caller_kind.to_string(),
                    callee: callee_name,
                    line: call_line,
                    is_local: callee_def_id.is_local(),
                });
            }
        }
    }

    // ── Output ──────────────────────────────────────────────────────
    let out_dir = env::var("MIR_CALLGRAPH_OUT").ok();

    if let Some(dir) = &out_dir {
        use std::io::Write;

        // Write edges
        let edge_path = format!("{dir}/{crate_name}.edges.jsonl");
        if let Ok(file) = std::fs::File::create(&edge_path) {
            let mut w = std::io::BufWriter::new(file);
            for edge in &edges {
                if let Ok(s) = serde_json::to_string(edge) {
                    let _ = writeln!(w, "{s}");
                }
            }
        }

        // Write chunks
        let chunk_path = format!("{dir}/{crate_name}.chunks.jsonl");
        if let Ok(file) = std::fs::File::create(&chunk_path) {
            let mut w = std::io::BufWriter::new(file);
            for chunk in &chunks {
                if let Ok(s) = serde_json::to_string(chunk) {
                    let _ = writeln!(w, "{s}");
                }
            }
        }

        eprintln!(
            "[mir-callgraph] {crate_name}: {} edges, {} chunks",
            edges.len(), chunks.len()
        );
    } else if json {
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut w = std::io::BufWriter::new(stdout.lock());
        for edge in &edges {
            if let Ok(s) = serde_json::to_string(edge) {
                let _ = writeln!(w, "{s}");
            }
        }
    } else {
        eprintln!(
            "[mir-callgraph] {crate_name}: {} edges, {} chunks",
            edges.len(), chunks.len()
        );
    }
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    // Mode 1: RUSTC_WRAPPER mode
    let is_wrapper = args.get(1).is_some_and(|a| a.contains("rustc") && !a.starts_with("-"));

    if is_wrapper {
        let rustc_args: Vec<String> = args[2..].to_vec();

        if env::var("MIR_CALLGRAPH_DEBUG").is_ok() {
            eprintln!("[mir-cg] wrapper args: {:?}", &rustc_args);
        }

        let is_local = rustc_args.iter().any(|a| {
            a.ends_with(".rs")
                && !a.contains(".cargo")
                && !a.contains("registry")
                && !a.contains("rustup")
        });
        let has_edition = rustc_args.iter().any(|a| a.starts_with("--edition"));
        let is_build_script = rustc_args.iter().any(|a| a == "build_script_build" || a.contains("build.rs"));

        if has_edition && is_local && !is_build_script {
            let mut full_args = vec![args[1].clone()];
            full_args.extend(rustc_args.iter().cloned());

            if !full_args.iter().any(|a| a.starts_with("--sysroot")) {
                if let Ok(output) = Command::new(&args[1]).arg("--print").arg("sysroot").output() {
                    let sysroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !sysroot.is_empty() {
                        full_args.push("--sysroot".to_string());
                        full_args.push(sysroot);
                    }
                }
            }

            let json = env::var("MIR_CALLGRAPH_JSON").is_ok();
            let mut callbacks = MirCallbacks { json };
            rustc_driver::run_compiler(&full_args, &mut callbacks);
        } else {
            let status = Command::new(&args[1])
                .args(&args[2..])
                .status()
                .expect("failed to run rustc");
            std::process::exit(status.code().unwrap_or(1));
        }
        return;
    }

    // Mode 2: CLI mode
    let json = args.iter().any(|a| a == "--json");
    let exe = env::current_exe().unwrap_or_default();

    let mut cmd = Command::new("cargo");
    cmd.arg("+nightly")
        .arg("check")
        .env("RUSTC_WRAPPER", &exe);

    if json {
        cmd.env("MIR_CALLGRAPH_JSON", "1");
    }

    for arg in args.iter().skip(1).filter(|a| *a != "--json") {
        cmd.arg(arg);
    }

    let status = cmd.status().expect("failed to run cargo check");
    std::process::exit(status.code().unwrap_or(1));
}
