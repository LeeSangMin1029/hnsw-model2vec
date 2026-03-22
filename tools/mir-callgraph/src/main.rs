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

#[derive(Serialize)]
struct CallEdge {
    caller: String,
    caller_file: String,
    callee: String,
    line: usize,
}

struct MirCallbacks {
    json: bool,
}

impl rustc_driver::Callbacks for MirCallbacks {
    fn after_analysis(
        &mut self,
        _compiler: &rustc_interface::interface::Compiler,
        tcx: TyCtxt<'_>,
    ) -> rustc_driver::Compilation {
        extract_call_edges(tcx, self.json);
        rustc_driver::Compilation::Continue
    }
}

fn extract_call_edges(tcx: TyCtxt<'_>, json: bool) {
    let crate_name = tcx.crate_name(LOCAL_CRATE).to_string();
    let source_map = tcx.sess.source_map();

    let mut edges: Vec<CallEdge> = Vec::new();

    for &def_id in tcx.mir_keys(()) {
        // Skip non-function items
        let def_kind = tcx.def_kind(def_id);
        if !matches!(def_kind,
            rustc_hir::def::DefKind::Fn
            | rustc_hir::def::DefKind::AssocFn
            | rustc_hir::def::DefKind::Closure
        ) {
            continue;
        }

        let body = tcx.optimized_mir(def_id);
        let caller_name = tcx.def_path_str(def_id.to_def_id());
        let caller_file = format!("{:?}", source_map.span_to_filename(body.span));

        for block in body.basic_blocks.iter() {
            let terminator = block.terminator();
            if let TerminatorKind::Call { ref func, .. } = terminator.kind {
                // Extract DefId from the function type
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
                    callee: callee_name,
                    line: call_line,
                });
            }
        }
    }

    // Write to file if MIR_CALLGRAPH_OUT is set, otherwise stdout
    let out_dir = env::var("MIR_CALLGRAPH_OUT").ok();

    if let Some(dir) = &out_dir {
        let path = format!("{dir}/{crate_name}.jsonl");
        if let Ok(file) = std::fs::File::create(&path) {
            let mut w = std::io::BufWriter::new(file);
            for edge in &edges {
                if let Ok(s) = serde_json::to_string(edge) {
                    use std::io::Write;
                    let _ = writeln!(w, "{s}");
                }
            }
        }
        eprintln!("[mir-callgraph] {crate_name}: {} edges → {path}", edges.len());
    } else if json {
        let stdout = std::io::stdout();
        let mut w = std::io::BufWriter::new(stdout.lock());
        for edge in &edges {
            if let Ok(s) = serde_json::to_string(edge) {
                use std::io::Write;
                let _ = writeln!(w, "{s}");
            }
        }
    } else {
        eprintln!("[mir-callgraph] {crate_name}: {} edges", edges.len());
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Mode 1: RUSTC_WRAPPER mode — cargo calls us with: <wrapper> <rustc> <args...>
    // Detect by checking if arg[1] looks like a rustc path
    let is_wrapper = args.get(1).is_some_and(|a| a.contains("rustc") && !a.starts_with("-"));

    if is_wrapper {
        // args[0] = us, args[1] = rustc path, args[2..] = rustc args
        let rustc_args: Vec<String> = args[2..].to_vec();

        // Debug: dump args
        if env::var("MIR_CALLGRAPH_DEBUG").is_ok() {
            eprintln!("[mir-cg] wrapper args: {:?}", &rustc_args);
        }

        // Check if this is a local lib/bin crate (not build script, not from registry)
        let is_local = rustc_args.iter().any(|a| {
            a.ends_with(".rs")
                && !a.contains(".cargo")
                && !a.contains("registry")
                && !a.contains("rustup")
        });
        let has_edition = rustc_args.iter().any(|a| a.starts_with("--edition"));
        let is_build_script = rustc_args.iter().any(|a| a == "build_script_build" || a.contains("build.rs"));
        let is_lib_or_bin = rustc_args.iter().any(|a| a == "lib" || a == "bin" || a == "proc-macro");

        if has_edition && is_local && !is_build_script {
            // Our target: intercept with MIR analysis
            let mut full_args = vec![args[1].clone()];
            full_args.extend(rustc_args.iter().cloned());

            // Ensure sysroot is set (needed for std resolution)
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
            // Pass through to real rustc (deps, version checks, build scripts)
            let status = Command::new(&args[1])
                .args(&args[2..])
                .status()
                .expect("failed to run rustc");
            std::process::exit(status.code().unwrap_or(1));
        }
        return;
    }

    // Mode 2: CLI mode — user runs: mir-callgraph [-p crate] [--json]
    let json = args.iter().any(|a| a == "--json");
    let exe = env::current_exe().unwrap_or_default();

    let mut cmd = Command::new("cargo");
    cmd.arg("+nightly")
        .arg("check")
        .env("RUSTC_WRAPPER", &exe);

    if json {
        cmd.env("MIR_CALLGRAPH_JSON", "1");
    }

    // Pass remaining args (e.g., -p crate_name)
    for arg in args.iter().skip(1).filter(|a| *a != "--json") {
        cmd.arg(arg);
    }

    let status = cmd.status().expect("failed to run cargo check");
    std::process::exit(status.code().unwrap_or(1));
}
