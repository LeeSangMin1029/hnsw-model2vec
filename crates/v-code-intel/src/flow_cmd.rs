//! Interprocedural string flow tracing.
//!
//! Traces how a string literal value flows through wrapper functions
//! by following parameter-to-callee argument mappings in the call graph.

use crate::graph::CallGraph;

/// A single step in an interprocedural string flow trace.
pub struct FlowStep {
    /// Index of the chunk containing this step.
    pub chunk_idx: u32,
    /// Callee function receiving the value.
    pub callee: String,
    /// The string value or parameter description at this step.
    pub value: String,
    /// 0-based source line where the flow occurs.
    pub line: u32,
    /// `true` = direct string literal argument, `false` = parameter relay.
    pub is_direct: bool,
}

/// Trace where a string literal value flows through the call graph.
///
/// Returns a list of flow paths. Each path starts from a direct string literal
/// match in `string_args` and extends through parameter flows in callee functions.
pub fn trace_string_flow(graph: &CallGraph, query: &str, max_depth: u32) -> Vec<Vec<FlowStep>> {
    let lower = query.to_lowercase();
    let mut all_paths = Vec::new();

    // 1. Find direct string literal matches in string_args.
    for (chunk_idx, args) in graph.string_args.iter().enumerate() {
        for (callee, value, line, _) in args {
            if !value.to_lowercase().contains(&lower) {
                continue;
            }

            let mut path = vec![FlowStep {
                chunk_idx: chunk_idx as u32,
                callee: callee.clone(),
                value: value.clone(),
                line: *line,
                is_direct: true,
            }];

            // 2. Follow param_flows from the callee onwards.
            follow_param_flows(graph, callee, 0, &mut path, max_depth, 0);
            all_paths.push(path);
        }
    }

    all_paths
}

/// Recursively follow parameter flows from a callee function.
fn follow_param_flows(
    graph: &CallGraph,
    callee_name: &str,
    arg_pos: u8,
    path: &mut Vec<FlowStep>,
    max_depth: u32,
    current_depth: u32,
) {
    if current_depth >= max_depth {
        return;
    }

    let indices = graph.resolve(callee_name);
    for idx in indices {
        let i = idx as usize;
        if i >= graph.param_flows.len() {
            continue;
        }

        for (pname, ppos, next_callee, next_arg, line) in &graph.param_flows[i] {
            if *ppos == arg_pos {
                path.push(FlowStep {
                    chunk_idx: idx,
                    callee: next_callee.clone(),
                    value: format!("{pname} [param {ppos}]"),
                    line: *line,
                    is_direct: false,
                });
                follow_param_flows(
                    graph,
                    next_callee,
                    *next_arg,
                    path,
                    max_depth,
                    current_depth + 1,
                );
            }
        }
    }
}
