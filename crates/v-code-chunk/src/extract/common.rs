//! Language-agnostic AST extraction helpers.
//!
//! Functions that work across all tree-sitter grammars: name, visibility,
//! signature, imports, calls, type refs, params, return type, doc comments.

/// Extract the symbol name from a node.
pub fn extract_name(node: &tree_sitter::Node, src: &[u8]) -> String {
    // Most items have a `name` field.
    if let Some(name_node) = node.child_by_field_name("name") {
        return name_node.utf8_text(src).unwrap_or_default().to_owned();
    }

    // impl blocks: `impl Trait for Type` or `impl Type`.
    if node.kind() == "impl_item" {
        let type_name = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(src).ok())
            .unwrap_or_default();

        if let Some(trait_node) = node.child_by_field_name("trait") {
            let trait_name = trait_node.utf8_text(src).unwrap_or_default();
            return format!("{trait_name} for {type_name}");
        }
        return type_name.to_owned();
    }

    String::new()
}

/// Extract visibility modifier (`pub`, `pub(crate)`, etc.).
pub fn extract_visibility(node: &tree_sitter::Node, src: &[u8]) -> String {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            return child.utf8_text(src).unwrap_or_default().to_owned();
        }
    }
    String::new()
}

/// Extract function signature (everything before the body block).
pub fn extract_function_signature(node: &tree_sitter::Node, src: &[u8]) -> String {
    if let Some(body) = node.child_by_field_name("body") {
        let sig_start = node.start_byte();
        let sig_end = body.start_byte();
        if sig_end > sig_start
            && let Ok(sig) = std::str::from_utf8(&src[sig_start..sig_end]) {
                return sig.trim().to_owned();
            }
    }
    node.utf8_text(src).unwrap_or_default().to_owned()
}

/// Extract all `use` declarations from the root node.
pub fn extract_imports(root: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
    super::lang::extract_imports_by_kind(root, src, &["use_declaration"])
}

/// Collect items via a walker, then sort and deduplicate.
pub fn collect_sorted_unique(
    node: &tree_sitter::Node,
    src: &[u8],
    walker: fn(&tree_sitter::Node, &[u8], &mut Vec<String>),
) -> Vec<String> {
    let mut items = Vec::new();
    walker(node, src, &mut items);
    items.sort();
    items.dedup();
    items
}

/// Recursively walk to find call nodes across languages (without line info).
///
/// Thin wrapper over `walk_for_calls_with_lines` — discards line data.
pub fn walk_for_calls(node: &tree_sitter::Node, src: &[u8], calls: &mut Vec<String>) {
    let mut lines = Vec::new();
    walk_for_calls_with_lines(node, src, calls, &mut lines);
}

/// Recursively walk to find call nodes, recording the 0-based source line of each call.
/// Extract and normalize the callee name from a call/method-invocation node.
///
/// Returns `None` for non-call nodes. Handles `call_expression`, `call`,
/// and `method_invocation` node kinds. Normalizes multiline names.
pub fn extract_callee_from_node(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    let raw = match node.kind() {
        "call_expression" | "call" => {
            node.child_by_field_name("function")
                .and_then(|f| extract_callee_name(f, src))
        }
        "method_invocation" => {
            node.child_by_field_name("name").and_then(|n| {
                n.utf8_text(src).ok().map(|name| {
                    if let Some(obj) = node.child_by_field_name("object")
                        && let Ok(obj_text) = obj.utf8_text(src) {
                            format!("{obj_text}.{name}")
                        } else {
                            name.to_owned()
                        }
                })
            })
        }
        _ => None,
    }?;

    // Normalize multiline call expressions: collapse whitespace
    // e.g. "self\n            .request" → "self.request"
    Some(if raw.contains('\n') {
        raw.split_whitespace().collect::<Vec<_>>().join("")
    } else {
        raw
    })
}

pub fn walk_for_calls_with_lines(
    node: &tree_sitter::Node,
    src: &[u8],
    calls: &mut Vec<String>,
    lines: &mut Vec<u32>,
) {
    if let Some(name) = extract_callee_from_node(node, src) {
        lines.push(node.start_position().row as u32);
        calls.push(name);
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_calls_with_lines(&child, src, calls, lines);
    }
}

/// A string literal argument found in a function call.
#[derive(Debug, Clone)]
pub struct StringArg {
    /// Callee name (e.g., `"Command::new"`).
    pub callee: String,
    /// String literal value (without quotes).
    pub value: String,
    /// 0-based source line.
    pub line: u32,
    /// 0-based argument position.
    pub arg_position: u8,
}

/// Recursively walk to find string literal arguments in call expressions.
///
/// For each call node, inspects arguments and collects those that are string
/// literals (`string_literal`, `string`, `interpreted_string_literal`,
/// `raw_string_literal`, `string_fragment`).
///
/// Also performs flow-insensitive constant propagation: if an argument is an
/// identifier bound to a string literal via `let`/`const`/`static`, the bound
/// value is used.
pub fn walk_for_string_args(node: &tree_sitter::Node, src: &[u8]) -> Vec<StringArg> {
    let mut result = Vec::new();
    let bindings = collect_string_bindings(node, src);
    walk_for_string_args_inner(node, src, &bindings, &mut result);
    result
}

/// Extract a clean callee name from a `function` field node of a `call_expression`.
///
/// For simple identifiers or scoped identifiers (`foo`, `Mod::func`), returns
/// the text directly. For `field_expression` nodes (method chains like
/// `a(x).b`), returns only the method name (e.g. `b`) since the receiver is
/// a separate call that gets walked independently.
fn extract_callee_name(func_node: tree_sitter::Node, src: &[u8]) -> Option<String> {
    match func_node.kind() {
        "field_expression" => {
            let field = func_node.child_by_field_name("field")?;
            let method = field.utf8_text(src).ok()?;
            if let Some(value) = func_node.child_by_field_name("value") {
                match value.kind() {
                    "identifier" | "self" => {
                        let recv = value.utf8_text(src).ok()?;
                        return Some(format!("{recv}.{method}"));
                    }
                    // self.field.method → "self.field.method"
                    "field_expression" => {
                        if let Some(recv) = extract_field_receiver(value, src) {
                            return Some(format!("{recv}.{method}"));
                        }
                    }
                    // chained call: a(x).b(y) → receiver is call_expression, just return method
                    _ => {}
                }
            }
            Some(method.to_owned())
        }
        _ => {
            let text = func_node.utf8_text(src).ok()?;
            if text.contains('(') {
                return None;
            }
            Some(text.to_owned())
        }
    }
}

/// Extract receiver path from nested field expressions: `self.foo.bar` → "self.foo.bar"
/// Stops at call expressions or other complex nodes.
fn extract_field_receiver(node: tree_sitter::Node, src: &[u8]) -> Option<String> {
    let field = node.child_by_field_name("field")?;
    let field_name = field.utf8_text(src).ok()?;
    let value = node.child_by_field_name("value")?;
    match value.kind() {
        "identifier" | "self" => {
            let recv = value.utf8_text(src).ok()?;
            Some(format!("{recv}.{field_name}"))
        }
        "field_expression" => {
            let inner = extract_field_receiver(value, src)?;
            Some(format!("{inner}.{field_name}"))
        }
        _ => None,
    }
}

/// Returns `true` for callee names that are enum variants, smart-pointer
/// constructors, error-handling chain methods, or other wrappers that don't
/// represent meaningful call sites for string-argument tracking.
fn is_noise_callee(name: &str) -> bool {
    matches!(
        name,
        // Enum variants & wrappers
        "Some" | "Ok" | "Err" | "None"
            | "Box::new" | "Arc::new" | "Rc::new"
            // Formatting & assertion macros
            | "vec" | "format" | "println" | "eprintln"
            | "write" | "writeln" | "panic" | "todo"
            | "unimplemented" | "unreachable" | "assert"
            | "assert_eq" | "assert_ne" | "debug_assert"
            | "debug_assert_eq" | "debug_assert_ne"
            // Error-handling chain methods (string args are error messages, not data flow)
            | "context" | "with_context" | "expect" | "unwrap_or"
            | "unwrap_or_else" | "map_err" | "ok_or" | "ok_or_else"
    )
}

fn walk_for_string_args_inner(
    node: &tree_sitter::Node,
    src: &[u8],
    bindings: &std::collections::HashMap<String, String>,
    out: &mut Vec<StringArg>,
) {
    if let Some(callee_name) = extract_callee_from_node(node, src) {
        // Skip enum variant constructors and wrappers — not meaningful call sites
        if is_noise_callee(&callee_name) {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                walk_for_string_args_inner(&child, src, bindings, out);
            }
            return;
        }

        if let Some(args_node) = node.child_by_field_name("arguments") {
            let mut cursor = args_node.walk();
            let mut pos: u8 = 0;
            for arg in args_node.children(&mut cursor) {
                if let Some(value) = extract_string_value(&arg, src) {
                    // Direct string literal
                    out.push(StringArg {
                        callee: callee_name.clone(),
                        value,
                        line: arg.start_position().row as u32,
                        arg_position: pos,
                    });
                } else if arg.kind() == "identifier"
                    && let Ok(ident) = arg.utf8_text(src)
                    && let Some(value) = bindings.get(ident)
                {
                    // Flow-insensitive constant propagation: resolve identifier
                    out.push(StringArg {
                        callee: callee_name.clone(),
                        value: value.clone(),
                        line: arg.start_position().row as u32,
                        arg_position: pos,
                    });
                }
                // Only count non-punctuation children as argument positions
                if !arg.kind().contains('(') && !arg.kind().contains(')')
                    && arg.kind() != "," && arg.kind() != "comment"
                {
                    pos = pos.saturating_add(1);
                }
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_string_args_inner(&child, src, bindings, out);
    }
}

/// Collect `let`/`const`/`static` bindings whose value is a string literal.
///
/// Traverses the subtree rooted at `node` and returns a map from variable name
/// to the unquoted string value. Only the first binding for each name is kept
/// (re-assignments are ignored for safety).
fn collect_string_bindings(
    node: &tree_sitter::Node,
    src: &[u8],
) -> std::collections::HashMap<String, String> {
    let mut bindings = std::collections::HashMap::new();
    collect_bindings_recursive(node, src, &mut bindings);
    bindings
}

fn collect_bindings_recursive(
    node: &tree_sitter::Node,
    src: &[u8],
    bindings: &mut std::collections::HashMap<String, String>,
) {
    if matches!(node.kind(), "let_declaration" | "const_item" | "static_item") {
        let name = node
            .child_by_field_name("pattern")
            .or_else(|| node.child_by_field_name("name"))
            .and_then(|n| n.utf8_text(src).ok())
            .map(|s| s.to_owned());
        let value = node
            .child_by_field_name("value")
            .and_then(|v| extract_string_value(&v, src));

        if let (Some(name), Some(value)) = (name, value) {
            // Keep only the first binding for each name
            bindings.entry(name).or_insert(value);
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_bindings_recursive(&child, src, bindings);
    }
}

/// Extract the string value from a node if it is a string literal.
///
/// Handles `string_literal`, `string`, `interpreted_string_literal`,
/// `raw_string_literal`, and `template_string` / `string_content` variants.
/// Returns `None` if the node is not a recognized string type.
fn extract_string_value(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    match node.kind() {
        "string_literal" | "string" | "interpreted_string_literal"
        | "raw_string_literal" | "string_value" => {
            let text = node.utf8_text(src).ok()?;
            Some(strip_string_quotes(text))
        }
        _ => None,
    }
}

/// Strip surrounding quotes from a string literal.
///
/// Handles `"..."`, `'...'`, `` `...` ``, `r"..."`, `r#"..."#` etc.
fn strip_string_quotes(s: &str) -> String {
    // Raw strings: r"...", r#"..."#
    if let Some(rest) = s.strip_prefix('r') {
        let hashes = rest.chars().take_while(|&c| c == '#').count();
        let prefix_len = 1 + hashes + 1; // r + #*n + "
        let suffix_len = 1 + hashes;     // " + #*n
        if s.len() > prefix_len + suffix_len {
            return s[prefix_len..s.len() - suffix_len].to_owned();
        }
    }
    // Standard quotes: "...", '...', `...`
    let trimmed = s.trim_matches(|c| c == '"' || c == '\'' || c == '`');
    trimmed.to_owned()
}

/// Extract doc comments (`///` or `//!`) preceding a node.
/// Collect doc-comment lines immediately before `target` under `root`.
///
/// `comment_kind` is the tree-sitter node kind for comments (e.g. `"line_comment"`,
/// `"comment"`). `prefixes` lists the prefix strings to strip (e.g. `&["///", "//!"]`).
/// Only comments that match at least one prefix are kept.
pub(super) fn collect_doc_comment_lines(
    root: &tree_sitter::Node,
    target: &tree_sitter::Node,
    src: &[u8],
    comment_kind: &str,
    prefixes: &[&str],
) -> Option<String> {
    let target_start = target.start_position().row;
    let mut doc_lines: Vec<String> = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        if child.start_position().row >= target_start {
            break;
        }

        if child.kind() == comment_kind {
            if let Ok(text) = child.utf8_text(src) {
                let stripped = prefixes
                    .iter()
                    .find_map(|p| text.strip_prefix(p));
                if let Some(doc) = stripped {
                    doc_lines.push(doc.trim().to_owned());
                }
            }
        } else {
            // Non-comment node between previous comments and target — reset
            doc_lines.clear();
        }
    }

    if doc_lines.is_empty() {
        None
    } else {
        Some(doc_lines.join("\n"))
    }
}

pub fn extract_doc_comment_before(
    root: &tree_sitter::Node,
    target: &tree_sitter::Node,
    src: &[u8],
) -> Option<String> {
    collect_doc_comment_lines(root, target, src, "line_comment", &["///", "//!"])
}

/// Walk recursively to collect `type_identifier` texts.
pub fn walk_for_type_ids(node: &tree_sitter::Node, src: &[u8], refs: &mut Vec<String>) {
    if node.kind() == "type_identifier"
        && let Ok(text) = node.utf8_text(src) {
            refs.push(text.to_owned());
        }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_type_ids(&child, src, refs);
    }
}

/// Extract struct/class field declarations as a compact signature string.
///
/// For Rust structs, returns `"field1: Type1, field2: Type2"`.
/// Returns `None` if the node has no `field_declaration_list` body.
pub fn extract_struct_fields(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    // Rust: "field_declaration_list", Go/TS: "body"
    let body = node.child_by_field_name("body")
        .or_else(|| {
            let mut c = node.walk();
            node.children(&mut c).find(|n| n.kind() == "field_declaration_list")
        })?;
    let mut fields = Vec::new();
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if child.kind() == "field_declaration" {
            let name = child.child_by_field_name("name")
                .and_then(|n| n.utf8_text(src).ok())
                .unwrap_or_default();
            let ty = child.child_by_field_name("type")
                .and_then(|n| n.utf8_text(src).ok())
                .unwrap_or_default();
            if !name.is_empty() && !ty.is_empty() {
                fields.push(format!("{name}: {ty}"));
            }
        }
    }
    if fields.is_empty() { None } else { Some(fields.join(", ")) }
}

/// Generic helper: extract parameter name-type pairs from a node's `parameters` field.
///
/// For each child whose kind is in `param_kinds`, calls `extractor` to produce
/// zero or more `(name, type)` pairs. This avoids duplicating the
/// "get parameters → filter by kind → collect" boilerplate across languages.
pub(super) fn extract_params_generic(
    node: &tree_sitter::Node,
    src: &[u8],
    param_kinds: &[&str],
    extractor: impl Fn(&tree_sitter::Node, &[u8]) -> Vec<(String, String)>,
) -> Vec<(String, String)> {
    let Some(params) = node.child_by_field_name("parameters") else {
        return Vec::new();
    };

    let mut result = Vec::new();
    let mut cursor = params.walk();

    for child in params.children(&mut cursor) {
        if !param_kinds.contains(&child.kind()) {
            continue;
        }
        result.extend(extractor(&child, src));
    }

    result
}

/// Helper: extract a single `(name, type)` pair via field names on a parameter node.
///
/// Returns a one-element vec on success, empty vec otherwise.
pub(super) fn field_name_type_pair(
    child: &tree_sitter::Node,
    src: &[u8],
    name_field: &str,
    type_field: &str,
) -> Vec<(String, String)> {
    let name = child
        .child_by_field_name(name_field)
        .and_then(|n| n.utf8_text(src).ok())
        .unwrap_or_default();
    let ty = child
        .child_by_field_name(type_field)
        .and_then(|n| n.utf8_text(src).ok())
        .unwrap_or_default();

    if !name.is_empty() && !ty.is_empty() {
        vec![(name.to_owned(), ty.to_owned())]
    } else {
        Vec::new()
    }
}

/// Extract parameter name-type pairs from a function's `parameters` field.
///
/// For each parameter, extracts the `pattern` (name) and `type` (type text).
/// Skips `self` parameters.
pub fn extract_param_types(
    node: &tree_sitter::Node,
    src: &[u8],
) -> Vec<(String, String)> {
    extract_params_generic(node, src, &["parameter"], |child, s| {
        field_name_type_pair(child, s, "pattern", "type")
    })
}

/// A parameter-to-callee argument flow within a function body.
#[derive(Debug, Clone)]
pub struct ParamFlow {
    /// Parameter name in the enclosing function.
    pub param_name: String,
    /// 0-based position of this parameter in the function signature.
    pub param_position: u8,
    /// Name of the callee receiving this parameter as an argument.
    pub callee: String,
    /// 0-based argument position in the callee's argument list.
    pub callee_arg: u8,
    /// 0-based source line where the flow occurs.
    pub line: u32,
}

/// Walk a function node to find parameter-to-callee argument flows.
///
/// For each call expression in the function body, if an argument is an
/// identifier matching one of the function's parameters, a `ParamFlow` is emitted.
pub fn walk_for_param_flows(node: &tree_sitter::Node, src: &[u8]) -> Vec<ParamFlow> {
    let params = collect_param_names(node, src);
    if params.is_empty() {
        return Vec::new();
    }
    let mut flows = Vec::new();
    walk_param_flows_inner(node, src, &params, &mut flows);
    flows
}

/// Collect parameter names and their 0-based positions from a function node.
fn collect_param_names(node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, u8)> {
    let Some(params_node) = node.child_by_field_name("parameters") else {
        return Vec::new();
    };
    let mut result = Vec::new();
    let mut cursor = params_node.walk();
    let mut pos: u8 = 0;
    for child in params_node.children(&mut cursor) {
        if child.kind() == "parameter" {
            if let Some(pattern) = child.child_by_field_name("pattern")
                && let Ok(name) = pattern.utf8_text(src)
            {
                result.push((name.to_owned(), pos));
            }
            pos = pos.saturating_add(1);
        } else if child.kind() == "self_parameter" {
            pos = pos.saturating_add(1);
        }
    }
    result
}

fn walk_param_flows_inner(
    node: &tree_sitter::Node,
    src: &[u8],
    params: &[(String, u8)],
    flows: &mut Vec<ParamFlow>,
) {
    if let Some(callee_name) = extract_callee_from_node(node, src) {
        if !is_noise_callee(&callee_name) {
            if let Some(args_node) = node.child_by_field_name("arguments") {
                let mut cursor = args_node.walk();
                let mut arg_pos: u8 = 0;
                for arg in args_node.children(&mut cursor) {
                    if arg.kind() == "identifier"
                        && let Ok(ident) = arg.utf8_text(src)
                        && let Some((_, param_pos)) =
                            params.iter().find(|(name, _)| name == ident)
                    {
                        flows.push(ParamFlow {
                            param_name: ident.to_owned(),
                            param_position: *param_pos,
                            callee: callee_name.clone(),
                            callee_arg: arg_pos,
                            line: arg.start_position().row as u32,
                        });
                    }
                    if !arg.kind().contains('(')
                        && !arg.kind().contains(')')
                        && arg.kind() != ","
                        && arg.kind() != "comment"
                    {
                        arg_pos = arg_pos.saturating_add(1);
                    }
                }
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_param_flows_inner(&child, src, params, flows);
    }
}

/// Extract the return type string from a function's `return_type` field.
pub fn extract_return_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    extract_return_type_skip(node, src, "->", "-> ")
}

/// Extract return type from a field node, skipping a delimiter token.
///
/// Used by Rust (`"->"`, `"-> "`) and TypeScript (`":"`, `": "`).
pub(super) fn extract_return_type_skip(
    node: &tree_sitter::Node,
    src: &[u8],
    skip_kind: &str,
    strip_prefix: &str,
) -> Option<String> {
    let ret = node.child_by_field_name("return_type")?;

    let mut cursor = ret.walk();
    for child in ret.children(&mut cursor) {
        if child.kind() != skip_kind {
            return child.utf8_text(src).ok().map(|s| s.to_owned());
        }
    }

    ret.utf8_text(src)
        .ok()
        .map(|s| s.strip_prefix(strip_prefix).unwrap_or(s).to_owned())
}
