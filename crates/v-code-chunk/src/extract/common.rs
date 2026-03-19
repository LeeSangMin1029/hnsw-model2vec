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

/// Return the best source line for a call node.
///
/// For method calls (`call_expression` with `field_expression` function),
/// returns the line of the method name (field), not the receiver.
/// This correctly handles multiline chains:
/// ```text
/// engine          ← line 5 (receiver)
///     .checkpoint() ← line 6 (method name — we return this)
/// ```
fn call_site_line(node: &tree_sitter::Node) -> u32 {
    if (node.kind() == "call_expression" || node.kind() == "call")
        && let Some(func) = node.child_by_field_name("function") {
            if func.kind() == "field_expression"
                && let Some(field) = func.child_by_field_name("field") {
                    return field.start_position().row as u32;
                }
            // For scoped_identifier (Type::method), use the name part
            if func.kind() == "scoped_identifier"
                && let Some(name) = func.child_by_field_name("name") {
                    return name.start_position().row as u32;
                }
        }
    node.start_position().row as u32
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
        // For method calls (field_expression), use the method name's line,
        // not the receiver's line. This handles multiline chains like:
        //   engine
        //       .checkpoint()   ← method name is on this line
        let line = call_site_line(node);
        lines.push(line);
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
pub(crate) fn extract_callee_name(func_node: tree_sitter::Node, src: &[u8]) -> Option<String> {
    match func_node.kind() {
        "field_expression" => {
            let field = func_node.child_by_field_name("field")?;
            let method = field.utf8_text(src).ok()?;
            if let Some(value) = func_node.child_by_field_name("value") {
                // Peel transparent wrappers: `?`, `.await`, `(expr)` — iteratively.
                // Recovers receiver from `self.foo?.method()`, `(self.bar).method()`, etc.
                let inner = peel_transparent(value);
                match inner.kind() {
                    "identifier" | "self" => {
                        let recv = inner.utf8_text(src).ok()?;
                        return Some(format!("{recv}.{method}"));
                    }
                    "field_expression" => {
                        if let Some(recv) = extract_field_receiver(inner, src) {
                            return Some(format!("{recv}.{method}"));
                        }
                    }
                    // Chained call: a().b() or a()?.b() — extract inner callee's
                    // leaf method name as receiver hint for return-type resolution.
                    // e.g. `sema.type_of_expr(x)?.adjusted()` → "type_of_expr.adjusted"
                    "call_expression" => {
                        if let Some(leaf) = extract_inner_call_leaf(inner, src) {
                            return Some(format!("{leaf}.{method}"));
                        }
                    }
                    // Fallback: use receiver text if it's a clean identifier/path.
                    // Covers scoped identifiers (`AutoDistance::L2.distance`).
                    // Skip complex expressions (calls, long chains) — bare method is safer.
                    _ => {
                        if let Ok(recv) = inner.utf8_text(src) {
                            if !recv.contains('(') && recv.len() <= 60 {
                                return Some(format!("{recv}.{method}"));
                            }
                        }
                    }
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

/// Iteratively peel `?`, `.await`, and `(expr)` wrappers from an expression node.
fn peel_transparent(mut node: tree_sitter::Node) -> tree_sitter::Node {
    loop {
        match node.kind() {
            "try_expression" | "await_expression" => {
                if let Some(inner) = node.child(0) {
                    node = inner;
                } else {
                    break;
                }
            }
            "parenthesized_expression" => {
                if let Some(inner) = node.named_child(0) {
                    node = inner;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    node
}

/// Extract leaf method name from an inner `call_expression` for chained call resolution.
/// `sema.type_of_expr(x)` → "type_of_expr", `Foo::new()` → "new"
fn extract_inner_call_leaf(call_node: tree_sitter::Node, src: &[u8]) -> Option<String> {
    let func = call_node.child_by_field_name("function")?;
    let name = extract_callee_name(func, src)?;
    let leaf = name
        .rsplit_once('.')
        .map_or_else(|| name.rsplit_once("::").map_or(name.as_str(), |p| p.1), |p| p.1);
    Some(leaf.to_owned())
}

/// Extract receiver path from nested field expressions: `self.foo.bar` → "self.foo.bar"
/// Also unwraps `?` and `.await` transparently: `self.foo?.bar` → "self.foo.bar"
/// Stops at call expressions or other complex nodes.
pub(crate) fn extract_field_receiver(node: tree_sitter::Node, src: &[u8]) -> Option<String> {
    let field = node.child_by_field_name("field")?;
    let field_name = field.utf8_text(src).ok()?;
    let value = node.child_by_field_name("value")?;
    let value = peel_transparent(value);
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
pub(crate) fn is_noise_callee(name: &str) -> bool {
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
pub(crate) fn extract_string_value(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
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
pub fn strip_string_quotes(s: &str) -> String {
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

/// Extract struct field name-type pairs from a struct definition node.
///
/// Returns `Vec<(field_name, type_name)>` for struct fields. The type name
/// extracts only the base type identifier (strips generic parameters).
/// Works for Rust (`field_declaration`), Go, TypeScript, and C/C++.
pub(crate) fn extract_struct_field_types(
    node: &tree_sitter::Node,
    src: &[u8],
) -> Vec<(String, String)> {
    let body = node
        .child_by_field_name("body")
        .or_else(|| {
            let mut c = node.walk();
            node.children(&mut c)
                .find(|n| n.kind() == "field_declaration_list")
        });
    let Some(body) = body else {
        return Vec::new();
    };
    let mut fields = Vec::new();
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        let kind = child.kind();
        // Rust: field_declaration, Go: field_declaration, TS: property_signature,
        // C/C++: field_declaration
        if kind != "field_declaration" && kind != "property_signature" {
            continue;
        }
        let name = child
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(src).ok())
            .unwrap_or_default();
        let ty_text = child
            .child_by_field_name("type")
            .and_then(|n| extract_leaf_type_name(&n, src))
            .unwrap_or_default();
        if !name.is_empty() && !ty_text.is_empty() {
            fields.push((name.to_owned(), ty_text));
        }
    }
    fields
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
pub(crate) fn collect_param_names(node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, u8)> {
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

pub(crate) fn walk_param_flows_inner(
    node: &tree_sitter::Node,
    src: &[u8],
    params: &[(String, u8)],
    flows: &mut Vec<ParamFlow>,
) {
    if let Some(callee_name) = extract_callee_from_node(node, src)
        && !is_noise_callee(&callee_name)
            && let Some(args_node) = node.child_by_field_name("arguments") {
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
    _skip_kind: &str,
    strip_prefix: &str,
) -> Option<String> {
    let ret = node.child_by_field_name("return_type")?;

    // Return the full type text (e.g. "Result<Vec<Item>>") not just the leaf.
    ret.utf8_text(src)
        .ok()
        .map(|s| s.strip_prefix(strip_prefix).unwrap_or(s).to_owned())
}

/// Walk a function body to collect `let` binding type annotations.
///
/// Returns `(variable_name, leaf_type_name)` pairs.
/// For example, `let x: Vec<String> = ...;` yields `("x", "Vec")`.
/// Tuple patterns and complex destructuring are skipped.
pub(crate) fn walk_for_let_types(node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
    let mut result = Vec::new();
    walk_let_types_inner(node, src, &mut result);
    result
}

fn walk_let_types_inner(
    node: &tree_sitter::Node,
    src: &[u8],
    out: &mut Vec<(String, String)>,
) {
    let kind = node.kind();

    // Rust: let_declaration
    if kind == "let_declaration" {
        if let Some(pat) = node.child_by_field_name("pattern")
            && pat.kind() == "identifier"
            && let Ok(name) = pat.utf8_text(src)
        {
            if let Some(ty) = node.child_by_field_name("type")
                && let Some(leaf) = extract_leaf_type_name(&ty, src)
            {
                // Explicit type annotation: `let x: Type = ...`
                out.push((name.to_owned(), leaf));
            } else if let Some(val) = node.child_by_field_name("value")
                && let Some(leaf) = infer_type_from_rhs(&val, src)
            {
                // Inferred from RHS: `let x = Type::new()` or `let x = Type { .. }`
                out.push((name.to_owned(), leaf));
            }
        }
    }
    // Go: short_var_declaration (x := expr) has no type; var_declaration with type
    else if kind == "var_spec" {
        if let Some(name_node) = node.child_by_field_name("name")
            && let Ok(name) = name_node.utf8_text(src)
            && let Some(ty) = node.child_by_field_name("type")
            && let Some(leaf) = extract_leaf_type_name(&ty, src)
        {
            out.push((name.to_owned(), leaf));
        }
    }
    // TypeScript/JavaScript: variable_declarator with type_annotation
    else if kind == "variable_declarator" {
        if let Some(name_node) = node.child_by_field_name("name")
            && name_node.kind() == "identifier"
            && let Ok(name) = name_node.utf8_text(src)
            && let Some(ty) = node.child_by_field_name("type")
            && let Some(leaf) = extract_leaf_type_name(&ty, src)
        {
            out.push((name.to_owned(), leaf));
        }
    }
    // C/C++: declaration → type + declarator
    else if kind == "declaration" {
        if let Some(ty) = node.child_by_field_name("type")
            && let Some(leaf) = extract_leaf_type_name(&ty, src)
            && let Some(decl) = node.child_by_field_name("declarator")
        {
            // declarator may be identifier or pointer/array declarator
            let ident = find_identifier_in_declarator(&decl, src);
            if let Some(name) = ident {
                out.push((name, leaf));
            }
        }
    }
    // Python: type (annotated assignment) — `x: int = 5`
    else if kind == "assignment" || kind == "expression_statement" {
        // Python typed assignment: left : type = right
        // tree-sitter-python uses `type` field on assignment node
        if let Some(ty) = node.child_by_field_name("type")
            && let Some(leaf) = extract_leaf_type_name(&ty, src)
            && let Some(left) = node.child_by_field_name("left")
            && left.kind() == "identifier"
            && let Ok(name) = left.utf8_text(src)
        {
            out.push((name.to_owned(), leaf));
        }
    }

    // Enum variant pattern binding: `Expr::BinExpr(x)` → ("x", "BinExpr")
    // Works in if-let, match arms, and let declarations.
    // tuple_struct_pattern with scoped_identifier = qualified variant.
    if kind == "tuple_struct_pattern" {
        extract_variant_binding_type(node, src, out);
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_let_types_inner(&child, src, out);
    }
}

/// Extract variable bindings from enum variant patterns.
///
/// `Expr::BinExpr(x)` → `("x", "BinExpr")` — the variant name is the type hint.
/// Skips prelude wrappers like `Some(x)`, `Ok(x)`, `Err(e)` which don't give type info.
fn extract_variant_binding_type(
    pat: &tree_sitter::Node,
    src: &[u8],
    out: &mut Vec<(String, String)>,
) {
    static PRELUDE_WRAPPERS: &[&str] = &["Some", "Ok", "Err", "None", "Box"];

    // First child should be the variant path (scoped_identifier or identifier)
    let mut cursor = pat.walk();
    let children: Vec<_> = pat.children(&mut cursor).collect();

    // Find the variant type name
    let variant_type = children.iter().find_map(|child| {
        match child.kind() {
            // Expr::BinExpr(x) → scoped_identifier, last segment = "BinExpr"
            "scoped_identifier" => {
                let last = child.child_by_field_name("name")?;
                let name = last.utf8_text(src).ok()?;
                if name.starts_with(char::is_uppercase) && !PRELUDE_WRAPPERS.contains(&name) {
                    Some(name.to_owned())
                } else {
                    None
                }
            }
            // Some(x) → plain identifier, skip prelude wrappers
            "identifier" => {
                let name = child.utf8_text(src).ok()?;
                if name.starts_with(char::is_uppercase) && !PRELUDE_WRAPPERS.contains(&name) {
                    Some(name.to_owned())
                } else {
                    None
                }
            }
            _ => None,
        }
    });

    let Some(variant_type) = variant_type else { return };

    // Find bound variable identifiers (lowercase names inside the pattern)
    for child in &children {
        if child.kind() == "identifier" {
            if let Ok(name) = child.utf8_text(src) {
                if name.starts_with(char::is_lowercase) {
                    out.push((name.to_owned(), variant_type.clone()));
                }
            }
        }
    }
}

/// Walk a function body to collect `let x = callee()` bindings.
///
/// Returns `(variable_name, callee_name)` pairs for 1-hop return type propagation.
/// Handles `let x = foo()`, `let x = foo()?`, `let x = self.method()`, `let x = recv.method()`.
pub(crate) fn walk_for_let_call_bindings(
    node: &tree_sitter::Node,
    src: &[u8],
) -> Vec<(String, String)> {
    let mut result = Vec::new();
    walk_let_call_bindings_inner(node, src, &mut result);
    result
}

fn walk_let_call_bindings_inner(
    node: &tree_sitter::Node,
    src: &[u8],
    out: &mut Vec<(String, String)>,
) {
    match node.kind() {
        // `let x = foo()` or `let Some(x) = foo()` or `let (a, b) = foo()`
        "let_declaration" => {
            if let Some(pat) = node.child_by_field_name("pattern")
                && let Some(val) = node.child_by_field_name("value")
                && let Some(callee) = extract_rhs_callee(&val, src)
            {
                // For tuple patterns, emit a binding for each element.
                // All share the same callee — the graph layer uses return_type_map
                // to infer each variable's type.
                let bindings = extract_all_bindings_from_pattern(&pat, src);
                if bindings.is_empty() {
                    if let Some(var) = extract_binding_from_pattern(&pat, src) {
                        out.push((var, callee));
                    }
                } else {
                    for var in bindings {
                        out.push((var, callee.clone()));
                    }
                }
            }
        }
        // `if let Some(x) = foo()` — let_condition inside if_expression
        "let_condition" => {
            if let Some(pat) = node.child_by_field_name("pattern")
                && let Some(val) = node.child_by_field_name("value")
                && let Some(callee) = extract_rhs_callee(&val, src)
            {
                if let Some(var) = extract_binding_from_pattern(&pat, src) {
                    out.push((var, callee));
                }
            }
        }
        // `match expr { Some(x) => ... }` — match arm patterns
        "match_expression" => {
            if let Some(scrutinee) = node.child_by_field_name("value")
                && let Some(callee) = extract_rhs_callee(&scrutinee, src)
            {
                // Walk match arms to find pattern bindings
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "match_block" {
                        let mut arm_cursor = child.walk();
                        for arm in child.children(&mut arm_cursor) {
                            if arm.kind() == "match_arm"
                                && let Some(pat) = arm.child_by_field_name("pattern")
                                && let Some(var) = extract_binding_from_pattern(&pat, src)
                            {
                                out.push((var, callee.clone()));
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_let_call_bindings_inner(&child, src, out);
    }
}

/// Extract variable names from a pattern, handling:
/// - `identifier` → direct name
/// - `tuple_struct_pattern` (`Some(x)`, `Ok(x)`) → inner identifier
/// - `tuple_pattern` (`(a, b)`) → first identifier
/// - `struct_pattern` → skip (too complex)
fn extract_binding_from_pattern(pat: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    match pat.kind() {
        "identifier" => pat.utf8_text(src).ok().map(|s| s.to_owned()),
        "tuple_struct_pattern" => {
            // Some(x), Ok(x), Err(e) — find first identifier child that's not the type name
            let mut cursor = pat.walk();
            for child in pat.children(&mut cursor) {
                if child.kind() == "identifier" && child.is_named() {
                    // Skip the type name (first identifier = constructor name like "Some")
                    let is_type = child.utf8_text(src).ok()
                        .is_some_and(|t| t.starts_with(|c: char| c.is_uppercase()));
                    if !is_type {
                        return child.utf8_text(src).ok().map(|s| s.to_owned());
                    }
                }
            }
            None
        }
        "tuple_pattern" => {
            // (a, b) — return first non-underscore identifier
            let mut cursor = pat.walk();
            for child in pat.children(&mut cursor) {
                if child.kind() == "identifier" {
                    let text = child.utf8_text(src).ok()?;
                    if text != "_" {
                        return Some(text.to_owned());
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract ALL variable names from a pattern for multi-binding.
/// Used for tuple destructuring: `let (a, b) = callee()` → vec!["a", "b"].
fn extract_all_bindings_from_pattern(pat: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
    match pat.kind() {
        "identifier" => {
            if let Ok(text) = pat.utf8_text(src) {
                if text != "_" { return vec![text.to_owned()]; }
            }
            Vec::new()
        }
        "tuple_pattern" => {
            let mut names = Vec::new();
            let mut cursor = pat.walk();
            for child in pat.children(&mut cursor) {
                if child.kind() == "identifier" {
                    if let Ok(text) = child.utf8_text(src) {
                        if text != "_" {
                            names.push(text.to_owned());
                        }
                    }
                }
            }
            names
        }
        "tuple_struct_pattern" => {
            if let Some(name) = extract_binding_from_pattern(pat, src) {
                vec![name]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

/// Extract the callee name from a `let` RHS expression.
///
/// Peels `?` / `.await`, then extracts:
/// - `foo()` → `"foo"`
/// - `self.method()` → `"self.method"`
/// - `receiver.method()` → `"receiver.method"`
/// - `Type::new()` → `"Type::new"`
fn extract_rhs_callee(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    // Peel off `?` (try_expression) and `.await` (await_expression)
    let inner = match node.kind() {
        "try_expression" | "await_expression" => node.child(0)?,
        _ => *node,
    };

    if inner.kind() != "call_expression" {
        return None;
    }

    let func = inner.child_by_field_name("function")?;
    match func.kind() {
        "identifier" => func.utf8_text(src).ok().map(|s| s.to_owned()),
        "scoped_identifier" => func.utf8_text(src).ok().map(|s| s.to_owned()),
        "field_expression" => {
            // receiver.method — extract "receiver.method"
            let obj = func.child_by_field_name("value")?;
            let field = func.child_by_field_name("field")?;
            let obj_text = obj.utf8_text(src).ok()?;
            let field_text = field.utf8_text(src).ok()?;
            if obj.kind() == "identifier" || obj.kind() == "self" {
                Some(format!("{obj_text}.{field_text}"))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Walk a function body to collect non-call field accesses: `recv.field`.
///
/// Excludes method calls (`recv.method()`) — those are in `calls`.
/// Returns deduplicated `(receiver, field_name)` pairs.
pub(crate) fn walk_for_field_accesses(
    node: &tree_sitter::Node,
    src: &[u8],
) -> Vec<(String, String)> {
    let mut result = Vec::new();
    walk_field_accesses_inner(node, src, &mut result);
    result.sort();
    result.dedup();
    result
}

fn walk_field_accesses_inner(
    node: &tree_sitter::Node,
    src: &[u8],
    out: &mut Vec<(String, String)>,
) {
    if node.kind() == "field_expression" {
        // Skip if this field_expression is the function part of a call_expression
        // (that's a method call, not a field access).
        let is_call_func = node.parent().is_some_and(|p| {
            p.kind() == "call_expression"
                && p.child_by_field_name("function")
                    .is_some_and(|f| f.id() == node.id())
        });
        if !is_call_func
            && let Some(obj) = node.child_by_field_name("value")
                && let Some(field) = node.child_by_field_name("field")
                    && let (Ok(recv), Ok(fname)) = (obj.utf8_text(src), field.utf8_text(src)) {
                        // Only simple receivers (identifier, self) — not chains
                        if obj.kind() == "identifier" || obj.kind() == "self" {
                            out.push((recv.to_owned(), fname.to_owned()));
                        }
                    }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_field_accesses_inner(&child, src, out);
    }
}

/// Infer type from RHS expression of a `let` binding.
///
/// Handles:
/// - `Type::new()` / `Type::default()` / `Type::from(x)` → "Type"
/// - `Type { field: val }` (struct literal) → "Type"
/// - `Type(val)` (tuple struct constructor) → "Type"
/// - Unwrap wrappers: `Type::new()?.method()` ignores trailing `?`/`.await`
fn infer_type_from_rhs(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    // Peel off `?` (try_expression) and `.await` (await_expression)
    let inner = match node.kind() {
        "try_expression" | "await_expression" => {
            node.child(0)?
        }
        _ => *node,
    };

    match inner.kind() {
        // `Type::new()` — call_expression with scoped_identifier function
        "call_expression" => {
            let func = inner.child_by_field_name("function")?;
            match func.kind() {
                // Type::method() — scoped_identifier path=Type, name=method
                "scoped_identifier" => {
                    let path = func.child_by_field_name("path")?;
                    extract_leaf_type_name(&path, src)
                }
                // Type(val) — tuple struct or enum variant constructor
                "identifier" => {
                    let text = func.utf8_text(src).ok()?;
                    // Only PascalCase names (starts with uppercase) are types
                    if text.starts_with(|c: char| c.is_uppercase()) {
                        Some(text.to_owned())
                    } else {
                        None
                    }
                }
                // Turbofish: expr.cast::<BinExpr>() or foo::<MyType>()
                // generic_function wraps the inner function + type_arguments
                "generic_function" => {
                    let type_args = func.child_by_field_name("type_arguments")?;
                    // Use the first type argument as the inferred type
                    let mut cursor = type_args.walk();
                    for child in type_args.children(&mut cursor) {
                        if let Some(leaf) = extract_leaf_type_name(&child, src) {
                            return Some(leaf);
                        }
                    }
                    None
                }
                _ => None,
            }
        }
        // `Type { field: val }` — struct expression
        "struct_expression" => {
            let name_node = inner.child_by_field_name("name")?;
            extract_leaf_type_name(&name_node, src)
        }
        // `expr as Type` — type cast expression
        "type_cast_expression" => {
            let ty = inner.child_by_field_name("type")?;
            extract_leaf_type_name(&ty, src)
        }
        _ => None,
    }
}

/// Extract the leaf (outermost) type name from a type node.
///
/// Strips references, generics, pointers, etc. to get the base type name.
/// - `type_identifier` → text as-is
/// - `generic_type` → first child (the base type)
/// - `reference_type` / `pointer_type` → recurse into inner type
/// - `scoped_type_identifier` → last `type_identifier` child
/// - `type_annotation` (TS) → first meaningful child
fn extract_leaf_type_name(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" | "identifier" | "primitive_type" => {
            node.utf8_text(src).ok().map(|s| s.to_owned())
        }
        "generic_type" => {
            // First child is the base type (e.g., Vec in Vec<T>)
            let child = node.child(0)?;
            extract_leaf_type_name(&child, src)
        }
        "reference_type" | "pointer_type" | "mutable_specifier"
        | "dynamic_type" | "abstract_type" => {
            // Skip & / &mut / * / dyn / impl / lifetime and get the inner type
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                let ck = child.kind();
                if ck == "lifetime" || ck == "mutable_specifier" || ck == "&" {
                    continue;
                }
                if let Some(name) = extract_leaf_type_name(&child, src) {
                    return Some(name);
                }
            }
            None
        }
        "scoped_type_identifier" | "scoped_identifier" => {
            // Take the last type_identifier segment
            let mut cursor = node.walk();
            let mut last = None;
            for child in node.children(&mut cursor) {
                if child.kind() == "type_identifier" || child.kind() == "identifier" {
                    last = child.utf8_text(src).ok().map(|s| s.to_owned());
                }
            }
            last
        }
        "type_annotation" => {
            // TypeScript: skip the `: ` token and get the actual type
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() != ":" {
                    return extract_leaf_type_name(&child, src);
                }
            }
            None
        }
        _ => {
            // Try first named child as fallback
            if let Some(child) = node.named_child(0) {
                return extract_leaf_type_name(&child, src);
            }
            None
        }
    }
}

/// Find the identifier name inside a C/C++ declarator node.
fn find_identifier_in_declarator(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    if node.kind() == "identifier" {
        return node.utf8_text(src).ok().map(|s| s.to_owned());
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(name) = find_identifier_in_declarator(&child, src) {
            return Some(name);
        }
    }
    None
}

/// Extract variant names from an enum definition node.
///
/// Walks children of the enum body (`enum_variant_list` in Rust) and collects
/// identifier names of each variant.  Returns lowercase names suitable for
/// matching against call targets.
pub(crate) fn extract_enum_variant_names(node: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
    let mut variants = Vec::new();
    // Rust: body is `enum_variant_list`, children are `enum_variant` nodes.
    // Other langs: `enum_body` or similar — fall through gracefully.
    let body = node.child_by_field_name("body")
        .or_else(|| {
            let mut cursor = node.walk();
            node.children(&mut cursor).find(|c| {
                let k = c.kind();
                k.contains("variant_list") || k.contains("enum_body")
            })
        });
    let Some(body) = body else { return variants; };

    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        let kind = child.kind();
        // Rust: "enum_variant", TS: "enum_member", etc.
        if !kind.contains("variant") && !kind.contains("member") {
            continue;
        }
        // The name is typically the first `identifier` or `type_identifier` child,
        // or the `name` field.
        let name_node = child.child_by_field_name("name")
            .or_else(|| {
                let mut c2 = child.walk();
                child.children(&mut c2).find(|c| c.kind() == "identifier" || c.kind() == "type_identifier")
            });
        if let Some(n) = name_node
            && let Ok(text) = n.utf8_text(src) {
                variants.push(text.to_lowercase());
            }
    }
    variants
}
