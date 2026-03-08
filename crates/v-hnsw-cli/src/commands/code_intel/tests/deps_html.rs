use crate::commands::code_intel::deps_html;

#[test]
fn render_produces_valid_html_structure() {
    let html = deps_html::render(
        r#"{"n":"foo","f":"a.rs","k":"function","s":"fn foo()","l":"1-5","g":0}"#,
        r#"[0,0,"c"]"#,
        r#""group0""#,
    );

    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("<title>Call Graph Explorer</title>"));
    assert!(html.contains("d3.v7.min.js"));
    assert!(html.ends_with("</html>"));
}

#[test]
fn render_embeds_data() {
    let nodes = r#"{"n":"bar","f":"b.rs","k":"struct","s":"","l":"","g":0}"#;
    let links = r#"[0,1,"t"]"#;
    let groups = r#""crates/core""#;

    let html = deps_html::render(nodes, links, groups);

    assert!(html.contains(r#""n":"bar""#));
    assert!(html.contains(r#"[0,1,"t"]"#));
    assert!(html.contains(r#""crates/core""#));
}

#[test]
fn render_empty_data() {
    let html = deps_html::render("", "", "");
    // Should still produce valid HTML even with empty data
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("const N=[];"));
    assert!(html.contains("const E=[];"));
    assert!(html.contains("const G=[];"));
}
