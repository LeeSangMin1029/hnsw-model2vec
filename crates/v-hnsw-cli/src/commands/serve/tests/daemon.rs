use crate::commands::common::SearchResultItem;
use crate::commands::serve::daemon::SearchResponse;

#[test]
fn search_response_serialize_empty_results() {
    let resp = SearchResponse {
        results: vec![],
        elapsed_ms: 0.0,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["results"], serde_json::json!([]));
    assert_eq!(v["elapsed_ms"], serde_json::json!(0.0));
}

#[test]
fn search_response_serialize_with_results() {
    let resp = SearchResponse {
        results: vec![
            SearchResultItem {
                id: 1,
                score: 0.95,
                text: Some("hello world".into()),
                source: Some("test.txt".into()),
                title: None,
                url: None,
            },
            SearchResultItem {
                id: 2,
                score: 0.80,
                text: None,
                source: None,
                title: Some("Title".into()),
                url: Some("https://example.com".into()),
            },
        ],
        elapsed_ms: 12.345,
    };

    let json = serde_json::to_string(&resp).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["results"].as_array().unwrap().len(), 2);
    assert_eq!(v["elapsed_ms"], serde_json::json!(12.345));

    // First result has text and source, no title/url
    let r0 = &v["results"][0];
    assert_eq!(r0["id"], 1);
    assert!(r0.get("title").is_none());
    assert!(r0.get("url").is_none());

    // Second result has title and url, no text/source
    let r1 = &v["results"][1];
    assert_eq!(r1["id"], 2);
    assert!(r1.get("text").is_none());
    assert!(r1.get("source").is_none());
}

#[test]
fn search_response_elapsed_ms_precision() {
    let resp = SearchResponse {
        results: vec![],
        elapsed_ms: 0.001,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let elapsed = v["elapsed_ms"].as_f64().unwrap();
    assert!((elapsed - 0.001).abs() < 1e-9);
}

#[test]
fn is_newer_both_exist() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.txt");
    let b = dir.path().join("b.txt");

    // Create b first, then a — a should be newer
    std::fs::write(&b, "old").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    std::fs::write(&a, "new").unwrap();

    assert!(super::super::daemon::is_newer(&a, &b));
    assert!(!super::super::daemon::is_newer(&b, &a));
}

#[test]
fn is_newer_a_missing() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("missing.txt");
    let b = dir.path().join("b.txt");
    std::fs::write(&b, "data").unwrap();

    // a doesn't exist → false
    assert!(!super::super::daemon::is_newer(&a, &b));
}

#[test]
fn is_newer_b_missing() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.txt");
    let b = dir.path().join("missing.txt");
    std::fs::write(&a, "data").unwrap();

    // b doesn't exist → true (a is considered newer)
    assert!(super::super::daemon::is_newer(&a, &b));
}

#[test]
fn is_newer_both_missing() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("x.txt");
    let b = dir.path().join("y.txt");

    // a doesn't exist → false
    assert!(!super::super::daemon::is_newer(&a, &b));
}
