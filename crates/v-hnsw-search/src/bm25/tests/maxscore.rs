use std::collections::HashMap;

use rustc_hash::FxHashMap;
use v_hnsw_core::PointId;

use crate::bm25::index::Posting;
use crate::bm25::maxscore::maxscore_search;
use crate::bm25::scorer::Bm25Params;


fn make_postings(entries: &[(PointId, u32)]) -> Vec<Posting> {
    let mut postings: Vec<Posting> = entries
        .iter()
        .map(|&(doc_id, tf)| Posting { doc_id, tf })
        .collect();
    postings.sort_unstable_by_key(|p| p.doc_id);
    postings
}

fn make_doc_lengths(docs: &[(PointId, u32)]) -> HashMap<PointId, u32> {
    docs.iter().copied().collect()
}

#[test]
fn test_maxscore_basic() {
    let pl1 = make_postings(&[(1, 2), (2, 1), (3, 3)]);
    let pl2 = make_postings(&[(2, 1), (3, 1)]);
    let doc_lengths = make_doc_lengths(&[(1, 10), (2, 15), (3, 8)]);
    let params = Bm25Params::default();

    let terms: Vec<(&[Posting], f32)> = vec![
        (&pl1, params.idf(3, 3)),
        (&pl2, params.idf(2, 3)),
    ];

    let results = maxscore_search(&terms, 2, &params, &doc_lengths, 11.0, None, &FxHashMap::default());
    assert!(!results.is_empty());
    assert!(results.len() <= 2);
    assert_eq!(results[0].0, 3);
}

#[test]
fn test_maxscore_matches_brute_force() {
    let pl1 = make_postings(&[(1, 1), (2, 3), (3, 1), (4, 2)]);
    let pl2 = make_postings(&[(1, 2), (3, 1), (5, 1)]);
    let pl3 = make_postings(&[(2, 1), (4, 1), (5, 2)]);
    let doc_lengths = make_doc_lengths(&[
        (1, 10), (2, 20), (3, 10), (4, 15), (5, 12),
    ]);
    let params = Bm25Params::default();
    let avg_dl = 13.4;
    let total_docs = 5;

    let terms: Vec<(&[Posting], f32)> = vec![
        (&pl1, params.idf(pl1.len() as u32, total_docs)),
        (&pl2, params.idf(pl2.len() as u32, total_docs)),
        (&pl3, params.idf(pl3.len() as u32, total_docs)),
    ];

    let maxscore_results = maxscore_search(&terms, 3, &params, &doc_lengths, avg_dl, None, &FxHashMap::default());

    // Brute-force scoring
    let mut brute_scores: HashMap<PointId, f32> = HashMap::new();
    for &(postings, idf) in &terms {
        for p in postings {
            let dl = doc_lengths.get(&p.doc_id).copied().unwrap_or(0);
            *brute_scores.entry(p.doc_id).or_insert(0.0) +=
                idf * params.tf_norm(p.tf, dl, avg_dl);
        }
    }
    let mut brute_results: Vec<(PointId, f32)> = brute_scores.into_iter().collect();
    brute_results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    brute_results.truncate(3);

    assert_eq!(maxscore_results.len(), brute_results.len());
    for (ms, bf) in maxscore_results.iter().zip(brute_results.iter()) {
        assert_eq!(ms.0, bf.0, "doc ID mismatch");
        assert!((ms.1 - bf.1).abs() < 1e-5, "score mismatch: {} vs {}", ms.1, bf.1);
    }
}

#[test]
fn test_maxscore_empty() {
    let doc_lengths = HashMap::new();
    let params = Bm25Params::default();
    let terms: Vec<(&[Posting], f32)> = vec![];
    let results = maxscore_search(&terms, 10, &params, &doc_lengths, 0.0, None, &FxHashMap::default());
    assert!(results.is_empty());
}

#[test]
fn test_maxscore_single_term() {
    let pl = make_postings(&[(1, 3), (5, 1), (10, 2)]);
    let doc_lengths = make_doc_lengths(&[(1, 10), (5, 20), (10, 10)]);
    let params = Bm25Params::default();

    let terms: Vec<(&[Posting], f32)> = vec![(&pl, params.idf(3, 10))];
    let results = maxscore_search(&terms, 2, &params, &doc_lengths, 13.3, None, &FxHashMap::default());
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_maxscore_k_zero() {
    let pl = make_postings(&[(1, 1)]);
    let doc_lengths = make_doc_lengths(&[(1, 10)]);
    let params = Bm25Params::default();
    let terms: Vec<(&[Posting], f32)> = vec![(&pl, params.idf(1, 1))];
    let results = maxscore_search(&terms, 0, &params, &doc_lengths, 10.0, None, &FxHashMap::default());
    assert!(results.is_empty());
}

#[test]
fn test_maxscore_all_empty_posting_lists() {
    let pl1: Vec<Posting> = vec![];
    let pl2: Vec<Posting> = vec![];
    let doc_lengths = HashMap::new();
    let params = Bm25Params::default();
    let terms: Vec<(&[Posting], f32)> = vec![(&pl1, 1.0), (&pl2, 1.0)];
    let results = maxscore_search(&terms, 10, &params, &doc_lengths, 0.0, None, &FxHashMap::default());
    assert!(results.is_empty());
}

#[test]
fn test_maxscore_k_larger_than_docs() {
    let pl = make_postings(&[(1, 2), (2, 1)]);
    let doc_lengths = make_doc_lengths(&[(1, 10), (2, 15)]);
    let params = Bm25Params::default();
    let terms: Vec<(&[Posting], f32)> = vec![(&pl, params.idf(2, 2))];
    let results = maxscore_search(&terms, 100, &params, &doc_lengths, 12.5, None, &FxHashMap::default());
    assert_eq!(results.len(), 2);
}

#[test]
fn test_maxscore_no_overlap_between_terms() {
    let pl1 = make_postings(&[(1, 2), (2, 1)]);
    let pl2 = make_postings(&[(3, 1), (4, 3)]);
    let pl3 = make_postings(&[(5, 2)]);
    let doc_lengths = make_doc_lengths(&[(1, 10), (2, 15), (3, 8), (4, 12), (5, 20)]);
    let params = Bm25Params::default();
    let avg_dl = 13.0;
    let total_docs = 5;

    let terms: Vec<(&[Posting], f32)> = vec![
        (&pl1, params.idf(pl1.len() as u32, total_docs)),
        (&pl2, params.idf(pl2.len() as u32, total_docs)),
        (&pl3, params.idf(pl3.len() as u32, total_docs)),
    ];

    let results = maxscore_search(&terms, 5, &params, &doc_lengths, avg_dl, None, &FxHashMap::default());
    assert_eq!(results.len(), 5);
}

#[test]
fn test_maxscore_all_terms_same_posting_list() {
    let pl1 = make_postings(&[(1, 3), (2, 1)]);
    let pl2 = make_postings(&[(1, 2), (2, 4)]);
    let pl3 = make_postings(&[(1, 1), (2, 2)]);
    let doc_lengths = make_doc_lengths(&[(1, 10), (2, 15)]);
    let params = Bm25Params::default();
    let avg_dl = 12.5;

    let terms: Vec<(&[Posting], f32)> = vec![
        (&pl1, params.idf(2, 2)),
        (&pl2, params.idf(2, 2)),
        (&pl3, params.idf(2, 2)),
    ];

    let maxscore_results = maxscore_search(&terms, 2, &params, &doc_lengths, avg_dl, None, &FxHashMap::default());
    assert_eq!(maxscore_results.len(), 2);

    let mut brute_scores: HashMap<PointId, f32> = HashMap::new();
    for &(postings, idf) in &terms {
        for p in postings {
            let dl = doc_lengths.get(&p.doc_id).copied().unwrap_or(0);
            *brute_scores.entry(p.doc_id).or_insert(0.0) +=
                idf * params.tf_norm(p.tf, dl, avg_dl);
        }
    }
    let mut brute_results: Vec<(PointId, f32)> = brute_scores.into_iter().collect();
    brute_results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    for (ms, bf) in maxscore_results.iter().zip(brute_results.iter()) {
        assert_eq!(ms.0, bf.0);
        assert!((ms.1 - bf.1).abs() < 1e-5);
    }
}

#[test]
fn test_maxscore_results_sorted_descending() {
    let pl1 = make_postings(&[(1, 1), (2, 3), (3, 1), (4, 2), (5, 1)]);
    let pl2 = make_postings(&[(1, 2), (3, 1), (5, 1)]);
    let pl3 = make_postings(&[(2, 1), (4, 1), (5, 2)]);
    let doc_lengths = make_doc_lengths(&[
        (1, 10), (2, 20), (3, 10), (4, 15), (5, 12),
    ]);
    let params = Bm25Params::default();

    let terms: Vec<(&[Posting], f32)> = vec![
        (&pl1, params.idf(5, 5)),
        (&pl2, params.idf(3, 5)),
        (&pl3, params.idf(3, 5)),
    ];

    let results = maxscore_search(&terms, 5, &params, &doc_lengths, 13.4, None, &FxHashMap::default());
    for w in results.windows(2) {
        assert!(w[0].1 >= w[1].1, "Results not sorted descending: {:?}", results);
    }
}

#[test]
fn test_maxscore_missing_doc_length() {
    let pl1 = make_postings(&[(1, 2), (2, 1), (999, 3)]);
    let pl2 = make_postings(&[(1, 1), (999, 1)]);
    let pl3 = make_postings(&[(2, 1), (999, 2)]);
    let doc_lengths = make_doc_lengths(&[(1, 10), (2, 15)]);
    let params = Bm25Params::default();

    let terms: Vec<(&[Posting], f32)> = vec![
        (&pl1, params.idf(3, 3)),
        (&pl2, params.idf(2, 3)),
        (&pl3, params.idf(2, 3)),
    ];

    let results = maxscore_search(&terms, 5, &params, &doc_lengths, 10.0, None, &FxHashMap::default());
    assert!(!results.is_empty());
}
