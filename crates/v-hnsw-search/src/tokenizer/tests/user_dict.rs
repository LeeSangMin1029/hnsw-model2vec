use crate::tokenizer::user_dict::*;

#[test]
fn test_dictionary_entry_new() {
    let entry = DictionaryEntry::new("삼성전자", "NNP", "삼성전자");
    assert_eq!(entry.term, "삼성전자");
    assert_eq!(entry.pos, "NNP");
    assert_eq!(entry.reading, "삼성전자");
}

#[test]
fn test_dictionary_entry_simple() {
    let entry = DictionaryEntry::simple("카카오톡", "NNP");
    assert_eq!(entry.term, "카카오톡");
    assert_eq!(entry.pos, "NNP");
    assert_eq!(entry.reading, "카카오톡");
}

#[test]
fn test_user_dictionary_add() {
    let mut dict = UserDictionary::new();
    assert!(dict.is_empty());

    dict.add_term("네이버", "NNP");
    dict.add_entry(DictionaryEntry::new("인공지능", "NNG", "인공지능"));

    assert_eq!(dict.len(), 2);
    assert!(!dict.is_empty());
}

#[test]
fn test_load_from_str() {
    let csv = r#"
# This is a comment
삼성전자,NNP,삼성전자
카카오,NNP

# Another comment
네이버,NNP,네이버
"#;

    let dict = UserDictionary::load_from_str(csv).expect("valid csv");
    assert_eq!(dict.len(), 3);

    let entries = dict.entries();
    assert_eq!(entries[0].term, "삼성전자");
    assert_eq!(entries[1].term, "카카오");
    assert_eq!(entries[1].reading, "카카오"); // auto-filled
    assert_eq!(entries[2].term, "네이버");
}

#[test]
fn test_load_from_str_two_column() {
    let csv = "인공지능,NNG\n기계학습,NNG";
    let dict = UserDictionary::load_from_str(csv).expect("valid csv");
    assert_eq!(dict.len(), 2);
    assert_eq!(dict.entries()[0].term, "인공지능");
    assert_eq!(dict.entries()[0].reading, "인공지능");
}

#[test]
fn test_load_from_str_invalid() {
    let csv = "invalid,entry,with,too,many,columns";
    let result = UserDictionary::load_from_str(csv);
    assert!(result.is_err());
}

#[test]
fn test_to_lindera_csv() {
    let mut dict = UserDictionary::new();
    dict.add_entry(DictionaryEntry::new("삼성", "NNP", "삼성"));
    dict.add_entry(DictionaryEntry::new("전자", "NNG", "전자"));

    let csv = dict.to_lindera_csv();
    assert!(csv.contains("삼성,0,NNP,삼성"));
    assert!(csv.contains("전자,0,NNG,전자"));
}

#[test]
fn test_empty_dictionary() {
    let dict = UserDictionary::new();
    assert!(dict.is_empty());
    assert_eq!(dict.len(), 0);
    assert!(dict.to_lindera_csv().is_empty());
}

// ============================================================================
// Additional user_dict tests
// ============================================================================

#[test]
fn test_load_from_str_empty() {
    let dict = UserDictionary::load_from_str("").expect("empty is valid");
    assert!(dict.is_empty());
}

#[test]
fn test_load_from_str_only_comments_and_blanks() {
    let csv = "# comment 1\n\n# comment 2\n  \n";
    let dict = UserDictionary::load_from_str(csv).expect("valid");
    assert!(dict.is_empty());
}

#[test]
fn test_load_from_str_single_column_error() {
    let csv = "onlyonecolumn";
    let result = UserDictionary::load_from_str(csv);
    assert!(result.is_err());
}

#[test]
fn test_load_from_str_four_columns_error() {
    let csv = "a,b,c,d";
    let result = UserDictionary::load_from_str(csv);
    assert!(result.is_err());
}

#[test]
fn test_dictionary_entry_to_lindera_csv() {
    let entry = DictionaryEntry::new("테스트", "NNG", "테스트읽기");
    let csv = entry.to_lindera_csv();
    assert_eq!(csv, "테스트,0,NNG,테스트읽기");
}

#[test]
fn test_with_entries() {
    let entries = vec![
        DictionaryEntry::simple("가", "NNG"),
        DictionaryEntry::simple("나", "NNG"),
    ];
    let dict = UserDictionary::with_entries(entries);
    assert_eq!(dict.len(), 2);
    assert_eq!(dict.entries()[0].term, "가");
    assert_eq!(dict.entries()[1].term, "나");
}

#[test]
fn test_to_lindera_csv_multiline() {
    let mut dict = UserDictionary::new();
    dict.add_term("가", "NNG");
    dict.add_term("나", "NNP");
    let csv = dict.to_lindera_csv();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("가"));
    assert!(lines[1].contains("나"));
}

#[test]
fn test_load_from_str_whitespace_trimming() {
    let csv = "  삼성전자 , NNP , 삼성전자 ";
    let dict = UserDictionary::load_from_str(csv).expect("valid");
    assert_eq!(dict.len(), 1);
    assert_eq!(dict.entries()[0].term, "삼성전자");
    assert_eq!(dict.entries()[0].pos, "NNP");
    assert_eq!(dict.entries()[0].reading, "삼성전자");
}

#[test]
fn test_dictionary_entry_equality() {
    let e1 = DictionaryEntry::new("테스트", "NNG", "테스트");
    let e2 = DictionaryEntry::new("테스트", "NNG", "테스트");
    let e3 = DictionaryEntry::new("다른", "NNG", "다른");
    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}
