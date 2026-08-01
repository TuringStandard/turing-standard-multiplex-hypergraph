"""Tests for entity mention normalization."""

from mh_rag.ingest.normalization import clean_string, name_hash


def test_clean_string_whitespace_and_case():
    assert clean_string("  Insulin-like   Growth FACTOR ") == "insulin-like growth factor"


def test_name_hash_type_salted():
    assert name_hash("apple", "ORG") != name_hash("apple", "OTHER")


def test_clean_string_nfkc_fullwidth():
    assert clean_string("Ｇｌｕｃｏｓｅ") == "glucose"
