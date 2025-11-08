# -*- coding: utf-8 -*-
"""
Tests for dataio module.
"""
import pytest
from dvtm_utils import dataio


def test_is_real_sentence():
    """Test sentence validation."""
    # Real sentence
    sentence = [('The', 'DT', 'O'), ('cat', 'NN', 'O')]
    assert dataio.is_real_sentence(False, sentence) is True

    # Document separator
    sentence_docstart = [('-DOCSTART-', 'X', 'O')]
    assert dataio.is_real_sentence(False, sentence_docstart) is False


def test_normalize_text():
    """Test text normalization for itWac embedding."""
    # Test URL detection
    assert dataio._normalize_text('http://example.com') == '___URL___'

    # Test long word
    assert dataio._normalize_text('a' * 30) == '__LONG-LONG__'

    # Test capitalization
    assert dataio._normalize_text('Hello') == 'Hello'
    assert dataio._normalize_text('hello') == 'hello'
    assert dataio._normalize_text('HELLO') == 'Hello'


def test_get_digits():
    """Test digit preprocessing."""
    # Valid year
    assert dataio._get_digits('2020') == '2020'

    # Large number
    assert dataio._get_digits('999999') == 'DIGLEN_6'

    # Non-numeric
    result = dataio._get_digits('abc123')
    assert '@Dg' in result


def test_itwac_preprocess_data():
    """Test itWac preprocessing."""
    sentences = [['Hello', 'world'], ['Test', '123']]
    result = dataio.itwac_preprocess_data(sentences)

    assert len(result) == 2
    assert len(result[0]) == 2
    assert isinstance(result[0][0], str)
