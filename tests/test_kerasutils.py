# -*- coding: utf-8 -*-
"""
Tests for kerasutils module.

Note: These tests require TensorFlow to be installed.
Most tests are skipped if TensorFlow is not available.
"""
import pytest
import numpy as np

try:
    from dvtm_utils import kerasutils
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
def test_create_bilstm():
    """Test BiLSTM model creation."""
    model = kerasutils.create_BiLSTM(
        vocabulary_size=1000,
        seq_len=50,
        n_classes=5,
        hidden_cells=64,
        embed_dim=32,
        drop=0.3
    )

    assert model is not None
    assert len(model.layers) > 0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
def test_create_paper_bilstm():
    """Test paper BiLSTM model creation."""
    model = kerasutils.create_paper_BiLSTM(
        vocabulary_size=1000,
        seq_len=50,
        n_classes=5,
        hidden_cells=100,
        embed_dim=50,
        drop=0.4
    )

    assert model is not None
    assert len(model.layers) > 0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
def test_remove_flat_padding():
    """Test flat padding removal."""
    X = np.array([[1, 2, 0, 0], [3, 4, 5, 0]])
    y_true = np.array([[0, 1, 2, 2], [1, 0, 1, 2]])
    y_pred = np.array([[0, 1, 1, 2], [1, 0, 0, 2]])

    new_true, new_pred = kerasutils.remove_flat_padding(X, y_true, y_pred, pad=0)

    # Should remove elements where X == 0 (3 padding elements)
    assert len(new_true) == 5
    assert len(new_pred) == 5


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
def test_remove_seq_padding():
    """Test sequence padding removal."""
    X = np.array([[1, 2, 0], [3, 0, 0]])
    y_true = np.array([[0, 1, 2], [1, 2, 2]])
    y_pred = np.array([[0, 1, 1], [1, 1, 2]])

    new_true, new_pred = kerasutils.remove_seq_padding(X, y_true, y_pred, pad=0)

    assert len(new_true) == 2
    assert len(new_true[0]) == 2  # First sequence has 2 non-padding elements
    assert len(new_true[1]) == 1  # Second sequence has 1 non-padding element
