# -*- coding: utf-8 -*-
"""
Tests for modelutils module.
"""
import pytest
import numpy as np
from dvtm_utils import modelutils


class MockModel:
    """Mock model for testing."""
    def predict(self, dataset):
        return np.random.rand(len(dataset), 10)


def test_compute_prediction_latency():
    """Test prediction latency computation."""
    model = MockModel()
    dataset = np.random.rand(100, 10)

    latency = modelutils.compute_prediction_latency(dataset, model, n_instances=10)

    assert latency > 0
    assert isinstance(latency, float)


def test_from_encode_to_literal_labels():
    """Test label encoding to literal conversion."""
    y_true = [[0, 1, 2], [1, 2, 0]]
    y_pred = [[0, 1, 1], [1, 2, 0]]
    idx2tag = {0: 'O', 1: 'B-PER', 2: 'I-PER'}

    lit_true, lit_pred = modelutils.from_encode_to_literal_labels(
        y_true, y_pred, idx2tag
    )

    assert lit_true == [['O', 'B-PER', 'I-PER'], ['B-PER', 'I-PER', 'O']]
    assert lit_pred == [['O', 'B-PER', 'B-PER'], ['B-PER', 'I-PER', 'O']]
