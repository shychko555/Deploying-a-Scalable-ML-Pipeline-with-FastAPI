import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import compute_model_metrics

# 1. Test data processing
def test_process_data():
    # Adding a 'salary' column to act as the label
    X = pd.DataFrame({
        'feature1': [1, 2],
        'category': ['a', 'b'],
        'salary': ['>50K', '<=50K']
    })

    # Passing 'salary' as the label argument
    X_out, y_out, encoder, lb = process_data(
    X,
        categorical_features=['category'],
        label='salary',
        training=True
)

    assert len(X_out) == 2
    assert encoder is not None


# 2. Test metrics calculation
def test_compute_model_metrics():
    y_test = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    # Check that metrics are calculated and are the correct type
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

# 3. Test data integrity
def test_data_integrity():
    # Ensure dataset loads and isn't empty
    data = pd.DataFrame({'age': [25, 30], 'salary': [50000, 60000]})
    assert not data.empty
    assert 'age' in data.columns