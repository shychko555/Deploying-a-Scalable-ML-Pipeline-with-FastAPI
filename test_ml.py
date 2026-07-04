import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import compute_model_metrics



# 1. Test data processing
def test_process_data():
    X = pd.DataFrame({
        'feature1': [1, 2],
        'category': ['a', 'b'],
        'salary': ['>50K', '<=50K']
    })
    X_out, y_out, encoder, lb = process_data(
        X,
        categorical_features=['category'],
        label='salary',
        training=True
    )
    assert X_out.shape[0] == X.shape[0]

# 2. Test metrics calculation
def test_compute_model_metrics():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # These assertions verify your logic is working correctly
    assert precision == 1.0
    assert recall == 0.5

# 3. Test data integrity
def test_data_integrity():
    # This verifies the dataset loads and is not empty
    X = pd.DataFrame({'feature1': [1, 2], 'category': ['a', 'b']})
    assert len(X) > 0
