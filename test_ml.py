import pytest
import pandas as pd
import numpy as np
import os
from ml.data import process_data
from ml.model import compute_model_metrics, load_model
from sklearn.ensemble import RandomForestClassifier

# 1. Test data processing: Check shape and verify no nulls
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
    # Check shape
    assert X_out.shape[0] == X.shape[0]
    # Check that no null values exist in the output
    assert not np.isnan(X_out).any()

# 2. Test metrics calculation: Verify values are within range [0, 1]
def test_compute_model_metrics():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # Assert metrics are floats and within logical range
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0

# 3. Test data integrity: Ensure dataset is loaded and not null
def test_data_integrity():
    # Use the directory of the current file to build the path
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data", "census.csv")

    assert os.path.exists(data_path), f"File not found at {data_path}"
    data = pd.read_csv(data_path)
    # Check that the data is not empty
    assert data.shape[0] > 0
    # Check that there are no null values in the raw dataset
    assert data.isnull().sum().sum() == 0

# 4. Test model loading
def test_load_model():
    # Use the directory of the current file to build the path
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "model", "model.pkl")

    assert os.path.exists(model_path), f"File not found at {model_path}"
    model = load_model(model_path)
    # Verify the model is the correct type
    assert isinstance(model, RandomForestClassifier)
