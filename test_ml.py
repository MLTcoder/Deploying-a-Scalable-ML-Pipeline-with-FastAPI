import pytest
import pandas as pd
import numpy as np

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier


def test_for_expected_return():
    """
    A test that process_data returns numpy arrays and fitted encoders.
    """
    df = pd.DataFrame({
        "workclass":["Private", "Federal-gov"],
        "age": [25, 40],
        "salary": ["<=50K", ">50K"]
    })

    cat_features = ["workclass"]

    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == 2
    assert encoder is not None
    assert lb is not None

def test_for_expected_model():
    """
    Tests that train_model returns a RandomForestClassifier instance.
    """
    X = np.array([[0, 1], [1, 0], [1,1]])
    y = np.array([0, 1, 0])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)

def test_for_expected_metrics():
    """
    Tests compute_model_metrics returns expected values
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert pytest.approx(precision, 0.01) == 1.0
    assert pytest.approx(recall, 0.01) == 0.666666666
    assert pytest.approx(fbeta, 0.01) == 0.8
