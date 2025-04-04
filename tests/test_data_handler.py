import pytest
import pandas as pd
import numpy as np
import os
from app.utils.data_handler import DataHandler


@pytest.fixture
def sample_data():
    # Create a sample CSV file
    data = {
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10),
        "target": np.random.randint(0, 2, 10),
    }
    df = pd.DataFrame(data)
    filepath = "test_data.csv"
    df.to_csv(filepath, index=False)
    yield filepath
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)


def test_data_handler_initialization(sample_data):
    handler = DataHandler(sample_data)
    assert handler.filepath == sample_data
    assert isinstance(handler.df, pd.DataFrame)
    assert len(handler.columns) == 3


def test_get_preview(sample_data):
    handler = DataHandler(sample_data)
    preview = handler.get_preview(n_rows=5)
    assert isinstance(preview, pd.DataFrame)
    assert len(preview) == 5


def test_get_columns(sample_data):
    handler = DataHandler(sample_data)
    columns = handler.get_columns()
    assert "names" in columns
    assert "types" in columns
    assert len(columns["names"]) == 3


def test_prepare_data(sample_data):
    handler = DataHandler(sample_data)
    X_train, X_test, y_train, y_test = handler.prepare_data(
        features=["feature1", "feature2"],
        target="target",
        train_size=0.8,
        normalize=True,
    )
    assert len(X_train) > len(X_test)
    assert len(y_train) > len(y_test)
    assert X_train.shape[1] == 2
