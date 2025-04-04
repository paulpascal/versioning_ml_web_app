import pytest
import numpy as np
from app.utils.model_handler import ModelHandler
import os


@pytest.fixture
def sample_data():
    # Create sample training data
    X_train = np.random.rand(100, 3)
    X_test = np.random.rand(20, 3)
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 20)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def model_config():
    return {
        "model_type": "rf",
        "features": ["feature1", "feature2", "feature3"],
        "target": "target",
        "train_size": 0.8,
        "normalize": True,
    }


def test_model_creation(model_config):
    handler = ModelHandler(**model_config)
    assert handler.model_type == model_config["model_type"]
    assert handler.features == model_config["features"]
    assert handler.target == model_config["target"]
    assert handler.model is None  # Model should not be created until training


def test_model_training_classification(sample_data, model_config):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)

    # Test Random Forest
    handler.model_type = "rf"
    handler.train(X_train, X_test, y_train, y_test)
    assert handler.model is not None
    assert "accuracy" in handler.metrics
    assert "precision" in handler.metrics
    assert "recall" in handler.metrics
    assert "f1" in handler.metrics

    # Test SVM
    handler.model_type = "svm"
    handler.train(X_train, X_test, y_train, y_test)
    assert handler.model is not None
    assert "accuracy" in handler.metrics
    assert "precision" in handler.metrics
    assert "recall" in handler.metrics
    assert "f1" in handler.metrics


def test_model_training_regression(sample_data, model_config):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)
    handler.model_type = "lr"

    handler.train(X_train, X_test, y_train, y_test)
    assert handler.model is not None
    assert "mse" in handler.metrics
    assert "rmse" in handler.metrics
    assert "r2" in handler.metrics


def test_feature_importance(sample_data, model_config):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)
    handler.model_type = "rf"

    handler.train(X_train, X_test, y_train, y_test)
    assert handler.feature_importance is not None
    assert len(handler.feature_importance) == len(model_config["features"])


def test_confusion_matrix(sample_data, model_config):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)
    handler.model_type = "rf"

    handler.train(X_train, X_test, y_train, y_test)
    assert handler.confusion_matrix is not None


def test_model_saving(sample_data, model_config, tmp_path):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)
    handler.model_type = "rf"

    handler.train(X_train, X_test, y_train, y_test)
    filepath = handler.save_model("test_model")

    assert os.path.exists(filepath)
    assert filepath.endswith(".joblib")
