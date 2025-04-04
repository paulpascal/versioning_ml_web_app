import pytest
import numpy as np
from app.utils.model_handler import ModelHandler
import os
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def model_config():
    """Provide model configuration"""
    return {
        "features": ["feature1", "feature2", "feature3"],
        "model_type": "random_forest_classifier",
        "normalize": True,
        "target": "target",
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
    handler.model_type = "random_forest_classifier"
    handler.train(X_train, X_test, y_train, y_test)
    assert handler.model is not None
    assert handler.results is not None
    assert "train_accuracy" in handler.results
    assert "test_accuracy" in handler.results


def test_model_training_regression(sample_data, model_config):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)
    handler.model_type = "linear_regression"

    handler.train(X_train, X_test, y_train, y_test)
    assert handler.model is not None
    assert handler.results is not None
    assert "mse" in handler.results


def test_feature_importance(sample_data, model_config):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)
    handler.model_type = "random_forest_classifier"

    handler.train(X_train, X_test, y_train, y_test)
    assert "feature_importance" in handler.results
    assert len(handler.results["feature_importance"]) == len(model_config["features"])


def test_confusion_matrix(sample_data, model_config):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)
    handler.model_type = "random_forest_classifier"

    handler.train(X_train, X_test, y_train, y_test)
    assert "confusion_matrix" in handler.results
    assert len(handler.results["confusion_matrix"]) == 2
    assert len(handler.results["confusion_matrix"][0]) == 2


def test_model_saving(sample_data, model_config, tmp_path):
    X_train, X_test, y_train, y_test = sample_data
    handler = ModelHandler(**model_config)
    handler.model_type = "random_forest_classifier"

    handler.train(X_train, X_test, y_train, y_test)
    filepath = handler.save_model("test_model")
    assert os.path.exists(filepath)

    # Clean up
    if os.path.exists(filepath):
        os.remove(filepath)
