import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    learning_curve,
    cross_val_score,
    StratifiedKFold,
    KFold,
)
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime
import subprocess
from dotenv import load_dotenv
from scripts.dvc_setup import add_and_push_model
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


class ModelHandler:
    def __init__(self, model_type, features, target, train_size=0.8, normalize=True):
        self.model_type = model_type
        self.features = features
        self.target = target
        self.train_size = float(train_size)  # Convert to float
        self.normalize = normalize
        self.model = None
        self.scaler = StandardScaler() if normalize else None
        self.results = {}
        self.learning_curve_data = None
        self.feature_importance = None
        self.confusion_matrix = None
        load_dotenv()

    def create_model(self):
        """Create the appropriate model based on type"""
        if self.model_type == "svm":
            self.model = SVC(kernel="rbf", probability=True)
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "lr":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X_train, X_test, y_train, y_test):
        """Train the model with the given data"""
        try:
            # Normalize data if requested
            if self.normalize:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)

            # Create and train model based on type
            if self.model_type == "linear_regression":
                self.model = LinearRegression()
                is_classifier = False
            elif self.model_type == "random_forest_classifier":
                # Adjusted parameters to prevent overfitting
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,  # Limit tree depth
                    min_samples_split=5,  # Require more samples to split
                    min_samples_leaf=2,  # Require more samples in leaves
                    random_state=42,
                )
                is_classifier = True
            elif self.model_type == "svm":
                # Adjusted parameters for better generalization
                self.model = SVC(
                    kernel="rbf",
                    C=1.0,  # Reduced from default to prevent overfitting
                    probability=True,
                    random_state=42,
                )
                is_classifier = True
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Perform cross-validation based on model type
            if is_classifier:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    self.model, X_train, y_train, cv=cv, scoring="accuracy"
                )
            else:
                # For regression, use KFold and R² scoring
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    self.model, X_train, y_train, cv=cv, scoring="r2"
                )

            # Train the model
            self.model.fit(X_train, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test)
            y_train_pred = self.model.predict(X_train)

            # Calculate metrics based on model type
            if is_classifier:
                self.results = {
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted"),
                    "recall": recall_score(y_test, y_pred, average="weighted"),
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                    "cv_scores": {
                        "mean": float(cv_scores.mean()),
                        "std": float(cv_scores.std()),
                        "scores": cv_scores.tolist(),
                    },
                }

                # Check for perfect scores and add warning if needed
                if (
                    not self.normalize
                    and self.results["train_accuracy"] == 1.0
                    and self.results["test_accuracy"] == 1.0
                    and self.results["precision"] == 1.0
                    and self.results["recall"] == 1.0
                    and cv_scores.mean() == 1.0
                ):
                    self.results["warning"] = {
                        "type": "perfect_scores",
                        "message": "Perfect scores detected! This might indicate that normalization is needed. Consider enabling data normalization and retraining the model for more reliable results.",
                    }

                # Add feature importance for supported classifiers
                if isinstance(self.model, RandomForestClassifier):
                    # For Random Forest, use built-in feature importance
                    feature_importance = [
                        {"feature": feature, "importance": float(importance)}
                        for feature, importance in zip(
                            self.features, self.model.feature_importances_
                        )
                    ]
                    self.results["feature_importance"] = sorted(
                        feature_importance, key=lambda x: x["importance"], reverse=True
                    )
                elif isinstance(self.model, SVC):
                    if self.model.kernel == "linear":
                        feature_importance = [
                            {"feature": feature, "importance": float(abs(coef))}
                            for feature, coef in zip(self.features, self.model.coef_[0])
                        ]
                        self.results["feature_importance"] = sorted(
                            feature_importance,
                            key=lambda x: x["importance"],
                            reverse=True,
                        )
                    else:
                        # For non-linear kernels, use permutation importance
                        r = permutation_importance(
                            self.model, X_test, y_test, n_repeats=10, random_state=42
                        )
                        feature_importance = [
                            {"feature": feature, "importance": float(imp)}
                            for feature, imp in zip(self.features, r.importances_mean)
                        ]
                        self.results["feature_importance"] = sorted(
                            feature_importance,
                            key=lambda x: x["importance"],
                            reverse=True,
                        )
            else:
                # Calculate predictions for both train and test sets
                self.results = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "r2": r2_score(y_test, y_pred),
                    "train_mse": mean_squared_error(y_train, y_train_pred),
                    "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    "train_r2": r2_score(y_train, y_train_pred),
                    "cv_scores": {
                        "mean": float(cv_scores.mean()),
                        "std": float(cv_scores.std()),
                        "scores": cv_scores.tolist(),
                    },
                }

                # For linear regression, use absolute coefficients as importance
                if isinstance(self.model, LinearRegression):
                    feature_importance = [
                        {"feature": feature, "importance": float(abs(coef))}
                        for feature, coef in zip(self.features, self.model.coef_)
                    ]
                    self.results["feature_importance"] = sorted(
                        feature_importance, key=lambda x: x["importance"], reverse=True
                    )

            return True

        except Exception as e:
            print(f"Error during training: {str(e)}")
            self.results = {"error": str(e)}
            return False

    def _create_confusion_matrix(self, y_true, y_pred):
        """Create confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400,
        )

        return fig.to_dict()

    def save_model(self, name):
        """Save the trained model"""
        if not self.model:
            raise ValueError("No trained model to save")

        # Create models directory if it doesn't exist
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
        )
        os.makedirs(models_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.joblib"
        filepath = os.path.join(models_dir, filename)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "features": self.features,
            "target": self.target,
            "model_type": self.model_type,
            "normalize": self.normalize,
            "results": self.results,
            "timestamp": timestamp,
        }

        joblib.dump(model_data, filepath)

        # Version the model with DVC
        try:
            add_and_push_model(filepath)
            print(f"Model {filename} saved and versioned successfully!")
        except Exception as e:
            print(f"Warning: Failed to version model with DVC: {str(e)}")
            print("Model was saved locally but not versioned.")

        return filepath

    def get_results(self):
        """Get the training results"""
        return self.results

    def get_default_name(self):
        """Generate a default name based on model type"""
        model_name_map = {
            "linear_regression": "lr",
            "random_forest_classifier": "rf",
            "svm": "svm",
        }
        return f"{model_name_map.get(self.model_type, 'model')}_"
