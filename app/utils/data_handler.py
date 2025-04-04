import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self._load_data()
        self.columns = self.df.columns.tolist()
        self.dtypes = self.df.dtypes.to_dict()

    def _load_data(self):
        """Load data based on file extension"""
        if self.filepath.endswith(".csv"):
            return pd.read_csv(self.filepath)
        elif self.filepath.endswith((".xlsx", ".xls")):
            return pd.read_excel(self.filepath)
        else:
            raise ValueError("Unsupported file format")

    def get_preview(self, n_rows=5):
        """Get preview of the data"""
        return self.df.head(n_rows)

    def get_columns(self):
        """Get column information"""
        return {
            "names": self.columns,
            "types": {col: str(dtype) for col, dtype in self.dtypes.items()},
        }

    def get_column_stats(self):
        """Get basic statistics for each column"""
        stats = {}
        for col in self.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stats[col] = {
                    "type": "numeric",
                    "mean": float(self.df[col].mean()),
                    "std": float(self.df[col].std()),
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                }
            else:
                stats[col] = {
                    "type": "categorical",
                    "unique_values": int(self.df[col].nunique()),
                    "missing_values": int(self.df[col].isna().sum()),
                }
        return stats

    def prepare_data(self, features, target, train_size=0.8, normalize=True):
        """Prepare data for training"""
        X = self.df[features]
        y = self.df[target]

        # Handle categorical variables
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]):
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42
        )

        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
