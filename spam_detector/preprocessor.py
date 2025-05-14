import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Handles preprocessing of the data.
    Currently implements feature scaling.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train: pd.DataFrame)->pd.DataFrame:
        """
        Fits the scaler on the training data and transforms it.

        Args:
            X_train (pd.DataFrame): Training features.

        Returns:
            pd.DataFrame: Scaled training features.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    def transform(self, X_test: pd.DataFrame)->pd.DataFrame:
        """
        Transforms the test data using the fitted scaler.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: Scaled test features.
        """
        X_test_scaled = self.scaler.transform(X_test)
        return pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)