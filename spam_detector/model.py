import pandas as pd
from sklearn.linear_model import Log, LogisticRegression
from sklearn.naive_bayes import GaussianNB


class SpamClassifier:
    """
    Handles training and prediction with the classification model.
    """
    def __init__(self, model_type: str = 'logistic_regression', random_state: int = 42):
        """
        Initializes the classifier.
        
        Args:
            model_type (str): Type of model to use ('logistic_regression' or 'naive_bayes').
            random_state (int): Random state for model initialization.
        """
        if model_type =='logistic_regression':
            self.model = LogisticRegression(random_state=random_state,solver='liblinear')
        elif model_type == 'naive_bayes':
            self.model = GaussianNB()
        else:
            raise ValueError("Unsupported model type. Choose 'logistic_regression' or 'naive_bayes'.")
        self.model_type = model_type

    def train(self, X_train: pd.DataFrame, y_train: pd.Series)->None:
        """
        Trains the classification model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        self.model.fit(X_train, y_train)
        print(f"{self.model_type} model trained successfully.")

    def predict(self, X_test:pd.DataFrame)->pd.Series:
        """
        Makes predictions on the test data.
        
        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted labels.
        """
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test:pd.DataFrame)->pd.DataFrame:
        """
        Makes probability predictions on the test data.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: Predicted probabilities for each class.
        """
        return self.model.predict_proba(X_test)