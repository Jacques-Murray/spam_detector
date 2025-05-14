from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Handles loading and splitting the dataset.
    """
    def __init__(self,data_path:str, feature_names:List[str]=None, target_column:str='is_spam'):
        """
        Initializes the DataLoader.
        
        Args:
            data_path (str): Path to the CSV data file.
            feature_names (List[str], optional): List of column names.
                                                 If None, assumes no header in CSV.
            target_column (str): Name of the target variable column.
        """
        self.data_path = data_path
        self.feature_names = feature_names
        self.target_column = target_column

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            # Spambase dataset has no header. Names are in spambase.names.
            # For simplicity, we'll create generic feature names if not provided.
            # The last column is the target.
            df = pd.read_csv(self.data_path,header=None)

            if self.feature_names:
                if len(self.feature_names) + 1 == len(df.columns):
                    df.columns = self.feature_names + [self.target_column]
                else:
                    raise ValueError("Number of feature names does not match number of columns.")
            else:
                # Create generic feature names
                num_features = len(df.columns) - 1
                column_names = [f'feature_{i}' for i in range(num_features)] + [self.target_column]
                df.columns = column_names

            return df
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
        
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state:int=42) -> Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        """
        Splits data into training and testing sets.

        Args:
            df (pd.Dataframe): The dataframe to split.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before splitting.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                X_train, X_test, y_train, y_test
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty. Cannot split data.")
        
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,y,test_size=test_size,random_state=random_state,stratify=y
        )
        return X_train, X_test, y_train, y_test