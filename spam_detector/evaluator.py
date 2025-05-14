import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """
    Handles evaluation of the model's performance.
    """
    def __init__(self, y_true:pd.Series, y_pred:pd.Series):
        """
        Initializes the ModelEvaluator.

        Args:
            y_true (pd.Series): True labels.
            y_pred (pd.Series): Predicted labels.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def get_accuracy(self)->float:
        """
        Calculates accuracy.
        """
        return accuracy_score(self.y_true,self.y_pred)
    
    def get_precision(self)->float:
        """
        Calculates precision.
        """
        return precision_score(self.y_true,self.y_pred,zero_division=0)
    
    def get_recall(self)->float:
        """
        Calculates recall.
        """
        return recall_score(self.y_true,self.y_pred,zero_division=0)
    
    def get_f1_score(self)->float:
        """
        Calculates F1 score.
        """
        return f1_score(self.y_true,self.y_pred,zero_division=0)
    
    def get_confusion_matrix(self):
        """
        Calculates the confusion matrix.
        """
        return confusion_matrix(self.y_true,self.y_pred)
    
    def print_evaluation_summary(self):
        """
        Prints a summary of the evaluation metrics.
        """
        print("\nModel Evaluation:")
        print(f"  Accuracy: {self.get_accuracy():.4f}")
        print(f"  Precision: {self.get_precision():.4f}")
        print(f"  Recall: {self.get_recall():.4f}")
        print(f"  F1 Score: {self.get_f1_score():.4f}")

        print("\nConfusion Matrix:")
        cm = self.get_confusion_matrix()
        print(cm)

        # Optional: Plot confusion matrix
        try:
            plt.figure(figsize=(6,4))
            sns.heatmap(cm,annot=True,fmt="d",cmap="Blue",
                        xticklabels=["Not Spam","Spam"],
                        yticklabels=["Not Spam","Spam"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.show()
        except Exception as e:
            print(f"Could not plot confusion matrix. Matplotlib or Seaborn might have issues: {e}")
            print("Ensure you have a GUI backend if running locally for plots to show.")