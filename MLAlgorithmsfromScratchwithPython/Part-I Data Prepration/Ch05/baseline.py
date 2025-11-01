import numpy as np
import random
from collections import Counter

# -------------------------
# Random Prediction Algorithm
# -------------------------
class RandomPrediction:
    def __init__(self, labels=None):
        self.labels = labels  # list of possible labels
    
    def fit(self, X, y):
        # Learn class labels from training set
        self.labels = np.unique(y)
    
    def predict(self, X):
        # Predict random label for each sample
        return np.array([random.choice(self.labels) for _ in range(len(X))])

# -------------------------
# Zero Rule Algorithm (ZeroR)
# -------------------------
class ZeroRule:
    def __init__(self, task="classification"):
        assert task in ["classification", "regression"], "Task must be 'classification' or 'regression'"
        self.task = task
        self.rule = None  # majority class or mean
    
    def fit(self, X, y):
        if self.task == "classification":
            # Majority class
            self.rule = Counter(y).most_common(1)[0][0]
        else:
            # Mean for regression
            self.rule = np.mean(y)
    
    def predict(self, X):
        return np.array([self.rule for _ in range(len(X))])

# -------------------------
# Test Cases
# -------------------------
if __name__ == "__main__":
    # Sample classification dataset
    X_class = np.array([[1], [2], [3], [4], [5], [6]])
    y_class = np.array([0, 1, 1, 0, 1, 1])  # imbalanced (majority = 1)
    
    print("=== Classification Task ===")
    # Random Prediction
    random_clf = RandomPrediction()
    random_clf.fit(X_class, y_class)
    print("Random Prediction:", random_clf.predict(X_class))
    
    # Zero Rule
    zeror_clf = ZeroRule(task="classification")
    zeror_clf.fit(X_class, y_class)
    print("ZeroR Prediction:", zeror_clf.predict(X_class))  # always predicts majority class
    
    # Sample regression dataset
    X_reg = np.array([[1], [2], [3], [4], [5]])
    y_reg = np.array([10.0, 12.0, 14.0, 13.0, 11.0])
    
    print("\n=== Regression Task ===")
    # Random Prediction (not meaningful for regression, but possible if discrete labels known)
    random_reg = RandomPrediction(labels=np.unique(y_reg))
    print("Random Prediction:", random_reg.predict(X_reg))
    
    # Zero Rule
    zeror_reg = ZeroRule(task="regression")
    zeror_reg.fit(X_reg, y_reg)
    print("ZeroR Prediction:", zeror_reg.predict(X_reg))  # always mean of y_reg
