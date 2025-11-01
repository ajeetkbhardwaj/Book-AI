import random
from random import randrange
import numpy as np

def train_test_split(dataset, split=0.70)
    train = list()
    train_size = split * len(dataset)
    dataset_cp = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_cp))
        train.append(dataset_cp.pop(index))
    return train, dataset_cp

def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_cp = list(dataset)
    fold_size = int(len(dataset)/folds)

    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_cp))
            fold.append(dataset_cp.pop(index))
        dataset_split.append(fold)
    return dataset_split

def stratified_oversampling(X, y):
    """
    Stratified Oversampling to balance classes
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    X_resampled, y_resampled = [], []

    for c in classes:
        X_c = X[y==c]
        y_c = y[y==c]

        # rsampling with replacement
        indices = np.random.choice(len(X_c), size=max_count, replace=True)
        X_resampled.append(X_c[indices])
        y_resampled.append(y_c[indices])
    return np.vstack(X_resampled), np.hstack(y_resampled)

def stratified_undersampling(X, y):
    """
    Stratified Undersampling to balance classes
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.min(counts)

    X_resampled, y_resampled = [], []

    for c in classes:
        X_c = X[y==c]
        y_c = y[y==c]

        # resampling without replacement
        indices = np.random.choice(len(X_c), size=max_count, replace=False)
        X_resampled.append(X_c[indices])
        y_resampled.append(y_c[indices])
    return np.vstack(X_resampled), np.hstack(y_resampled)
