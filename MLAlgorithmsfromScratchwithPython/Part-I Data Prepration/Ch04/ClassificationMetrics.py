import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix values: TP, TN, FP, FN.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP + 1e-10)   # avoid div by zero


def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN + 1e-10)


def specificity(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TN / (TN + FP + 1e-10)


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-10)


def log_loss(y_true, y_prob):
    """
    Log Loss (Cross-Entropy).
    y_prob: predicted probabilities for class 1
    """
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)  # numerical stability
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def roc_curve(y_true, y_prob, thresholds=None):
    """
    Compute ROC curve (TPR vs. FPR) for different thresholds.
    Returns FPR, TPR.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    TPR, FPR = [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        TPR.append(TP / (TP + FN + 1e-10))  # Recall
        FPR.append(FP / (FP + TN + 1e-10))  # False Positive Rate

    return np.array(FPR), np.array(TPR), thresholds


def auc(fpr, tpr):
    """
    Compute AUC using trapezoidal rule.
    """
    return np.trapz(tpr, fpr)


import numpy as np

def confusion_matrix_multiclass(y_true, y_pred, labels=None):
    """
    Compute confusion matrix for multi-class classification.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm, labels


def accuracy_multiclass(y_true, y_pred):
    cm, _ = confusion_matrix_multiclass(y_true, y_pred)
    return np.trace(cm) / np.sum(cm)


def precision_recall_f1_multiclass(y_true, y_pred, average="macro"):
    """
    Compute precision, recall, F1 for multi-class.
    average = 'macro', 'micro', or 'weighted'
    """
    cm, labels = confusion_matrix_multiclass(y_true, y_pred)
    n_classes = len(labels)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    precision_per_class = TP / (TP + FP + 1e-10)
    recall_per_class    = TP / (TP + FN + 1e-10)
    f1_per_class        = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-10)

    if average == "macro":
        return np.mean(precision_per_class), np.mean(recall_per_class), np.mean(f1_per_class)
    elif average == "micro":
        TP_sum = np.sum(TP)
        FP_sum = np.sum(FP)
        FN_sum = np.sum(FN)
        precision = TP_sum / (TP_sum + FP_sum + 1e-10)
        recall    = TP_sum / (TP_sum + FN_sum + 1e-10)
        f1        = 2 * precision * recall / (precision + recall + 1e-10)
        return precision, recall, f1
    elif average == "weighted":
        weights = np.sum(cm, axis=1) / np.sum(cm)
        return (np.sum(weights * precision_per_class),
                np.sum(weights * recall_per_class),
                np.sum(weights * f1_per_class))
    else:
        raise ValueError("average must be 'macro', 'micro', or 'weighted'")

if __name__=='__main__':
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_curve, auc
    X, y_true = make_classification(n_samples=200, n_features=5, n_classes=2,
                                    n_informative=3, random_state=42)

    model = LogisticRegression()
    model.fit(X, y_true)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Binary Classification
    print("=== Binary Classification ===")
    print("Confusion Matrix (TP, TN, FP, FN):", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall (Sensitivity):", recall(y_true, y_pred))
    print("Specificity:", specificity(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

    print("Log Loss:", log_loss(y_true, y_prob))

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    print("AUC:", auc(fpr, tpr))

    # multi-class classification
    X_multi, y_true_multi = make_classification(n_samples=300, n_features=5, n_classes=3,
                                                n_informative=4, n_clusters_per_class=1,
                                                random_state=42)

    model_multi = LogisticRegression(multi_class="multinomial", max_iter=1000)
    model_multi.fit(X_multi, y_true_multi)

    y_pred_multi = model_multi.predict(X_multi)

    print("\n=== Multi-class Classification ===")
    cm, labels = confusion_matrix_multiclass(y_true_multi, y_pred_multi)
    print("Confusion Matrix:\n", cm)

    print("Accuracy:", accuracy_multiclass(y_true_multi, y_pred_multi))

    print("Macro Avg (Precision, Recall, F1):",
          precision_recall_f1_multiclass(y_true_multi, y_pred_multi, average="macro"))
    print("Micro Avg (Precision, Recall, F1):",
          precision_recall_f1_multiclass(y_true_multi, y_pred_multi, average="micro"))
    print("Weighted Avg (Precision, Recall, F1):",
          precision_recall_f1_multiclass(y_true_multi, y_pred_multi, average="weighted"))