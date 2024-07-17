# src/model_evaluation.py
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, f1_score, precision_score, recall_score, 
    roc_auc_score, log_loss
)

def eval_model(model, X_train, X_test, y_train, y_test):
    # Predictions
    train_class_preds = model.predict(X_train)
    test_class_preds = model.predict(X_test)
    train_prob_preds = model.predict_proba(X_train)[:, 1]
    test_prob_preds = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "train_roc_auc": round(roc_auc_score(y_train, train_prob_preds),3),
        "test_roc_auc": round(roc_auc_score(y_test, test_prob_preds),3),
        "train_log_loss": round(log_loss(y_train, train_prob_preds),3),
        "test_log_loss": round(log_loss(y_test, test_prob_preds),3),
        "train_f1_score": round(f1_score(y_train, train_class_preds),3),
        "test_f1_score": round(f1_score(y_test, test_class_preds),3),
        "train_precision": round(precision_score(y_train, train_class_preds),3),
        "test_precision": round(precision_score(y_test, test_class_preds),3),
        "train_recall": round(recall_score(y_train, train_class_preds),3),
        "test_recall": round(recall_score(y_test, test_class_preds),3),
        "train_kappa_score": round(cohen_kappa_score(train_class_preds, y_train, weights='quadratic'), 3),
        "test_kappa_score": round(cohen_kappa_score(test_class_preds, y_test, weights='quadratic'), 3)
    }

    # Print performance
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Count the number of claims
    count_ones = np.sum(test_class_preds == 1.0)
    print(f"Number of claim predictions: {count_ones}")

    return metrics