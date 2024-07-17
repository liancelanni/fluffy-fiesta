# test_model_training.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pandas as pd
import model_training
import model_evaluation

def test_train_model():
    # Create a sample dataframe
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40]
    })
    target = pd.Series([0, 1, 0, 1])
    
    # Train the model
    model, X_train, X_val, y_train, y_val = model_training.train_model(True, data, target)
    
    # Check if the model is trained
    assert model is not None
    assert len(X_test) > 0
    assert len(y_test) > 0

def test_evaluate_model():
    # Create a sample dataframe
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40]
    })
    target = pd.Series([0, 1, 0, 1])
    
    # Train the model
    model, X_train, X_val, y_train, y_val = model_training.train_model(True, data, target)
    
    # Evaluate the model
    metrics = model_evaluation.eval_model(model, X_train, X_val, y_train, y_val)
    
    # Check if the metrics are calculated
    assert 'train_roc_auc' in metrics
    assert 'train_precision' in metrics
    assert 'train_recall' in metrics
    assert 'train_f1_score' in metrics
