# test_model_training.py

import pandas as pd
from model_training.train_model import train_model
from model_training.evaluate_model import evaluate_model

def test_train_model():
    # Create a sample dataframe
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40]
    })
    target = pd.Series([0, 1, 0, 1])
    
    # Train the model
    model, X_test, y_test = train_model(data, target)
    
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
    model, X_test, y_test = train_model(data, target)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Check if the metrics are calculated
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
