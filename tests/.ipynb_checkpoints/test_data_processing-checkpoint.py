# test_data_processing.py

import pandas as pd
from data_processing.clean_data import clean_data
from data_processing.preprocess_data import preprocess_data

def test_clean_data():
    # Create a sample dataframe
    data = pd.DataFrame({
        'feature': [1, 2, None, 4, 1000],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Clean the data
    cleaned_data = clean_data(data)
    
    # Check if missing values are handled
    assert cleaned_data.isnull().sum().sum() == 0
    
    # Check if outliers are removed
    assert cleaned_data['feature'].max() < 1000

def test_preprocess_data():
    # Create a sample dataframe
    data = pd.DataFrame({
        'num_feature1': [1, 2, 3, 4],
        'num_feature2': [10, 20, 30, 40],
        'cat_feature1': ['A', 'B', 'A', 'B'],
        'cat_feature2': ['X', 'Y', 'X', 'Y']
    })
    
    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    
    # Check if the data is transformed correctly
    assert preprocessed_data.shape[1] > data.shape[1]  # Assuming one-hot encoding increases the number of columns
