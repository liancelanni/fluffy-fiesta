# test_data_processing.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pandas as pd
import pytest
import data_preprocessing

def test_data_cleaning_missing_values():
    # Test case for handling missing values
    input_data = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})
    cleaned_data = data_preprocessing.clean_data(input_data)
    expected_data = pd.DataFrame({'A': [1, 2, 0, 4], 'B': [5, 0, 7, 8]})
    assert cleaned_data.equals(expected_data)

def test_data_cleaning_outliers():
    # Test case for outlier removal
    input_data = pd.DataFrame({'A': [1, 2, 100, 4], 'B': [5, 10, 7, 8]})
    cleaned_data = data_preprocessing.clean_data(input_data)
    expected_data = pd.DataFrame({'A': [1, 2, 4, 4], 'B': [5, 10, 7, 8]})
    assert cleaned_data.equals(expected_data)

def test_data_transformation_scaling():
    # Test case for scaling numerical features
    input_data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
    transformed_data = data_preprocessing.transform_data(input_data)
    assert transformed_data['A'].mean() == 0 and transformed_data['B'].mean() == 0
