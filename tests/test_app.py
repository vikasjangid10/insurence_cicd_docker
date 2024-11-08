import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Import the main application code (adjust path if necessary)
from src.app import df, x, y

def test_data_loading():
    # Test if data is loaded correctly
    assert not df.empty, "DataFrame should not be empty"
    assert "charges" in df.columns, "Target column 'charges' should be in the DataFrame"

def test_data_types():
    # Check if columns are of correct data types
    assert df["charges"].dtype == float, "Target column 'charges' should be float"

def test_train_test_split():
    # Test if train/test split works
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    assert len(x_train) > 0 and len(x_test) > 0, "Train/test split should not be empty"

def test_model_training():
    # Test model training and prediction accuracy
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)
    assert score > 0, "Model's R2 score should be greater than 0"
