# src/test_model.py

import pandas as pd

def test_accuracy_above_threshold():
    metrics = pd.read_csv('metrics.csv')
    accuracy = metrics.loc[metrics['metric'] == 'accuracy', 'value'].values[0]
    assert accuracy >= 0.85, f"Accuracy {accuracy:.2f} is below 85%"
