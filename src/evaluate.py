# src/evaluate.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import numpy as np
import pandas as pd

# Load model and test data
model = joblib.load('model.joblib')
data = np.load('test_data.npz')
X_test, y_test = data['X_test'], data['y_test']

# Predict
y_pred = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Save to CSV
metrics = pd.DataFrame({
    'metric': ['accuracy', 'precision', 'recall', 'f1'],
    'value': [accuracy, precision, recall, f1]
})
metrics.to_csv('metrics.csv', index=False)

print(metrics)
