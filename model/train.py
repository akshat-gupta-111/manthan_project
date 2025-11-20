import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

rf = RandomForestClassifier(n_estimators=10, max_depth=5)
rf.fit(X, y)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'model.pkl')

with open(file_path, 'wb') as f:
    pickle.dump(rf, f)

print(f"Model saved to {file_path}")