# ------------------------------------------------------------------------------
# CELL 1: Load, Preprocess, and Split
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load Data
df = pd.read_csv('raw_diarrhea_data.csv')

# Minimalist Preprocessing
# 1. Target: < 48 hours is 'Recovered Fast' (1), else 0
df['target'] = (df['hours_to_resolution'] < 48).astype(int)

# 2. Impute MUAC with median
df['muac_cm'] = df['muac_cm'].fillna(df['muac_cm'].median())

# 3. Encode Categoricals (Simple approach)
le = LabelEncoder()
categorical_cols = ['gender', 'dehydration_grade', 'pathogen_identified', 'treatment_group']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# 4. Select Features (X) and Target (y)
features = ['age_months', 'weight_kg', 'muac_cm', 'dehydration_grade', 
            'pathogen_identified', 'azithro_resistance_detected', 'treatment_group']
X = df[features]
y = df['target']

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Ready. Features:", features)

# ------------------------------------------------------------------------------
# CELL 2: Data Visualization (Correlation Heatmap)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
# We combine X and y temporarily to see correlation with target
viz_df = X.copy()
viz_df['target'] = y

# Correlation Matrix
corr = viz_df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap (Identify redundancies)')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# CELL 3: Training Loop & Comparison
# ------------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []

print("\n" + "="*40)
print("      MODEL PERFORMANCE REPORT      ")
print("="*40)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Score
    acc = accuracy_score(y_test, preds)
    results.append({'Model': name, 'Accuracy': acc})
    
    print(f"--- {name} Report ---")
    print(classification_report(y_test, preds))

# ------------------------------------------------------------------------------
# CELL 4: Final Visualization (Comparison & Feature Importance)
# ------------------------------------------------------------------------------
# 1. Model Accuracy Comparison
results_df = pd.DataFrame(results)
plt.figure(figsize=(8, 4))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Model Comparison: Accuracy')
plt.ylim(0, 1.0)
plt.show()

# 2. Feature Importance (Using Random Forest)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances (Random Forest)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()