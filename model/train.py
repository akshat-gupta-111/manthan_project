import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/Users/akshatgupta111/Documents/manthan_project/data/data_final.csv')



df['muac_cm'] = df['muac_cm'].fillna(df['muac_cm'].median())

df['target'] = (df['hours_to_resolution'] < 48).astype(int)

encoders_dict = {} 
categorical_cols = ['dehydration_grade', 'pathogen_identified', 'treatment_group']

for col in categorical_cols:
    
    df[col] = df[col].fillna('None')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders_dict[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}



features = ['age_months', 'weight_kg', 'muac_cm', 'dehydration_grade', 
            'pathogen_identified', 'azithro_resistance_detected', 'treatment_group']

X = df[features]
y = df['target']

model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X, y)



with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders_dict, f)

print("Success! 'model.pkl' and 'encoders.pkl' are ready for your app.")