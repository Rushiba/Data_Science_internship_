import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the data
# Make sure insurance.csv is in this folder too!
df = pd.read_csv('insurance.csv')
print(df.columns) # This will print the names in your terminal

# 2. Basic Preprocessing (Matching your app's logic)
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})

X = df.drop('expenses', axis=1)
y = df['charges']

# 3. Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# 4. Save EXACTLY where Streamlit expects them
joblib.dump(model, 'insurance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ SUCCESS: 'insurance_model.pkl' and 'scaler.pkl' have been created!")
print("Now refresh your Streamlit app.")