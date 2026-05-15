import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load the dataset
# Ensure insurance.csv is in the same folder as this script
df = pd.read_csv('insurance.csv')

# 2. Data Preprocessing
# Convert categorical text data into numbers 
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])       # male: 1, female: 0
df['smoker'] = le.fit_transform(df['smoker']) # yes: 1, no: 0
df['region'] = le.fit_transform(df['region']) # northeast: 0, northwest: 1, southeast: 2, southwest: 3

# 3. Exploratory Data Analysis (Visualization)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Split data into Features (X) and Target (y)
X = df.drop('expenses', axis=1) # All columns except expenses [cite: 1]
y = df['expenses']              # The target variable [cite: 1]

# 5. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scaling (Standardize the numbers for better accuracy)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Implement Supervised Learning (Linear Regression)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 8. Model Evaluation
predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Performance:")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"R2 Score (Accuracy): {r2:.2f}")

# 9. Save the model and scaler for the Streamlit app
joblib.dump(model, 'insurance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved successfully!")
