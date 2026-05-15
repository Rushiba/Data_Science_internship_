import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv('Iris.csv')
if 'Id' in df.columns: df = df.drop('Id', axis=1)

X = df.drop('Species', axis=1)
y = df['Species']

# Scaler is the secret to accurate predictions!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# RBF Kernel for perfect classification
model = SVC(kernel='rbf', probability=True)
model.fit(X_scaled, y_encoded)

# Save the 3 required files
joblib.dump(model, 'iris_model.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("✅ Success! Files created. You can now run the app.")