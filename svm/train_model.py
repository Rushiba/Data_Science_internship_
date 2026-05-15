import pandas as pd
import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
df = pd.read_csv('Iris.csv')

# 2. Preprocessing
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

X = df.drop('Species', axis=1)
y = df['Species']

# 3. Encode the target labels (Iris-setosa -> 0, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Train the SVM Model
model = svm.SVC(kernel='linear', probability=True)
model.fit(X, y_encoded)

# 5. Save the model and label encoder for the app
joblib.dump(model, 'iris_svm_model.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("Model trained and saved successfully!")