from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import joblib

# Load and Preprocess
df = pd.read_csv('employee_data.csv')

# Encoding categorical values
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_dept = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Education'] = le_edu.fit_transform(df['Education'])
df['Department'] = le_dept.fit_transform(df['Department'])

# Create Salary Category for Logistic Regression
# Low: < 60k, Medium: 60k-90k, High: > 90k
def classify_salary(val):
    if val < 60000: return 0 # Low
    elif val < 90000: return 1 # Medium
    else: return 2 # High

df['Salary_Cat'] = df['Salary'].apply(classify_salary)

# Features and Targets
X = df.drop(['Employee_ID', 'Salary', 'Salary_Cat'], axis=1)
y_reg = df['Salary']
y_clf = df['Salary_Cat']

# Split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_reg)

# 2. Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(X_train_scaled, y_train_reg)

# 3. Logistic Regression (Classification)
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train_scaled, y_train_clf)

# Save models and assets
joblib.dump(lin_reg, 'lin_model.pkl')
joblib.dump(rf_reg, 'rf_model.pkl')
joblib.dump(log_clf, 'log_model.pkl')
joblib.dump(scaler, 'scaler.pkl')