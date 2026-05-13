import pandas as pd
import numpy as np

# 1. Define the number of employees
n_rows = 250 

# 2. Generate random but logical data
np.random.seed(42)
data = {
    'Employee_ID': range(1001, 1001 + n_rows),
    'Age': np.random.randint(22, 60, n_rows),
    'Gender': np.random.choice(['Male', 'Female'], n_rows),
    'Education': np.random.choice(['Bachelor', 'Master', 'PhD'], n_rows),
    'Department': np.random.choice(['IT', 'HR', 'Sales', 'Finance', 'Marketing'], n_rows),
    'Experience': np.random.randint(0, 35, n_rows),
    'Working_Hours': np.random.randint(35, 60, n_rows),
    'Performance_Score': np.random.randint(1, 11, n_rows),
    'Projects_Completed': np.random.randint(1, 20, n_rows)
}

df = pd.DataFrame(data)

# 3. Create a logic for Salary (Base + Experience + Performance)
# This ensures the ML model actually has a pattern to learn!
df['Salary'] = (30000 + (df['Experience'] * 2200) + 
                (df['Performance_Score'] * 1200) + 
                np.random.normal(0, 3000, n_rows))

# 4. Save the file
df.to_csv('employee_data.csv', index=False)

print("Success! 'employee_data.csv' has been created in your folder.")