import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
train_df = pd.read_csv(r'E:/Minor Project/LoanDataset/train_u6lujuX_CVtuZ9i.csv')
test_df = pd.read_csv(r'E:/Minor Project/LoanDataset/test_Y3wMUE5_7gLdaTN.csv')

# Define replacement values for each column with NaN values
replacement_values = {
    'Loan_ID': 'N/A',
    'Gender': 'Unknown',
    'Married': 'Unknown',
    'Dependents': '0',
    'Education': 'Unknown',
    'Self_Employed': 'Unknown',
    'ApplicantIncome': 0,
    'CoapplicantIncome': 0,
    'LoanAmount': 0,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Unknown'
}

# Preprocess the training data
train_df['Loan_Status'] = train_df['Loan_Status'].map({'Y': 1, 'N': 0})
train_df['Dependents'] = train_df['Dependents'].replace('3+', 4)
train_df.fillna(replacement_values, inplace=True)
train_df.to_csv(r'E:/Minor Project/LoanDataset/filled_dataset.csv', index=False)

# Preprocess the test data
test_df['Dependents'] = test_df['Dependents'].replace('3+', 4)
test_df.fillna(replacement_values, inplace=True)
test_df.to_csv(r'E:/Minor Project/LoanDataset/filled_dataset2.csv', index=False)

# Load the processed datasets
train_data = pd.read_csv(r'E:/Minor Project/LoanDataset/filled_dataset.csv')
test_data = pd.read_csv(r'E:/Minor Project/LoanDataset/filled_dataset2.csv')

# Feature Engineering
train_data['TotalIncome'] = train_data['ApplicantIncome'] + train_data['CoapplicantIncome']
test_data['TotalIncome'] = test_data['ApplicantIncome'] + test_data['CoapplicantIncome']

train_data['EMI'] = train_data['LoanAmount'] / train_data['Loan_Amount_Term']
test_data['EMI'] = test_data['LoanAmount'] / test_data['Loan_Amount_Term']

train_data['Dependents'] = train_data['Dependents'].astype(int)
test_data['Dependents'] = test_data['Dependents'].astype(int)

# Encode categorical variables using Label Encoding
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
label_encoder = LabelEncoder()
for col in categorical_cols:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col] = label_encoder.transform(test_data[col])

# Separate features (X) and target (y)
features = ['Credit_History', 'Dependents', 'TotalIncome', 'EMI'] + categorical_cols
X_train = train_data[features]
y_train = train_data['LoanAmount']

X_test = test_data[features]
y_test = test_data['LoanAmount']

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions with Random Forest Regressor
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest Regressor model
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Random Forest Mean Squared Error: {mse_rf}')

# Calculate deviation for Random Forest Regressor
deviation_rf = y_test - y_pred_rf

# Print the deviation for Random Forest Regressor
print("Deviation (Random Forest Regressor):")
for actual, predicted, dev in zip(y_test, y_pred_rf, deviation_rf):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f} | Deviation: {dev:.2f}")
