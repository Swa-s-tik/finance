import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('BankChurners.csv')

# Define features and target
features = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 
            'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 
            'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
target = 'Attrition_Flag'

# Split the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor
numeric_features = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
                    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 
                    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# Function to make predictions
def predict_attrition(input_data):
    prediction = model.predict_proba(input_data)[0][1]
    return prediction

# Function to get feature importances
def get_feature_importances():
    feature_names = (numeric_features + 
                     model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names(categorical_features).tolist())
    importances = model.named_steps['classifier'].feature_importances_
    return dict(zip(feature_names, importances))

# Function to calculate financial health score
def calculate_financial_health(input_data):
    credit_limit = input_data['Credit_Limit'].values[0]
    total_revolving_bal = input_data['Total_Revolving_Bal'].values[0]
    total_trans_amt = input_data['Total_Trans_Amt'].values[0]
    avg_utilization_ratio = input_data['Avg_Utilization_Ratio'].values[0]
    
    utilization_score = max(0, 1 - avg_utilization_ratio)
    balance_to_limit_ratio = total_revolving_bal / credit_limit if credit_limit > 0 else 1
    balance_score = max(0, 1 - balance_to_limit_ratio)
    activity_score = min(total_trans_amt / 5000, 1)  # Assuming 5000 is a good activity level
    
    financial_health_score = (utilization_score * 0.4 + balance_score * 0.4 + activity_score * 0.2) * 100
    return financial_health_score