import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("../data/boston_housing.csv")

# Select features and target
X = df[['chas', 'nox', 'rm', 'dis', 'ptratio']]  # Example feature set
y = df['medv']  # Target variable

# Function to preprocess the data (impute missing values and scale features)
def preprocess(X_train, X_test):
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_processed, X_test_processed = preprocess(X_train, X_test)

# Load the trained model
model = load("../models/linear_regression_model.pkl")

# Predict on the test set
y_pred_test = model.predict(X_test_processed)

# Evaluate the model
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print evaluation metrics
print(f"Test Mean Squared Error (MSE): {mse_test:.4f}")
print(f"Test R² Score: {r2_test:.4f}")
