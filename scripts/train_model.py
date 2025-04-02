import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load dataset
df = pd.read_csv("../data/boston_housing.csv")

# Select features and target
X = df[['chas', 'nox', 'rm', 'dis', 'ptratio']]  # Features based on RFE
y = df['medv']  # Target variable

# Handle missing values by imputing with the median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
y_imputed = imputer.fit_transform(y.values.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the training and testing data
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate performance using RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Print evaluation results
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")

# Save the model to the 'models' directory
models_dir = "../models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(model, os.path.join(models_dir, "linear_regression_model.pkl"))

print("Model trained and saved successfully!")
