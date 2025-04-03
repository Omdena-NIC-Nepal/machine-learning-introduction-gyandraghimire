#!/usr/bin/env python
# coding: utf-8

# ## Task 2: Data Preprocessing

# In[1]:


get_ipython().system('pip install scipy')
get_ipython().system('pip install scikit-learn')


# ### 1. Handle Missing Values and Outliers

# #### Check for missing values in the dataset

# In[2]:


import pandas as pd

# Load the dataset
df = pd.read_csv("../data/boston_housing.csv")

# Check for missing values
print(df.isnull().sum())


# ##### Distribution of 'rm' (Average Number of Rooms per Dwelling)

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate skewness for the 'rm' feature
rm_skewness = df['rm'].skew()

# Display the skewness value
print(f"Skewness of 'rm': {rm_skewness}")

# Plot the distribution of the 'rm' feature
sns.histplot(df['rm'], kde=True, color='skyblue', bins=20)
plt.title("Distribution of 'rm' (Average Number of Rooms per Dwelling)")
plt.xlabel('rm')
plt.ylabel('Frequency')
plt.show()


# ##### Impute Missing Values

# In[4]:


# Impute missing values in 'rm' column with the median
df['rm'] = df['rm'].fillna(df['rm'].median())

# Verify that missing values are handled
print(df.isnull().sum())


# #### Handling Outliers

# ##### Detect outliers using the Interquartile Range (IQR) method

# In[5]:


import numpy as np

# Compute Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1  # Interquartile Range

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (df < lower_bound) | (df > upper_bound)
print(outliers.sum())  # Count of outliers per column


# ##### Square Root Transformation

# In[6]:


# Apply square root transformation
df_sqrt_transformed = df.copy()
df_sqrt_transformed = np.sqrt(df_sqrt_transformed)

# Check the transformed data
print(df_sqrt_transformed.head())


# ### 2. Encode Categorical Variables

# ##### One-Hot Encoding

# In[7]:


import pandas as pd

# Load the dataset
df = pd.read_csv('../data/boston_housing.csv')

# Strip spaces in column names, if any
df.columns = df.columns.str.strip()

# One-Hot Encode the 'chas' column
df = pd.get_dummies(df, columns=['chas'], drop_first=True)

# Check the first few rows to verify the transformation
print(df.head())


# ### 3. Normalize/Standardize Numerical Features

# ##### Standardization (Z-score Scaling)

# In[7]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('../data/boston_housing.csv')

# Strip spaces from column names (if any)
df.columns = df.columns.str.strip()

# One-Hot Encode the 'chas' column
df = pd.get_dummies(df, columns=['chas'], drop_first=True)

# Define the list of numerical features (excluding target variable 'medv')
num_features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical features
df[num_features] = scaler.fit_transform(df[num_features])

# Check the transformed dataset
print(df.head())


# ### 4. Split the Data into Training and Testing Sets

# In[8]:


from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
X = df.drop(columns=['medv', 'chas_1'])  # Features (exclude 'medv' and the 'chas_1' target column)
y = df['medv']  # Target variable (housing prices)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the dimensions of the split data
print("Training set X shape:", X_train.shape)
print("Testing set X shape:", X_test.shape)
print("Training set y shape:", y_train.shape)
print("Testing set y shape:", y_test.shape)

