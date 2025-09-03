# house_price_prediction.py
"""
🏡 House Price Prediction using Machine Learning

This project predicts house prices based on features like area, bedrooms, and location.

Steps:
1. Load dataset
2. Data preprocessing (encoding categorical variables)
3. Train-test split
4. Build Linear Regression model
5. Evaluate performance
"""

# =========================
# Import Libraries
# =========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =========================
# Load Dataset
# =========================
# Make sure you have 'house_data.csv' in the same folder
# Example columns: area, bedrooms, location, price
df = pd.read_csv('house_data.csv')

# =========================
# Data Preprocessing
# =========================
# Convert categorical columns into numerical (One-Hot Encoding)
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Features (X) and Target (y)
X = df.drop('price', axis=1)
y = df['price']

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# Evaluate Model
# =========================
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("✅ Model trained successfully!")
print("Mean Squared Error (MSE):", mse)

# =========================
# Example Prediction
# =========================
# Uncomment and edit values to test prediction
# sample_data = pd.DataFrame({
#     'area': [2000],
#     'bedrooms': [3],
#     'location_Bangalore': [1],
#     'location_Delhi': [0],
#     'location_Mumbai': [0]
# })
# print("Predicted Price:", model.predict(sample_data)[0])
