"""
Task-01: House Price Prediction using Linear Regression
---------------------------------------------------------
Predicts house prices based on square footage, number of bedrooms,
and number of bathrooms.

Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
(Place train.csv in the same folder as this script)

Author: Sahu ji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv("train.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ------------------------------------------------------------------
# 2. Select relevant features
# ------------------------------------------------------------------
# GrLivArea  -> above ground living area (square footage)
# BedroomAbvGr -> number of bedrooms
# FullBath + HalfBath -> number of bathrooms
df["Bathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]

features = ["GrLivArea", "BedroomAbvGr", "Bathrooms"]
target = "SalePrice"

data = df[features + [target]].dropna()

print("\nRows after dropping missing values:", data.shape[0])

# ------------------------------------------------------------------
# 3. Quick EDA
# ------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between features and SalePrice")
plt.tight_layout()
plt.savefig("task01_correlation.png")
plt.close()

# ------------------------------------------------------------------
# 4. Train / test split
# ------------------------------------------------------------------
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 5. Train model
# ------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 6. Evaluate
# ------------------------------------------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE : {rmse:,.2f}")
print(f"MAE  : {mae:,.2f}")
print(f"R2   : {r2:.4f}")

print("\nModel Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat}: {coef:,.2f}")
print(f"Intercept: {model.intercept_:,.2f}")

# ------------------------------------------------------------------
# 7. Visualize predictions vs actual
# ------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         "r--", linewidth=2)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("task01_predictions.png")
plt.close()

# ------------------------------------------------------------------
# 8. Example prediction
# ------------------------------------------------------------------
sample = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "Bathrooms": [2.5]
})
predicted_price = model.predict(sample)[0]
print(f"\nExample: A 2000 sqft house with 3 bed / 2.5 bath -> "
      f"predicted price: ${predicted_price:,.2f}")
