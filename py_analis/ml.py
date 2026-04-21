import polars as pl
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


outlier_ids = [1299, 524, 692, 1183]

traind_db = pl.read_csv("train.csv", infer_schema=10000, null_values=["NA"], try_parse_dates=True)
result = traind_db.select(
    pl.col("Id"),
    pl.col("YearRemodAdd"),
    pl.col("GrLivArea"),
    pl.col("TotRmsAbvGrd"),
    pl.col("GarageArea"),
    pl.col("TotalBsmtSF"),
    pl.col("OverallQual"),
    pl.col("SalePrice"),
    pl.col("Neighborhood"),
    pl.col("OverallCond"),
    pl.col("GarageCars"),
    pl.col("MSZoning")
).filter(~pl.col("Id").is_in(outlier_ids))

result_to_encode = result.select(
    pl.col("MSZoning"),
    pl.col("Neighborhood")

).to_numpy()

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_numeric = result.drop(["Neighborhood","MSZoning","Id","SalePrice"]).to_numpy()
X_encoded = encoder.fit_transform(result_to_encode)
X = np.hstack([X_numeric, X_encoded])
y = result["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regr = RandomForestRegressor(random_state=42)
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print(regr.score(X_test, y_test))
print(root_mean_squared_error(y_pred, y_test))

numeric_cols = ["YearRemodAdd", "GrLivArea", "TotRmsAbvGrd", "GarageArea",
                "TotalBsmtSF", "OverallQual", "OverallCond", "GarageCars"]

encoded_cols = list(encoder.get_feature_names_out(["MSZoning", "Neighborhood"]))
all_cols = numeric_cols + encoded_cols

feature_importances = regr.feature_importances_

importances_df = pd.DataFrame({
    "Feature": all_cols,
    "Importance": feature_importances
}).sort_values("Importance", ascending=False)

print("\nВажность признаков:")
print(importances_df.to_string(index=False))


plt.figure(figsize=(10, 6))
top_features = importances_df.head(15)
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()


joblib.dump(encoder, "D:/projects/house-price/fast_api/encoder.pkl")
joblib.dump(regr, "D:/projects/house-price/fast_api/regr.pkl")

