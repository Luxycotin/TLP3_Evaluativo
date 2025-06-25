import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


data = pd.read_csv("models/melbourne_housing.csv")
data = data.dropna(subset=["Price", "Rooms", "Bathroom", "Landsize", "Distance", "Car", "Type"])


data = pd.get_dummies(data, columns=["Type"])


features = data[["Rooms", "Bathroom", "Landsize", "Distance", "Car", "Type_h", "Type_u", "Type_t"]]
target = data["Price"]


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")  
print(f"R²: {r2_score(y_test, y_pred):.2f}")

joblib.dump(model, "models/model.pkl")
print("Modelo guardado en 'models/model.pkl'")