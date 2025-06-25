from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app) 

model = joblib.load("models/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = pd.DataFrame({
        "Rooms": [data["Rooms"]],
        "Bathroom": [data["Bathroom"]],
        "Landsize": [data["Landsize"]],
        "Distance": [data["Distance"]],
        "Car": [data["Car"]],
        "Type_h": [1 if data["Type"] == "h" else 0],
        "Type_u": [1 if data["Type"] == "u" else 0],
        "Type_t": [1 if data["Type"] == "t" else 0]
    })
    prediction = model.predict(input_data)[0]
    return jsonify({"price": round(prediction, 2)})

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "MAE": 285035.48, 
        "R²": 0.50,
        "model": "Árbol de Decisión"
    })

if __name__ == "__main__":
    app.run(debug=True)