from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model("neural_network_model.h5")

# Load preprocessing transformers (if needed)
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Define car features
FEATURES = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 'engine',
            'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from form
        input_data = [request.form.get(feature) for feature in FEATURES]

        # Convert numerical inputs
        input_data[2] = int(input_data[2])  # model_year
        input_data[3] = float(input_data[3])  # mileage
        input_data[9] = int(input_data[9])  # accident (binary)
        input_data[10] = int(input_data[10])  # clean_title (binary)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        # Preprocess input
        input_transformed = preprocessor.transform(input_df)

        # Make prediction (in USD)
        predicted_price_usd = model.predict(input_transformed)[0][0]

        # Convert USD to LKR
        exchange_rate = 280  # 1 USD = 280 LKR
        predicted_price_lkr = predicted_price_usd * exchange_rate
        total = predicted_price_lkr/280
        return render_template("index.html",
                               prediction_text=f"Estimated Car Price: LKR {total:,.2f}")

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
