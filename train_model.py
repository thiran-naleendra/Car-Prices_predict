import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# ✅ Load the trained model
model = load_model("neural_network_model.h5")

# ✅ Load the preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# ✅ Define car features
FEATURES = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 'engine',
            'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']

# 🎨 GUI Design
root = tk.Tk()
root.title("Used Car Price Estimator")
root.geometry("400x550")
root.configure(bg="lightblue")

tk.Label(root, text="Enter Car Details", font=("Arial", 14, "bold"), bg="lightblue").pack(pady=10)

# Entry fields
entries = {}
for feature in FEATURES:
    frame = tk.Frame(root, bg="lightblue")
    frame.pack(pady=5)
    tk.Label(frame, text=feature.replace("_", " ").title() + ":", font=("Arial", 10), bg="lightblue").pack(side="left")
    entry = tk.Entry(frame, width=30)
    entry.pack(side="right")
    entries[feature] = entry

# Function to predict car price
def predict_price():
    try:
        # Get user input
        input_data = [entries[feature].get() for feature in FEATURES]

        # Convert numerical inputs
        input_data[2] = int(input_data[2])  # model_year
        input_data[3] = float(input_data[3])  # mileage
        input_data[9] = int(input_data[9])  # accident (binary)
        input_data[10] = int(input_data[10])  # clean_title (binary)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        # Preprocess input
        input_transformed = preprocessor.transform(input_df)

        # Predict price
        predicted_price = model.predict(input_transformed)[0][0]

        # Show result
        messagebox.showinfo("Prediction Result", f"Estimated Car Price: ${predicted_price:,.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Predict button
tk.Button(root, text="Estimate Price", font=("Arial", 12), bg="green", fg="white", command=predict_price).pack(pady=20)

# Run the application
root.mainloop()
