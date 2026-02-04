from flask import Flask, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "decision_tree_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully")

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Predict page (GET)
@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# Result page (POST)
@app.route("/result", methods=["POST"])
def result():
    # TEMP: dummy input (we’ll connect form properly later)
    sample = np.zeros((1, model.n_features_in_))
    prediction = model.predict(sample)[0]

    result_text = "Fraud Transaction" if prediction == 1 else "Not a Fraud Transaction"

    return render_template("submit.html", prediction=result_text)

if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)

