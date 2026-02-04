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

# Predict page
@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# Result page
@app.route("/result", methods=["POST"])
def result():
    try:
        # 1️⃣ Read form values
        step = float(request.form["step"])
        txn_type = request.form["type"]
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        # 2️⃣ System-generated feature (NOT from user)
        isFlaggedFraud = 0

        # 3️⃣ One-hot encode transaction type
        type_cash_out = 1 if txn_type == "CASH_OUT" else 0
        type_debit = 1 if txn_type == "DEBIT" else 0
        type_payment = 1 if txn_type == "PAYMENT" else 0
        type_transfer = 1 if txn_type == "TRANSFER" else 0

        # 4️⃣ Create input array (EXACTLY 11 FEATURES — SAME AS TRAINING)
        input_data = np.array([[
            step,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            isFlaggedFraud,
            type_cash_out,
            type_debit,
            type_payment,
            type_transfer
        ]])

        # 5️⃣ Predict
        prediction = model.predict(input_data)[0]

        result_text = "🚨 Fraud Transaction" if prediction == 1 else "✅ Not a Fraud Transaction"

        return render_template("submit.html", prediction=result_text)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
