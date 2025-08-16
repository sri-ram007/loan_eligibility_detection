from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/check_eligibility", methods=["POST"])
def check_eligibility():
    try:
        # Get form data
        age = int(request.form["age"])
        income = int(request.form["income"])
        credit_score = int(request.form["credit_score"])
        loan_amount = int(request.form["loan_amount"])

        # Prepare data for prediction
        input_data = np.array([[age, income, credit_score, loan_amount]])

        # Predict eligibility
        prediction = model.predict(input_data)[0]

        # Send result to a result page
        if prediction == 1:
            message = "Congratulations! You are eligible for the loan."
        else:
            message = "Sorry, you are not eligible for the loan."

        return render_template("result.html", message=message)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
