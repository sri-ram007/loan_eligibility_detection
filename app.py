from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/form', methods=['GET'])
def form():
    return render_template("form.html")

@app.route('/About',methods=['GET'])
def About():
      return render_template('about.html');

@app.route("/check_eligibility", methods=["POST"])
def check_eligibility():
    try:
        # Get form data
        age = int(request.form["age"])
        income = int(request.form["income"])
        credit_score = int(request.form["credit_score"])
        loan_amount = int(request.form["loan_amount"])

        # Validate inputs
        if age < 18 or income <= 0 or credit_score < 0 or credit_score > 850 or loan_amount <= 0:
            error_message = "Invalid input. Please check your entries."
            return render_template("form.html", error=error_message)

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
        return render_template("error.html", error=str(e))

# Error handler for unexpected issues
@app.errorhandler(Exception)
def handle_exception(e):
    return render_template("error.html", error=str(e)), 500

# Error handler for 404 (Page Not Found)
@app.errorhandler(404)
def page_not_found(error):
    return render_template("error.html", error="The page you are looking for does not exist."), 404

# Error handler for 500 (Internal Server Error)
@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error="An unexpected error occurred. Please try again later."), 500

if __name__ == "__main__":
    app.run(debug=True)
