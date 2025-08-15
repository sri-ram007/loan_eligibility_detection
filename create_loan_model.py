import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

# Create dummy data
data = {
    "age": np.random.randint(18, 65, 1000),
    "income": np.random.randint(2000, 20000, 1000),
    "credit_score": np.random.randint(300, 900, 1000),
    "loan_amount": np.random.randint(500, 50000, 1000),
    "eligibility": np.random.choice([0, 1], size=1000, p=[0.6, 0.4])
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save synthetic datacreate_loan_model.py
df.to_csv("loan_data.csv", index=False)

# Features and target
X = df[["age", "income", "credit_score", "loan_amount"]]
y = df["eligibility"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler as .pkl files
with open("loan_model.pkl", "wb") as model_file, open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(model, model_file)
    pickle.dump(scaler, scaler_file)

print("Loan eligibility model and scaler saved as 'loan_model.pkl' and 'scaler.pkl'.")

