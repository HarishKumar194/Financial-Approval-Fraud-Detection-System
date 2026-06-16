import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# LOAD / CREATE DATASET
# -------------------------------
data = {
    "amount": [500, 2000, 15000, 700, 12000, 300, 8000, 400, 10000, 600],
    "category_risk": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    "employee_level": [1, 2, 3, 1, 3, 1, 2, 1, 3, 1],
    "is_fraud": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# -------------------------------
# TRAIN MODEL
# -------------------------------
X = df[["amount", "category_risk", "employee_level"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# ACCURACY
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -------------------------------
# APPROVAL SYSTEM
# -------------------------------
def approve_request(amount, risk, level):
    prediction = model.predict([[amount, risk, level]])[0]

    if prediction == 1:
        return "❌ REJECTED (Fraud Detected)"

    if amount < 1000:
        return "✅ AUTO APPROVED"
    elif amount < 10000:
        return "👨‍💼 MANAGER APPROVAL"
    else:
        return "🏦 FINANCE HEAD APPROVAL"

# -------------------------------
# TEST RUN
# -------------------------------
requests = [
    (500, 0, 1),
    (8000, 1, 2),
    (12000, 1, 3),
]

for amt, risk, lvl in requests:
    print(f"Request: {amt} -> {approve_request(amt, risk, lvl)}")
