import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Load the trained model
# ------------------------------
model = joblib.load("student_model.joblib")

# ------------------------------
# Step 2: New student data
# (contains student_id but will be masked for privacy)
# ------------------------------
new_data = pd.DataFrame([
    {
        "student_id": 7,
        "age": 19,
        "gender": "Female",
        "high_school_score": 85,
        "attendance_rate": 90
    },
    {
        "student_id": 8,
        "age": 18,
        "gender": "Male",
        "high_school_score": 60,
        "attendance_rate": 75
    },
    {
        "student_id": 9,
        "age": 20,
        "gender": "Female",
        "high_school_score": 92,
        "attendance_rate": 96
    }
])

# ------------------------------
# Step 3: Preprocess gender
# ------------------------------
new_data['gender'] = new_data['gender'].map({'Male': 0, 'Female': 1})

# ------------------------------
# Step 4: Extract features for prediction
# ------------------------------
X_new = new_data[['age', 'gender', 'high_school_score', 'attendance_rate']]

# ------------------------------
# Step 5: Predict outcomes
# ------------------------------
predictions = model.predict(X_new)

# ------------------------------
# Step 6: Display predictions (with masked IDs)
# ------------------------------
results = []
for i, prediction in enumerate(predictions):
    # Data privacy: Mask or omit student ID
    masked_id = f"Student #{i+1}"  # instead of showing real ID
    result_label = "Likely to Enroll" if prediction == 1 else "Needs Support"
    print(f"{masked_id}: {result_label}")
    results.append(result_label)

# ------------------------------
# Step 7: Visualization (Bar Chart)
# ------------------------------
result_counts = pd.Series(results).value_counts()

plt.figure(figsize=(6, 4))
result_counts.plot(kind='bar', color=['green', 'red'])
plt.title("Enrollment Prediction Summary")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()

# Optional: Save chart as image
plt.savefig("prediction_summary_chart.png")

# Show chart
plt.show()
