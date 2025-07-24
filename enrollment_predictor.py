import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Print current working directory
print("📁 Current working directory:", os.getcwd())

# Step 1: Load data
df = pd.read_csv("students.csv")

# Step 2: Preprocess data
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['enrolled'] = df['enrolled'].map({'yes': 1, 'no': 0})

X = df[['age', 'gender', 'high_school_score', 'attendance_rate']]
y = df['enrolled']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Step 6: Feature importance chart
features = X.columns
importances = model.feature_importances_

plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

# Show and auto-close plot
plt.show(block=False)
plt.pause(3)
plt.close()

# Step 7: Save model with full path
save_path = os.path.abspath("student_model.joblib")
joblib.dump(model, save_path)
print(f"\n✅ Model saved successfully at:\n{save_path}")
