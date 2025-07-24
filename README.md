# Student Enrollment Prediction Model

This project uses a machine learning model to predict whether a student is likely to enroll in a university program or may need additional support. It is developed using Python, scikit-learn, pandas, and matplotlib.

## Project Structure

- `train_model.py` – Trains and saves the logistic regression model using student data.
- `predict_new_student.py` – Loads the saved model, predicts on new student records, masks student IDs for privacy, and generates a bar chart summary.
- `student_data.csv` – Sample training dataset.
- `student_model.joblib` – Serialized (saved) model after training.
- `prediction_summary_chart.png` – A chart showing prediction outcomes.
- `README.md` – Project documentation.

## Model Features

The model uses the following features to make predictions:

- `age`
- `gender` (converted to numeric: 0 = Male, 1 = Female)
- `high_school_score`
- `attendance_rate`

## Prediction Outcome

The model outputs whether a student is:

- **Likely to Enroll ✅**
- **Needs Support ❌**

A visualization is generated to summarize predictions in a bar chart.

## Data Privacy Measures

To protect sensitive information:
- **Student IDs are masked** in visual outputs and only used internally.
- The model does not retain personally identifiable information (PII).

## Requirements

Install dependencies using:

```bash
pip install pandas scikit-learn matplotlib joblib
```

## How to Run

1. Train the model:

```bash
enrollment_predictor.py
```

2. Predict `new` students:

```bash
python predict_new_student.py
```

## License

[MIT](https://choosealicense.com/licenses/mit/)