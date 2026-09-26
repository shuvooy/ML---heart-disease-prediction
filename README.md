# Heart Disease Prediction

A simple ML pipeline that predicts the presence of heart disease from clinical
data, comparing Logistic Regression against a Random Forest classifier.

## Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
(918 records, 11 features + target). Place `heart.csv` in the project root
before running the script.

**Known data quality issue:** ~19% of rows have `Cholesterol == 0`, which is
physiologically impossible and almost certainly represents missing values
encoded as zero. This script does not impute or filter them — results should
be read with that in mind. A future improvement would be to treat `0` as
`NaN` and impute (e.g. median by class) before training.

## What it does

1. Loads and inspects the dataset (nulls, summary stats).
2. Builds a preprocessing pipeline: `StandardScaler` for numeric features,
   `OneHotEncoder` for categorical features.
3. Trains and evaluates two classifiers:
   - Logistic Regression
   - Random Forest (200 trees)
4. Prints classification reports (precision/recall/F1) for both.
5. Ranks features by importance using the Random Forest model.
6. Saves the Random Forest pipeline to `heart_model.pkl` and reloads it to
   verify a sample prediction round-trips correctly.

## Results

On a held-out 20% test split (random_state=42):

| Model               | Accuracy | Precision (Disease) | Recall (Disease) |
|---------------------|----------|----------------------|-------------------|
| Logistic Regression | 0.89     | 0.87                 | 0.93              |
| Random Forest       | 0.90     | 0.90                 | 0.93              |

Top predictive features (Random Forest): `ST_Slope`, `Oldpeak`,
`Cholesterol`, `ChestPainType`.

Tech Stack
Python
pandas / NumPy — data loading and manipulation
scikit-learn — preprocessing, modeling, evaluation
joblib — model serialization

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Notes

- `heart_model.pkl` is not committed — it's generated when you run the
  script, and is tied to your local scikit-learn version. Regenerate it
  rather than sharing the binary across environments.
- This is for educational purposes only and is not a diagnostic tool.
