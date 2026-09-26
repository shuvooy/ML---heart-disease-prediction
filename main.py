import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

FILE_PATH = 'heart.csv'
MODEL_PATH = 'heart_model.pkl'
label_map = {0: 'No Disease', 1: 'Heart Disease'}

# Loading & inspecting 
df = pd.read_csv(FILE_PATH)

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.head())

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

numerical_cols   = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing 
num_transformer = Pipeline([('scaler', StandardScaler())])
cat_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore',
                                                       sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models 
lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

lr_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

y_pred_lr = lr_pipe.predict(X_test)
y_pred_rf = rf_pipe.predict(X_test)

print("Logistic Regression")
print(classification_report(y_test, y_pred_lr, target_names=['No Disease', 'Heart Disease']))

print("Random Forest")
print(classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Heart Disease']))

# Feature importance (Random Forest) 
ohe_cols = (rf_pipe.named_steps['preprocessor']
                   .named_transformers_['cat']
                   .named_steps['encoder']
                   .get_feature_names_out(categorical_cols).tolist())

all_cols    = numerical_cols + ohe_cols
importances = rf_pipe.named_steps['classifier'].feature_importances_

feat_df = (pd.DataFrame({'Feature': all_cols, 'Importance': importances})
           .sort_values('Importance', ascending=False)
           .reset_index(drop=True))

print(feat_df.head())

# Saving & reloading the Random Forest pipeline (best performer) 
joblib.dump(rf_pipe, MODEL_PATH)

model  = joblib.load(MODEL_PATH)
sample = X_test.iloc[[0]]
pred   = model.predict(sample)[0]
proba  = model.predict_proba(sample)[0]

print(f'actual    : {y_test.iloc[0]} ({label_map[y_test.iloc[0]]})')
print(f'predicted : {pred} ({label_map[pred]})')
print(f'proba     : No Disease = {proba[0]:.3f} | Heart Disease = {proba[1]:.3f}')
