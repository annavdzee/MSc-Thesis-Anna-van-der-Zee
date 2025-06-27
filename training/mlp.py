import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

n_threads = 16
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
torch.set_num_threads(n_threads)

# load data
df = pd.read_csv('final_training.csv')
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['gender'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['gender'], random_state=42
)

print("Sizes:")
print(f"  Train:      {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"  Validation: {len(val_df)} ({len(val_df)/len(df):.1%})")
print(f"  Test:       {len(test_df)} ({len(test_df)/len(df):.1%})\n")

for name, subset in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
    dist = subset['gender'].value_counts(normalize=True).mul(100).round(2)
    print(f"{name} gender distribution:\n{dist.to_frame(name='%')}")

drop_cols = ['admissionid', 'gender']
X_train = train_df.drop(columns=drop_cols + ['los'])
X_val   = val_df.drop(columns=drop_cols + ['los'])
X_test  = test_df.drop(columns=drop_cols + ['los'])

y_train = train_df['los']
y_val   = val_df['los']
y_test  = test_df['los']

# encode categoricals as integer codes
categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols     = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
for col in categorical_cols:
    for df_ in [X_train, X_val, X_test]:
        df_[col] = df_[col].astype('category').cat.codes
categories = [X_train[col].nunique() for col in categorical_cols]


drop_cols = ['admissionid', 'gender']
X_train, y_train = train_df.drop(columns=drop_cols+['los']), train_df['los']
X_val,   y_val   = val_df.drop(columns=drop_cols+['los']),   val_df['los']
X_test,  y_test  = test_df.drop(columns=drop_cols+['los']),  test_df['los']

categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols     = X_train.select_dtypes(include=['int64','float64']).columns.tolist()

#  preprocessor 
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])

# Decision Tree pipeline
from sklearn.neural_network import MLPClassifier

mlp_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', MLPClassifier(
        early_stopping=True,
        random_state=42))
])

param_grid = {
    'clf__hidden_layer_sizes': [(10,30,10),(20,)],
    'clf__activation': ['tanh', 'relu'],
    'clf__solver': ['sgd', 'adam'],
    'clf__alpha': [0.0001, 0.05],
    'clf__learning_rate': ['constant','adaptive']
}


mlp_pipe.fit(X_train, y_train)
y_val_dt = mlp_pipe.predict(X_val)

from sklearn.model_selection import GridSearchCV

# Grid search setup
grid_search = GridSearchCV(
    estimator=mlp_pipe,
    param_grid=param_grid,
    scoring='accuracy', 
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Use best model from grid search
best_dt_pipe = grid_search.best_estimator_

# Predict and evaluate
y_val_dt = best_dt_pipe.predict(X_val)
print("Best parameters:", grid_search.best_params_)
print(classification_report(y_val, y_val_dt))
print("Confusion matrix:\n", confusion_matrix(y_val, y_val_dt))


y_test_dt = best_dt_pipe.predict(X_test)

print(classification_report(y_test, y_test_dt))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_dt))

acc    = accuracy_score(y_test, y_test_dt)
prec   = precision_score(y_test, y_test_dt)       
rec    = recall_score(y_test, y_test_dt)
f1     = f1_score(y_test, y_test_dt)

print(f"Accuracy : {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall   : {rec:.2f}")
print(f"F1-Score : {f1:.2f}")

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

y_test_scores = mlp_pipe.predict_proba(X_test)[:, 1]

# compute AUROC and AUPRC
auroc = roc_auc_score(y_test, y_test_scores)
auprc = average_precision_score(y_test, y_test_scores)
print(f"AUROC: {auroc:.2f}")
print(f"AUPRC: {auprc:.2f}")

#SHAP
import shap
import numpy as np
import matplotlib.pyplot as plt

background = X_train.sample(100, random_state=42)

def predict_fn(data_array):
    """
    SHAP will pass in a numpy array of shape (n_rows, n_features).
    Convert it back to a DataFrame so that ColumnTransformer
    can select columns by name.
    """
    df = pd.DataFrame(data_array, columns=X_train.columns)
    return mlp_pipe.predict_proba(df)

explainer = shap.KernelExplainer(predict_fn, background)

shap_values = explainer.shap_values(X_test.iloc[:800], nsamples=200)

all_shap = np.stack(shap_values, axis=0)

shap_vals_pos = all_shap[:, :, 1]

mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)

feature_importance = dict(zip(X_train.columns, mean_abs_shap))
top10 = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
features, importances = zip(*top10)

for feat, imp in top10:
    print(f"  {feat:20s} {imp:.4f}")
