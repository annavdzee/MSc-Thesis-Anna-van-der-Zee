from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np

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
X_val = val_df.drop(columns=drop_cols + ['los'])
X_test = test_df.drop(columns=drop_cols + ['los'])

y_train = train_df['los']
y_val = val_df['los']
y_test = test_df['los']

categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
for col in categorical_cols:
    for df_ in [X_train, X_val, X_test]:
        df_[col] = df_[col].astype('category').cat.codes

#scale for logistic regression
preprocessor = ColumnTransformer([
    ('num',   StandardScaler(), numeric_cols),
    ('cat',   OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

logreg_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(random_state=42))
])


logreg_pipe.fit(X_train, y_train)
y_val_dt = logreg_pipe.predict(X_val)

from sklearn.model_selection import GridSearchCV

# Grid search setup
param_grid = [
    {'clf__penalty':['l1','l2','elasticnet','none'],
    'clf__C' : [0.0001, 0.01, 1, 100],
    'clf__solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    'clf__max_iter'  : [100,1000,2500,5000]
}
]

grid_search = GridSearchCV(
    estimator=logreg_pipe,
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

y_test_scores = logreg_pipe.predict_proba(X_test)[:, 1]

# compute AUROC and AUPRC
auroc = roc_auc_score(y_test, y_test_scores)
auprc = average_precision_score(y_test, y_test_scores)
print(f"AUROC: {auroc:.2f}")
print(f"AUPRC: {auprc:.2f}")

