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
from xgboost import XGBClassifier

n_threads = 16
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
torch.set_num_threads(n_threads)

# Load data
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

# Encode categoricals as integer codes
categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
for col in categorical_cols:
    for df_ in [X_train, X_val, X_test]:
        df_[col] = df_[col].astype('category').cat.codes

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])

# XGBoost pipeline with GPU support
xgb_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', XGBClassifier(
        eval_metric='logloss',
        device='cuda',
        random_state=42
    ))
])

# Parameter grid for grid search
param_grid = {
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.1, 0.01, 0.001],
    'clf__subsample': [0.5, 0.7, 1]
}

# Grid search setup
grid_search = GridSearchCV(
    estimator=xgb_pipe,
    param_grid=param_grid,
    scoring='accuracy',  
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Use best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Predict and evaluate on validation set
y_val_pred = best_model.predict(X_val)
print(classification_report(y_val, y_val_pred))
print("Confusion matrix:\n", confusion_matrix(y_val, y_val_pred))

# Predict on test set
y_test_pred = best_model.predict(X_test)
print(classification_report(y_test, y_test_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

# Performance metrics
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
print(f"Accuracy : {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall   : {rec:.2f}")
print(f"F1-Score : {f1:.2f}")

# AUROC and AUPRC
y_test_scores = best_model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_test_scores)
auprc = average_precision_score(y_test, y_test_scores)
print(f"AUROC: {auroc:.2f}")
print(f"AUPRC: {auprc:.2f}")
