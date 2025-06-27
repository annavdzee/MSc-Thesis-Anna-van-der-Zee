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
from sklearn.model_selection import GridSearchCV
import re


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

def clean_columns(df):
    import re
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    return df

X_train = clean_columns(X_train)
X_val = clean_columns(X_val)
X_test = clean_columns(X_test)


from lightgbm import LGBMClassifier

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    for df_ in [X_train, X_val, X_test]:
        df_[col] = df_[col].astype('category').cat.codes

cat_col_idx = [X_train.columns.get_loc(col) for col in categorical_cols]

param_grid = {
    'num_leaves': [5, 20, 31],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150]
}

grid_search = GridSearchCV(
    estimator=LGBMClassifier(random_state=42, device_type='gpu'),
    param_grid=param_grid,
    scoring='accuracy', 
    cv=5,
    verbose=2
)
grid_search.fit(
    X_train, y_train,
    categorical_feature=cat_col_idx
)


from sklearn.model_selection import GridSearchCV

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

y_test_scores = best_dt_pipe.predict_proba(X_test)[:, 1]

# compute AUROC and AUPRC
auroc = roc_auc_score(y_test, y_test_scores)
auprc = average_precision_score(y_test, y_test_scores)
print(f"AUROC: {auroc:.2f}")
print(f"AUPRC: {auprc:.2f}")
