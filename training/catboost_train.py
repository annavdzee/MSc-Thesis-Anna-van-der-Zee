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
from catboost import CatBoostClassifier

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

print(df.columns.tolist())
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}: {col}")


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

# categoricals as integer codes
categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
for col in categorical_cols:
    for df_ in [X_train, X_val, X_test]:
        df_[col] = df_[col].astype('category').cat.codes

cat_features = categorical_cols

catboost_pipe = CatBoostClassifier(
    task_type='GPU',
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=2,
    boosting_type='Plain',
    verbose=0,
    random_state=42
)


# Grid for CatBoost
param_grid = {
    'learning_rate': [0.03, 0.06, 0.1],
    'depth': [3, 6],
    'l2_leaf_reg': [2, 3, 4],
    'boosting_type': ['Ordered', 'Plain']
}

# Grid search
# grid_search = GridSearchCV(
#     estimator=catboost_pipe,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=5,
#     verbose=2
# )

# # Fit grid search
# grid_search.fit(X_train, y_train, cat_features=cat_features)

# # Use best model
# best_model = grid_search.best_estimator_
# print("Best parameters:", grid_search.best_params_)


# Fit grid search
catboost_pipe.fit(X_train, y_train, cat_features=cat_features)

# Use best model


# # Evaluate on validation set
# y_val_pred = best_model.predict(X_val)
# print(classification_report(y_val, y_val_pred))
# print("Confusion matrix:\n", confusion_matrix(y_val, y_val_pred))

# # Evaluate on test set
# y_test_pred = best_model.predict(X_test)
# print(classification_report(y_test, y_test_pred))
# print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

# # Metrics
# acc = accuracy_score(y_test, y_test_pred)
# prec = precision_score(y_test, y_test_pred)
# rec = recall_score(y_test, y_test_pred)
# f1 = f1_score(y_test, y_test_pred)
# print(f"Accuracy : {acc:.2f}")
# print(f"Precision: {prec:.2f}")
# print(f"Recall   : {rec:.2f}")
# print(f"F1-Score : {f1:.2f}")

# # AUROC and AUPRC
# y_test_scores = best_model.predict_proba(X_test)[:, 1]
# auroc = roc_auc_score(y_test, y_test_scores)
# auprc = average_precision_score(y_test, y_test_scores)
# print(f"AUROC: {auroc:.2f}")
# print(f"AUPRC: {auprc:.2f}")

import shap
import numpy as np
import matplotlib.pyplot as plt


explainer = shap.TreeExplainer(catboost_pipe)

shap_values = explainer.shap_values(X_test)

shap_vals = shap_values

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
feature_importance = dict(zip(X_train.columns, mean_abs_shap))

top10 = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
features, importances = zip(*top10)

plt.figure(figsize=(8, 6))
plt.barh(features[::-1], importances[::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("Top 10 Feature Importances by SHAP")
plt.tight_layout()
#plt.savefig("shap_top10_feature_importance.png", dpi=150)

plt.show()
plt.clf()  

for feature, importance in top10:
    print(f"  {feature:20s} {importance:.4f}")

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals, 
    X_test, 
    plot_type="dot", 
    max_display=10,   
    show=False         
)
plt.tight_layout()
plt.savefig("shap_beeswarm_summary.png", dpi=150)

