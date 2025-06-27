import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from fairlearn.metrics import equalized_odds_ratio
from fairlearn.metrics import demographic_parity_ratio

df = pd.read_csv('final_training.csv')
train_df, temp_df = train_test_split(df, test_size=0.30,
                                     stratify=df['gender'],
                                     random_state=42)
val_df, test_df  = train_test_split(temp_df, test_size=0.50,
                                    stratify=temp_df['gender'],
                                    random_state=42)

drop_cols = ['admissionid','gender','los']
X_train, y_train = train_df.drop(columns=drop_cols), train_df['los']
X_test,  y_test  = test_df.drop(columns=drop_cols),  test_df['los']

categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols     = X_train.select_dtypes(include=['int64','float64']).columns.tolist()

# Cast categories for CatBoost
for df_ in (X_train, X_test):
    for c in categorical_cols:
        df_[c] = df_[c].astype('category')

# Preprocessors
ohe_passthrough = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])
ohe_scale = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

pipelines = {}

pipelines['Logistic Regression'] = Pipeline([
    ('pre', ohe_scale),
    ('clf', LogisticRegression(
        penalty='l1', C=100, solver='saga',
        max_iter=100, random_state=42
    ))
])

pipelines['Decision Tree'] = Pipeline([
    ('pre', ohe_passthrough),
    ('clf', DecisionTreeClassifier(
        max_depth=10, min_samples_split=10,
        min_samples_leaf=1, max_features='sqrt',
        random_state=42
    ))
])

pipelines['Random Forest'] = Pipeline([
    ('pre', ohe_passthrough),
    ('clf', RandomForestClassifier(
        n_estimators=200, max_depth=30, max_features='sqrt',
        min_samples_split=10, min_samples_leaf=1,
        random_state=42
    ))
])

try:
    pipelines['XGBoost'] = Pipeline([
        ('pre', ohe_passthrough),
        ('clf', XGBClassifier(
            use_label_encoder=False, eval_metric='logloss',
            learning_rate=0.1, max_depth=7, subsample=1,
            random_state=42
        ))
    ])
except Exception as e:
    print("Warning: XGBoost unavailable:", e)

pipelines['LightGBM'] = Pipeline([
    ('pre', ohe_passthrough),
    ('clf', LGBMClassifier(
        learning_rate=0.05, n_estimators=150,
        num_leaves=31, random_state=42
    ))
])

pipelines['MLP'] = Pipeline([
    ('pre', ohe_passthrough),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(20,), activation='relu',
        solver='adam', alpha=0.05, learning_rate='constant',
        max_iter=200, random_state=42
    ))
])

# Evaluate pipelines
sensitive = test_df['gender']
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    ratio  = equalized_odds_ratio(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    parity  = demographic_parity_ratio(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive
    )

    print(f"{name:20s} Equalized Odds Ratio: {ratio:.3f}")
    print(f"{name:20s} demographic parity ratio: {parity:.3f}")


from sklearn.metrics import confusion_matrix

def tpr_fpr_by_group(y_true, y_pred, groups):
    """
    Compute TPR and FPR for each group in 'groups'.
    Returns a dict: {group_value: (TPR, FPR)}
    """
    results = {}
    for g in groups.unique():
        mask = (groups == g)
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        results[g] = (tpr, fpr)
    return results

sensitive = test_df['gender']

# Evaluate each pipeline
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rates = tpr_fpr_by_group(y_test, y_pred, sensitive)
    tpr_man, fpr_man     = rates['Man']
    tpr_vrouw, fpr_vrouw = rates['Vrouw']

    print(f"\n{name}:")
    print(f"  Man    → TPR = {tpr_man:.3f}, FPR = {fpr_man:.3f}")
    print(f"  Vrouw  → TPR = {tpr_vrouw:.3f}, FPR = {fpr_vrouw:.3f}")
    print(f"  TPR ratio (Man/Vrouw) = {tpr_man/tpr_vrouw:.3f}")
    print(f"  FPR ratio (Man/Vrouw) = {fpr_man/fpr_vrouw:.3f}")

# CatBoost separately 
cb = CatBoostClassifier(
    learning_rate=0.03, depth=6, l2_leaf_reg=2,
    boosting_type='Plain', task_type='GPU', devices='0',
    verbose=0, random_state=42
)
cb.fit(X_train, y_train, cat_features=categorical_cols)
y_pred_cb = cb.predict(X_test)
ratio_cb  = equalized_odds_ratio(
    y_true=y_test,
    y_pred=y_pred_cb,
    sensitive_features=sensitive
)
parity_cb  = demographic_parity_ratio(
    y_true=y_test,
    y_pred=y_pred_cb,
    sensitive_features=sensitive
)

print(f"{'CatBoost':20s} Equalized Odds Ratio: {ratio_cb:.3f}")
print(f"{'CatBoost':20s} demographic parity ratio: {parity_cb:.3f}")

rates_cb = tpr_fpr_by_group(y_test, y_pred_cb, sensitive)

tpr_man_cb, fpr_man_cb       = rates_cb['Man']
tpr_vrouw_cb, fpr_vrouw_cb   = rates_cb['Vrouw']

print(f"\nCatBoost:")
print(f"  Man    → TPR = {tpr_man_cb:.3f}, FPR = {fpr_man_cb:.3f}")
print(f"  Vrouw  → TPR = {tpr_vrouw_cb:.3f}, FPR = {fpr_vrouw_cb:.3f}")
print(f"  TPR ratio (Man/Vrouw) = {tpr_man_cb/tpr_vrouw_cb:.3f}")
print(f"  FPR ratio (Man/Vrouw) = {fpr_man_cb/fpr_vrouw_cb:.3f}")
