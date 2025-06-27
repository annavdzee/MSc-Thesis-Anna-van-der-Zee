import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import shap

# Load & split data
df = pd.read_csv('final_training.csv')
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['gender'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['gender'], random_state=42
)
all_vars = df.columns.tolist()
print(all_vars)
print(len(all_vars))


# Prepare features/labels
drop_cols = ['admissionid', 'gender', 'los']
X_train, y_train = train_df.drop(columns=drop_cols), train_df['los']
X_test, y_test   = test_df.drop(columns=drop_cols),  test_df['los']

cat_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('num', 'passthrough', num_cols)
])

pipelines = {
    # 'RandomForest': Pipeline([
    #     ('pre', preprocessor),
    #     ('clf', RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42))
    # ]),
    'XGBoost': Pipeline([
        ('pre', preprocessor),
        ('clf', XGBClassifier(
            eval_metric='logloss',
            learning_rate=0.1,
            max_depth=7,
            subsample=1,
            random_state=42
        ))
    ]),
    'LightGBM': Pipeline([
        ('pre', preprocessor),
        ('clf', LGBMClassifier(
            learning_rate=0.05,
            n_estimators=150,
            num_leaves=31,
            random_state=42
        ))
    ]),
    'MLP': Pipeline([
        ('pre', preprocessor),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(20,),
            max_iter=200,
            random_state=42
        ))
    ]),
}

#SHAP
for name, pipe in pipelines.items():
    print(f"\n=== Processing {name} ===")
    pipe.fit(X_train, y_train)

    feature_names = pipe.named_steps['pre'].get_feature_names_out()
    X_pre = pipe.named_steps['pre'].transform(X_test)

    background = shap.sample(X_pre, 300, random_state=42)
    explainer = shap.Explainer(pipe.named_steps['clf'],
                               background,
                               feature_names=feature_names)
    shap_values = explainer(X_pre, check_additivity=False)
    sv = shap_values.values

    mean_abs = pd.Series(np.abs(sv).mean(axis=0),
                         index=feature_names)
    print(f"{name} — Top 10 SHAP features by mean(|value|):")
    print(mean_abs.nlargest(10).to_string(), "\n")

    # also get normal non-abs value
    mean_signed = pd.Series(sv.mean(axis=0),
                            index=feature_names)
    top_pos = mean_signed.nlargest(10)
    top_neg = mean_signed.nsmallest(10)

    print(f"{name} — Top 10 positive-mean SHAP features:")
    print(top_pos.to_string(), "\n")

    print(f"{name} — Top 10 negative-mean SHAP features:")
    print(top_neg.to_string())
