#!/usr/bin/env python
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# ── 1) Enable multi-threading ───────────────────────────────────────────────
n_threads = 16
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
torch.set_num_threads(n_threads)

# ── 2) Load & split your data ───────────────────────────────────────────────
df = pd.read_csv('final_training.csv')
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['los'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['los'], random_state=42
)

print("Sizes:")
print(f"  Train:      {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"  Validation: {len(val_df)} ({len(val_df)/len(df):.1%})")
print(f"  Test:       {len(test_df)} ({len(test_df)/len(df):.1%})\n")

for name, subset in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
    dist = subset['los'].value_counts(normalize=True).mul(100).round(2)
    print(f"{name} los distribution:\n{dist.to_frame(name='%')}")

# ── 3) Prepare feature matrices & targets ───────────────────────────────────
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
# list of cardinalities for each cat feature
categories = [X_train[col].nunique() for col in categorical_cols]

# torch tensors
X_train_num = torch.tensor(X_train[numeric_cols].values, dtype=torch.float32)
X_val_num   = torch.tensor(X_val[numeric_cols].values,   dtype=torch.float32)
X_test_num  = torch.tensor(X_test[numeric_cols].values,  dtype=torch.float32)

X_train_cat = torch.tensor(X_train[categorical_cols].values, dtype=torch.long)
X_val_cat   = torch.tensor(X_val[categorical_cols].values,   dtype=torch.long)
X_test_cat  = torch.tensor(X_test[categorical_cols].values,  dtype=torch.long)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32)

# create simple sample‐ids
ids_train = torch.arange(len(y_train_t), dtype=torch.long)
ids_val   = torch.arange(len(y_val_t),   dtype=torch.long)
ids_test  = torch.arange(len(y_test_t),  dtype=torch.long)

# ── 4) Build & train the T-MLP model ────────────────────────────────────────
from models.tmlp import tMLP

model_config = {
    'd_token':        1024,
    'n_layers':       1,
    'token_bias':     True,
    'd_ffn_factor':   0.66,
    'ffn_dropout':    None,
    'residual_dropout': 0.1,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = tMLP(
    model_config=model_config,
    n_num_features=X_train_num.shape[1],
    categories=categories,
    n_labels=1,
    device=device,
    feat_gate=None,
    pruning=None,
    dataset=None
)

model.fit(
    # training data (4-tuple)
    X_num = X_train_num,
    X_cat = X_train_cat,
    ys    = y_train_t,
    ids   = ids_train,     # ← tell it your sample IDs
    y_std = None,          # ← none for classification

    # evaluation: a tuple of (val_4tuple, test_4tuple)
    eval_set = (
        (X_val_num,  X_val_cat,  y_val_t,  ids_val),
        (X_test_num, X_test_cat, y_test_t, ids_test),
    ),

    patience      = 10,
    task          = 'binclass',
    training_args = {
        'lr':           1e-4,
        'optimizer':    'adamw',
        'weight_decay': 0.0,
        'batch_size':   256,
        'max_epochs':   50,      # ← use max_epochs
    },
    meta_args = {
        'save_path': 'results/tmlp/custom',
        'use_auc':   True,
        'use_r2':    False,
    },
)

# ── 5) Evaluate & print your desired metrics ───────────────────────────────
# get test‐set probabilities
preds_prob, _ = model.predict(
    X_num        = X_test_num,
    X_cat        = X_test_cat,
    ys           = y_test_t,
    ids          = ids_test,
    y_std        = None,
    task         = 'binclass',
    return_probs = True,
    return_metric= False,
    return_loss  = False,
    meta_args    = None,
)

y_true = y_test_t.cpu().numpy().astype(int)
y_pred = (preds_prob >= 0.5).astype(int)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)

print(f"\nTest set metrics:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 score : {f1:.4f}\n")

print("Full classification report:")
print(classification_report(y_true, y_pred))
