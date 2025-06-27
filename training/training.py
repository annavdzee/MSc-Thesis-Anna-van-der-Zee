import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

n_threads = 16
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
torch.set_num_threads(n_threads)

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

from models.tmlp import tMLP

model_config = {
    'd_token':        256,
    'n_layers':       3,
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

training_args = {
    'lr':           1e-3,
    'optimizer':    'adamw',
    'weight_decay': 0.0,
    'batch_size':   64,
    'max_epochs':   50
}

meta_args = {
    'save_path': 'results/tmlp/custom',
    'use_auc':   True,
    'use_r2':    False,
}


model.fit(
    # training data (4-tuple)
    X_num = X_train_num,
    X_cat = X_train_cat,
    ys    = y_train_t,
    ids   = ids_train,     
    y_std = None,

    # evaluation
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
        'max_epochs':   50,      
    },
    meta_args = {
        'save_path': 'results/tmlp/custom',
        'use_auc':   True,
        'use_r2':    False,
    },
)

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

from sklearn.metrics import roc_auc_score, average_precision_score

auroc = roc_auc_score(y_true, preds_prob)
auprc = average_precision_score(y_true, preds_prob)

print(f"  AUROC : {auroc:.4f}")
print(f"  AUPRC : {auprc:.4f}")


# Equalized Odds

import numpy as np

A_test = test_df['gender'].astype('category').cat.codes.values
print(A_test)

gender_cat = test_df['gender'].astype('category')
print("Gender categories (in code order):", list(gender_cat.cat.categories))

for val in gender_cat.cat.categories:
    code = gender_cat.cat.categories.get_loc(val)
    print(f"  {val!r} → code {code}")

metrics = {}
for group in [0, 1]:
    # mask for this group
    mask = (A_test == group)
    y_true_grp = y_true[mask]
    y_pred_grp = y_pred[mask]

    TP = np.logical_and(y_pred_grp == 1, y_true_grp == 1).sum()
    FN = np.logical_and(y_pred_grp == 0, y_true_grp == 1).sum()
    FP = np.logical_and(y_pred_grp == 1, y_true_grp == 0).sum()
    TN = np.logical_and(y_pred_grp == 0, y_true_grp == 0).sum()

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    metrics[group] = {'TPR': TPR, 'FPR': FPR, 'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}

delta_TPR = abs(metrics[0]['TPR'] - metrics[1]['TPR'])
delta_FPR = abs(metrics[0]['FPR'] - metrics[1]['FPR'])

eps = 1e-8
ratio_TPR = (metrics[0]['TPR'] + eps) / (metrics[1]['TPR'] + eps)
ratio_FPR = (metrics[0]['FPR'] + eps) / (metrics[1]['FPR'] + eps)

print("\n— Fairness Metrics (Equalized Odds) —")
print(f"Group 0 (gender=0): TPR = {metrics[0]['TPR']:.4f}, FPR = {metrics[0]['FPR']:.4f}")
print(f"Group 1 (gender=1): TPR = {metrics[1]['TPR']:.4f}, FPR = {metrics[1]['FPR']:.4f}")
print(f"Absolute TPR gap: ΔTPR = {delta_TPR:.4f}")
print(f"Absolute FPR gap: ΔFPR = {delta_FPR:.4f}")
print(f"TPR ratio (group0 / group1): {ratio_TPR:.4f}")
print(f"FPR ratio (group0 / group1): {ratio_FPR:.4f}")

dp_rate_0 = (y_pred[A_test == 0] == 1).mean()
dp_rate_1 = (y_pred[A_test == 1] == 1).mean()
dp_ratio  = dp_rate_0 / dp_rate_1 if dp_rate_1 > 0 else float('nan')

print("\n— Fairness Metrics (Demographic Parity) —")
print(f"P(Ŷ=1 | gender=0): {dp_rate_0:.4f}")
print(f"P(Ŷ=1 | gender=1): {dp_rate_1:.4f}")
print(f"Demographic Parity ratio (group0 / group1): {dp_ratio:.4f}")
print("tMLP attributes:", [k for k in vars(model) if not k.startswith("_")])
# and also:
print("Callables on model:", [k for k,v in model.__class__.__dict__.items() if callable(v)])

#SHAP
import shap
import numpy as np
n_num = X_train_num.shape[1]

def _predict_proba(self, data_array: np.ndarray) -> np.ndarray:
    X_num = data_array[:, :n_num]
    X_cat = data_array[:, n_num:]

    # to torch tensors
    num_t = torch.from_numpy(X_num).float().to(device)
    cat_t = torch.from_numpy(X_cat).long().to(device)

    # forward 
    with torch.no_grad():
        logits = self.model(num_t, cat_t).view(-1)    
        probs1 = torch.sigmoid(logits).cpu().numpy()
    probs0 = 1.0 - probs1
    return np.vstack([probs0, probs1]).T            

# attach it
model.predict_proba = _predict_proba.__get__(model, model.__class__)

bg = X_train.values[:100]
explainer = shap.KernelExplainer(model.predict_proba, bg)
test_bg = X_test.values[:800]
shap_values = explainer.shap_values(test_bg, nsamples=200)
all_shap = np.stack(shap_values, axis=0)
shap_vals_pos = all_shap[:, :, 1]

# mean absolute per feature
mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)

feature_names = list(X_test.columns)
feat_imp = dict(zip(feature_names, mean_abs_shap))
top10   = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
features, importances = zip(*top10)

for f, imp in top10:
    print(f"  {f:20s} {imp:.4f}")

# from itertools import product
# from sklearn.metrics import accuracy_score

# # Define the grid of hyperparameters
# grid = {
#     'lr':           [1e-4, 1e-3, 1e-2],
#     'batch_size':   [64, 128, 256],
#     'n_layers':     [1, 2, 3],
#     'd_token':      [256, 512, 1024],
# }

# results = []
# for lr, batch_size, n_layers, d_token in product(
#         grid['lr'], grid['batch_size'],
#         grid['n_layers'], grid['d_token']
#     ):
#     # 1) update configs
#     training_args['lr']         = lr
#     training_args['batch_size'] = batch_size
#     model_config['n_layers']    = n_layers
#     model_config['d_token']     = d_token

#     # 2) re-instantiate & train
#     model = tMLP(   # same code as above
#         model_config=model_config,
#         n_num_features=X_train_num.shape[1],
#         categories=categories,
#         n_labels=1,
#         device=device,
#         feat_gate=None,
#         pruning=None,
#         dataset=None
#     )
#     model.fit(
#         X_num   = X_train_num,
#         X_cat   = X_train_cat,
#         ys      = y_train_t,
#         ids     = ids_train,
#         y_std   = None,
#         eval_set=(
#           (X_val_num, X_val_cat, y_val_t, ids_val),
#           (X_test_num, X_test_cat, y_test_t, ids_test),
#         ),
#         patience      = 10,
#         task          = 'binclass',
#         training_args=training_args,
#         meta_args    = meta_args,
#     )

#     # 3) evaluate on test
#     preds_prob, _ = model.predict(
#         X_num        = X_test_num,
#         X_cat        = X_test_cat,
#         ys           = y_test_t,
#         ids          = ids_test,
#         y_std        = None,
#         task         = 'binclass',
#         return_probs = True,
#         return_metric=False,
#         return_loss  = False,
#         meta_args    = None,
#     )
#     y_pred = (preds_prob >= 0.5).astype(int)
#     acc = accuracy_score(y_test_t.numpy().astype(int), y_pred)

#     # 4) record
#     results.append({
#         'lr': lr,
#         'batch_size': batch_size,
#         'n_layers': n_layers,
#         'd_token': d_token,
#         'accuracy': acc,
#     })

# # 5) sort & print best
# results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
# print("Top 5 configs:")
# for r in results[:5]:
#     print(r)
