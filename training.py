import pandas as pd

df = pd.read_csv('final_training.csv')  

from sklearn.model_selection import train_test_split

# first split: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df['gender'],
    random_state=42
)

# second split: split the 30% temp into 15% validation and 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['gender'],
    random_state=42
)

# check shapes
print("Sizes:")
print(f"  Train:      {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"  Validation: {len(val_df)} ({len(val_df)/len(df):.1%})")
print(f"  Test:       {len(test_df)} ({len(test_df)/len(df):.1%})")

# check gender ratio in each split
for name, subset in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
    print(f"\n{name} gender distribution:")
    print(subset['los'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

drop_cols = ['admissionid', 'gender']
X_train, y_train = train_df.drop(columns=drop_cols+['los']), train_df['los']
X_val,   y_val   = val_df.drop(columns=drop_cols+['los']),   val_df['los']
X_test,  y_test  = test_df.drop(columns=drop_cols+['los']),  test_df['los']

categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols     = X_train.select_dtypes(include=['int64','float64']).columns.tolist()

for col in categorical_cols:
    for df_ in [X_train, X_val, X_test]:
        df_[col] = df_[col].astype('category').cat.codes
categories = [X_train[col].nunique() for col in categorical_cols]

import torch

X_train_num = torch.tensor(X_train[numeric_cols].values, dtype=torch.float32)
X_val_num   = torch.tensor(X_val[numeric_cols].values, dtype=torch.float32)
X_test_num  = torch.tensor(X_test[numeric_cols].values, dtype=torch.float32)

X_train_cat = torch.tensor(X_train[categorical_cols].values, dtype=torch.long)
X_val_cat   = torch.tensor(X_val[categorical_cols].values, dtype=torch.long)
X_test_cat  = torch.tensor(X_test[categorical_cols].values, dtype=torch.long)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
y_val_t   = torch.tensor(y_val.values, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32)
