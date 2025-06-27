from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

df_main = pd.read_csv('final_training.csv')
df_admissions = pd.read_csv('admissions_filtered.csv')  # has los_days
df = df_main.merge(df_admissions[['admissionid', 'los_days']], on='admissionid', how='left')

print(df['los_days'].describe())

df = df[df['los_days'] <= 21].copy()
print(df['los_days'].describe())

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42)

drop_cols   = ['admissionid', 'gender', 'los', 'los_days']
X_train, y_train = train_df.drop(columns=drop_cols), train_df['los_days']
X_val,   y_val   = val_df.drop(columns=drop_cols),   val_df['los_days']
X_test,  y_test  = test_df.drop(columns=drop_cols),  test_df['los_days']

categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols     = X_train.select_dtypes(include=['number']).columns.tolist()

ohe_passthrough = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])
ohe_scale = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

# regression pipelines
pipelines = {
    'Linear Regression': Pipeline([
        ('pre', ohe_scale),
        ('reg', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('pre', ohe_passthrough),
        ('reg', RandomForestRegressor(random_state=42))
    ]),
    'MLP': Pipeline([
        ('pre', ohe_scale),
        ('reg', MLPRegressor(random_state=42))
    ])
}
pipelines['XGBoost'] = Pipeline([
        ('pre', ohe_passthrough),
        ('reg', XGBRegressor(random_state=42))
    ])

param_grids = {
    'Random Forest': {
        'reg__n_estimators':      [500],
        'reg__max_depth':         [30, 40],
        'reg__min_samples_split': [5],
        'reg__min_samples_leaf':  [1, 2],
        'reg__max_features':      ['sqrt']
    },
    'MLP': {
        'reg__hidden_layer_sizes': [(10,30,10),(20,)],
        'reg__activation':         ['tanh','relu'],
        'reg__solver':             ['sgd','adam'],
        'reg__alpha':              [0.0001,0.05],
        'reg__learning_rate':      ['constant','adaptive'],
    },
    'XGBoost': {
        'reg__max_depth':    [3, 5],
        'reg__learning_rate':[0.1],
        'reg__subsample':    [0.7]
    },
    'Linear Regression': {
        'reg__copy_X':      [True, False],
        'reg__fit_intercept':[True, False],
        'reg__n_jobs':      [1, 5, 10, 15, None],
        'reg__positive':    [True, False]
    }
}

# run GridSearchCV 
best_pipelines = {}
for name, pipe in pipelines.items():
    print(f"\n>>> Tuning hyperparameters for {name}")
    grid = GridSearchCV(
        pipe,
        param_grid=param_grids[name],
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=3
    )
    grid.fit(X_train, y_train)
    best_pipelines[name] = grid.best_estimator_
    
    print(f"Best params for {name}:")
    for k,v in grid.best_params_.items():
        print(f"  {k} = {v}")
    print(f"CV RMSE: {(-grid.best_score_):.4f}")

    # evaluate on validation set
    y_val_pred = grid.predict(X_val)
    rmse_val   = root_mean_squared_error(y_val, y_val_pred)
    r2_val     = r2_score(y_val, y_val_pred)
    print(f"Validation RMSE: {rmse_val:.4f},  R2: {r2_val:.4f}")

print("\n>>> Final test performance:")
for name, best_pipe in best_pipelines.items():
    y_test_pred = best_pipe.predict(X_test)
    rmse_tst    = root_mean_squared_error(y_test, y_test_pred)
    r2_tst      = r2_score(y_test, y_test_pred)
    mae_tst = mean_absolute_error(y_test, y_test_pred)
    print(f"{name}: RMSE = {rmse_tst:.4f},  R2 = {r2_tst:.4f}, MAE = {mae_tst:.4f}")
