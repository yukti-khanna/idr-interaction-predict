
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.model_selection import KFold





def mlp_optimised(X_train, X_test, y_train, y_test):
    # 1) Convert targets to NumPy arrays
    y_train_arr = y_train.to_numpy()
    y_test_arr  = y_test.to_numpy()

    # 2) Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3) Set your previously found best params
    best_params = {
        'activation': 'tanh',
        'alpha': 0.01,
        'hidden_layer_sizes': (100, 50, 25),
        'learning_rate_init': 0.01,
        'max_iter': 500,
        'solver': 'adam'
    }

    # 4) Initialize the MLP
    mlp = MLPRegressor(random_state=42, **best_params)

    # 5) 10‑fold CV
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    r2_scores  = []

    for train_idx, val_idx in cv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train_arr[train_idx],   y_train_arr[val_idx]

        mlp.fit(X_tr, y_tr)
        y_pred_cv = mlp.predict(X_val)

        mse_scores.append(mean_squared_error(y_val, y_pred_cv))
        r2_scores.append(r2_score(y_val, y_pred_cv))

    avg_mse = sum(mse_scores) / len(mse_scores)
    avg_r2  = sum(r2_scores)  / len(r2_scores)

    # 6) Retrain on full training set, then evaluate on test set
    mlp.fit(X_train_scaled, y_train_arr)
    y_pred_test = mlp.predict(X_test_scaled)

    mse_test = mean_squared_error(y_test_arr, y_pred_test)
    r2_test  = r2_score(y_test_arr,     y_pred_test)

    '''import joblib
    # Save
    joblib.dump(mlp, f'{out_folder}/mlp_optimized_model.pkl')
    joblib.dump(scaler, f'{out_folder}/scaler.pkl')'''


    print(f"Average CV MSE: {avg_mse:.5f}")
    print(f"Average CV R2:  {avg_r2:.4f}")
    print(f"Test MSE:         {mse_test:.5f}")
    print(f"Test R2:          {r2_test:.4f}")

    return mlp, y_pred_test


