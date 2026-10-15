import json
import os
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
            return "tensorflow op determinism requested"
        except Exception as exc:
            return f"tensorflow op determinism unavailable: {exc}"
    except Exception as exc:
        return f"tensorflow unavailable during seed setup: {exc}"


def required_columns(base_config):
    return ["name", base_config["target"]] + list(base_config["features"])


def load_input_table(path, base_config):
    df = pd.read_csv(path)
    req = required_columns(base_config)
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != 917:
        raise ValueError(f"Expected 917 rows; observed {len(df)}")
    if not df["name"].is_unique:
        raise ValueError("Pair names are not unique")
    numeric = [base_config["target"]] + list(base_config["features"])
    if df[numeric].isna().any().any():
        raise ValueError("Missing values found in target or feature columns")
    if not np.isfinite(df[numeric].to_numpy(float)).all():
        raise ValueError("Non-finite values found in target or feature columns")
    return df


def make_outer_folds(n_rows, run_config):
    folds = []
    cv = run_config["outer_cv"]
    for seed in cv["seeds"]:
        kf = KFold(n_splits=cv["n_splits"], shuffle=cv["shuffle"], random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_rows)), start=1):
            folds.append((seed, fold, train_idx, val_idx))
    return folds


def make_inner_oof_base_predictions(X, y, base_config, run_config):
    oof = np.full(len(y), np.nan)
    inner_seed = run_config.get("_current_outer_seed")
    inner = KFold(
        n_splits=run_config["inner_cv"]["n_splits"],
        shuffle=run_config["inner_cv"]["shuffle"],
        random_state=inner_seed,
    )
    for train_idx, val_idx in inner.split(X):
        pred, _ = fit_base_predict(X[train_idx], y[train_idx], X[val_idx], base_config, run_config)
        oof[val_idx] = pred
    if not np.isfinite(oof).all():
        raise RuntimeError("Inner-OOF base predictions incomplete")
    return oof


def _fit_one_mlp(X_train_s, y_train, X_val_s, params, seed):
    model = MLPRegressor(random_state=seed, **params)
    model.fit(X_train_s, y_train)
    return model.predict(X_val_s)


def fit_base_predict(X_train, y_train, X_val, base_config, run_config):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    params = dict(base_config["mlp"])
    seeds = params.pop("ensemble_random_seeds")
    params.pop("class", None)
    params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    n_jobs = int(run_config.get("parallelism", {}).get("sklearn_n_jobs", 1))
    if n_jobs > 1:
        from joblib import Parallel, delayed
        preds = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_one_mlp)(X_train_s, y_train, X_val_s, params, seed)
            for seed in seeds
        )
    else:
        preds = [_fit_one_mlp(X_train_s, y_train, X_val_s, params, seed) for seed in seeds]
    return np.mean(preds, axis=0), scaler


def enumerate_candidates(grid):
    candidates = []
    if grid.get("ridge", {}).get("enabled", True):
        for alpha in grid["ridge"]["alpha"]:
            candidates.append({
                "model_type": "Ridge",
                "architecture": "",
                "ridge_alpha": alpha,
                "l2": np.nan,
                "learning_rate": np.nan,
                "huber_delta": np.nan,
                "complexity": 1,
            })
    if grid.get("nn", {}).get("enabled", True):
        nn = grid["nn"]
        for arch, l2, lr, delta in product(nn["architectures"], nn["l2"], nn["learning_rate"], nn["huber_delta"]):
            candidates.append({
                "model_type": "NN",
                "architecture": tuple(arch),
                "ridge_alpha": np.nan,
                "l2": l2,
                "learning_rate": lr,
                "huber_delta": delta,
                "complexity": 10 + len(arch) + sum(arch) / 1000.0,
            })
    return candidates


def soft_gate(ocn, run_config):
    center = run_config["gate"]["center"]
    scale = run_config["gate"]["scale"]
    return 1.0 / (1.0 + np.exp(-((np.asarray(ocn, dtype=float) - center) / scale)))


def calculate_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = y - pred
    if len(y) < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "mean_residual": np.nan,
                "median_residual": np.nan, "Pearson": np.nan, "Spearman": np.nan}
    return {
        "R2": r2_score(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "MAE": mean_absolute_error(y, pred),
        "mean_residual": float(np.mean(resid)),
        "median_residual": float(np.median(resid)),
        "Pearson": stats.pearsonr(y, pred).statistic,
        "Spearman": stats.spearmanr(y, pred).statistic,
    }


def calibrate(true_residual, predicted_residual, run_config):
    eps = run_config["lambda"]["epsilon"]
    raw = float(np.dot(true_residual, predicted_residual) / (np.dot(predicted_residual, predicted_residual) + eps))
    hat = float(np.clip(raw, run_config["lambda"]["clip_min"], run_config["lambda"]["clip_max"]))
    cap = float(np.percentile(np.abs(true_residual), run_config["residual_cap_percentile"]))
    return raw, hat, cap


def apply_hybrid(base_pred, residual_pred, ocn, lambda_hat, cap, run_config):
    clipped = np.clip(residual_pred, -cap, cap)
    correction = lambda_hat * soft_gate(ocn, run_config) * clipped
    return base_pred + correction, clipped, correction


def build_nn(input_dim, candidate):
    import tensorflow as tf
    from tensorflow.keras import callbacks as C
    from tensorflow.keras import layers as L
    from tensorflow.keras import optimizers as O
    from tensorflow.keras import regularizers as R

    inp = L.Input(shape=(input_dim,))
    h = inp
    for width in candidate["architecture"]:
        h = L.Dense(int(width), activation="tanh", kernel_regularizer=R.l2(float(candidate["l2"])))(h)
    out = L.Dense(1)(h)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=O.Adam(learning_rate=float(candidate["learning_rate"])),
        loss=tf.keras.losses.Huber(delta=float(candidate["huber_delta"])),
        weighted_metrics=[],
    )
    return model


def train_residual_model(candidate, X_fit, y_fit, sample_weight, run_config, seed):
    if candidate["model_type"] == "Ridge":
        model = Ridge(alpha=float(candidate["ridge_alpha"]))
        model.fit(X_fit, y_fit, sample_weight=sample_weight)
        return model, {"epochs_run": 0, "best_epoch": 0, "final_training_loss": np.nan, "final_validation_loss": np.nan}

    status = set_all_seeds(seed)
    from tensorflow.keras import callbacks as C
    model = build_nn(X_fit.shape[1], candidate)
    tf_conf = run_config["tensorflow"]
    batch = tf_conf["batch_size_small"] if len(y_fit) < tf_conf["large_n_threshold"] else tf_conf["batch_size_large"]
    callbacks = [
        C.EarlyStopping(patience=tf_conf["patience"], restore_best_weights=True, monitor="val_loss")
    ]
    if tf_conf.get("reduce_lr_on_plateau", True):
        callbacks.append(C.ReduceLROnPlateau(
            patience=tf_conf["reduce_lr_patience"],
            factor=tf_conf["reduce_lr_factor"],
            monitor="val_loss",
        ))
    hist = model.fit(
        X_fit,
        y_fit,
        validation_split=tf_conf["validation_split"],
        epochs=tf_conf["max_epochs"],
        batch_size=batch,
        sample_weight=sample_weight,
        callbacks=callbacks,
        verbose=0,
    )
    val_loss = hist.history.get("val_loss", [np.nan])
    loss = hist.history.get("loss", [np.nan])
    return model, {
        "epochs_run": len(loss),
        "best_epoch": int(np.nanargmin(val_loss) + 1) if len(val_loss) else np.nan,
        "final_training_loss": float(loss[-1]),
        "final_validation_loss": float(val_loss[-1]),
        "determinism_status": status,
    }


def predict_residual_model(model, X):
    pred = model.predict(X)
    if hasattr(pred, "ravel"):
        return np.asarray(pred).ravel()
    return np.asarray(pred, dtype=float)


def completed_fold_path(output_dir, outer_seed, outer_fold):
    return Path(output_dir) / "checkpoints" / f"outer_seed_{outer_seed}_fold_{outer_fold}.done"
