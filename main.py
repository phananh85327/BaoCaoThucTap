#!/usr/bin/env python3
"""
train_two_stage_weekly.py (fixed)

Two-stage LSTM pipeline on weekly features:
 - Stage 1: classifier for sale/no-sale next week
 - Stage 2: regressor for log1p(sales) trained on positive weeks

This version registers small custom layers NotEqual and Any so saved models
can be loaded later (predict script must also define/register them).
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics
from tensorflow.keras.utils import register_keras_serializable
import matplotlib.pyplot as plt
import joblib

# ---------------- CONFIG ----------------
CSV_PATH = "Order.all.weekly_build_features.csv"
SEQ_LEN = 12               # history length in weeks
FORECAST_HORIZON = 1       # 1-week ahead
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 64
EPOCHS = 80
RANDOM_SEED = 42
POS_WEIGHT = 4.0           # classifier sample weight for positive class (can tune)
LEARNING_RATE = 1e-3
EMBED_DIM = 8
LSTM_UNITS = 64
MIN_WEEKS_PER_SKU = 8      # filter short series
THRESH = 0.5               # threshold for thresholded inference
SAVE_PREFIX = "two_stage_weekly"
# ----------------------------------------

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------ Custom layers ------------------
@register_keras_serializable(package="custom_layers")
class NotEqual(layers.Layer):
    """
    Elementwise not-equal to `value` (default 0). Returns boolean tensor of same shape.
    Used to create a boolean mask of non-zero features per timestep.
    """
    def __init__(self, value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call(self, inputs):
        # inputs: (batch, time, features) -> boolean mask
        return tf.not_equal(inputs, tf.cast(self.value, inputs.dtype))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"value": self.value})
        return cfg

@register_keras_serializable(package="custom_layers")
class Any(layers.Layer):
    """
    Reduce-any along the last axis (features) returning per-timestep boolean mask.
    Example: inputs boolean (batch, time, features) -> output (batch, time)
    """
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_any(inputs, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg

# ---------------- utilities ----------------
def load_weekly_df(path):
    df = pd.read_csv(path, parse_dates=["week_start"])
    df = df.sort_values(["product_sku", "week_start"]).reset_index(drop=True)
    return df

def prepare_weekly(df):
    # required columns (if missing, create defaults)
    expected = [
        "product_sku", "week_start", "sold_sum", "avg_price", "orders_sum",
        "is_observed", "days_with_sales", "lag_1w", "lag_4w", "lag_12w",
        "roll4_mean", "roll12_mean", "woy_sin", "woy_cos", "month", "quarter",
        "price_change_pct", "promo_flag", "price_missing"
    ]
    for c in expected:
        if c not in df.columns:
            # numeric defaults
            if c in ("product_sku", "week_start"):
                df[c] = df.get(c, 0)
            else:
                df[c] = 0.0

    # avg_price: forward-fill per product for continuity, then fill remaining with 0
    df["avg_price_filled"] = df.groupby("product_sku")["avg_price"].ffill().fillna(0.0)

    # target: next-week sold_sum
    df["target"] = df.groupby("product_sku")["sold_sum"].shift(-FORECAST_HORIZON)

    # drop rows without target (end-of-series)
    df = df[~df["target"].isna()].copy()

    # optionally filter short series
    counts = df.groupby("product_sku").size()
    good = counts[counts >= MIN_WEEKS_PER_SKU].index
    df = df[df["product_sku"].isin(good)].copy()

    return df

def make_sequences(df, seq_len=SEQ_LEN):
    seq_cols = [
        "sold_sum", "avg_price_filled", "orders_sum", "is_observed", "days_with_sales",
        "lag_1w", "lag_4w", "lag_12w", "roll4_mean", "roll12_mean",
        "woy_sin", "woy_cos", "month", "quarter",
        "price_change_pct", "promo_flag", "price_missing"
    ]

    # ensure columns exist
    for c in seq_cols:
        if c not in df.columns:
            df[c] = 0.0

    # integer encode products
    le = LabelEncoder()
    df["prod_id"] = le.fit_transform(df["product_sku"])

    feature_cols = seq_cols

    X_list = []
    prod_list = []
    y_list = []
    y_raw_list = []
    dates = []
    skus = []

    grouped = df.groupby("prod_id")
    for prod_id, g in grouped:
        g = g.sort_values("week_start").reset_index(drop=True)
        feats = g[feature_cols].values
        targets = g["target"].values  # raw counts
        n = len(g)
        for i in range(seq_len - 1, n):
            start = i - seq_len + 1
            end = i + 1
            X_list.append(feats[start:end])
            prod_list.append(prod_id)
            y_list.append(1 if targets[i] > 0 else 0)       # classifier target
            y_raw_list.append(targets[i])                  # regressor raw target (may be 0)
            dates.append(g.loc[i, "week_start"])
            skus.append(g.loc[i, "product_sku"])

    X = np.stack(X_list) if len(X_list) > 0 else np.empty((0, seq_len, len(feature_cols)))
    prod_ids = np.array(prod_list, dtype=np.int32)
    y_bin = np.array(y_list, dtype=np.int32)
    y_raw = np.array(y_raw_list, dtype=np.float32)
    dates = np.array(dates)
    skus = np.array(skus)

    return X, prod_ids, y_bin, y_raw, dates, skus, le, feature_cols

def scale_X(X_train, X_val, X_test):
    ns, T, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    def trans(X):
        return scaler.transform(X.reshape(-1, F)).reshape(X.shape)
    return trans(X_train), trans(X_val), trans(X_test), scaler

# ---------------- models (use custom layers to compute mask passed to LSTM) ----------------
def build_shared_lstm(seq_len, n_feats, n_products, embed_dim=EMBED_DIM, lstm_units=LSTM_UNITS):
    """Return factory components for building classifier or regressor that use embedding + LSTM,
       compute explicit mask via NotEqual + Any and pass it to LSTM to match earlier model structure."""
    seq_input = layers.Input(shape=(seq_len, n_feats), name="history")
    prod_input = layers.Input(shape=(), dtype="int32", name="prod_id")

    # compute mask: elementwise not-equal to zero -> reduce-any across features -> (batch, seq_len) boolean
    mask_bool = NotEqual()(seq_input)   # (batch, seq_len, features) boolean
    mask_any = Any(axis=-1)(mask_bool)  # (batch, seq_len) boolean

    # also apply standard Masking (so zeros become masked values too)
    x_masked = layers.Masking(mask_value=0.0)(seq_input)

    # pass mask explicitly into LSTM
    x = layers.LSTM(lstm_units, return_sequences=False)(x_masked, mask=mask_any)

    emb = layers.Embedding(input_dim=n_products, output_dim=embed_dim, name="prod_emb")(prod_input)
    emb = layers.Flatten()(emb)

    h = layers.Concatenate()([x, emb])
    return seq_input, prod_input, h

def build_classifier_model(seq_len, n_feats, n_products):
    seq_input, prod_input, h = build_shared_lstm(seq_len, n_feats, n_products)
    x = layers.Dense(64, activation="relu")(h)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_pos")(x)
    model = models.Model(inputs=[seq_input, prod_input], outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=losses.BinaryCrossentropy(),
                  metrics=[metrics.AUC(name="auc"), metrics.Precision(name="precision"), metrics.Recall(name="recall")])
    return model

def build_regressor_model(seq_len, n_feats, n_products):
    seq_input, prod_input, h = build_shared_lstm(seq_len, n_feats, n_products)
    x = layers.Dense(64, activation="relu")(h)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="linear", name="y_log1p")(x)  # predict log1p(sales)
    model = models.Model(inputs=[seq_input, prod_input], outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=losses.MeanSquaredError(),
                  metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])
    return model

# ---------------- evaluation helpers ----------------
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}

def predict_and_eval(classifier, regressor, X_test_s, prod_test, y_raw_test, df_test_meta, thresh=THRESH):
    p_pos = classifier.predict({"history": X_test_s, "prod_id": prod_test}).ravel()
    reg_log = regressor.predict({"history": X_test_s, "prod_id": prod_test}).ravel()
    reg_raw = np.expm1(reg_log)
    reg_raw = np.maximum(0.0, reg_raw)
    expected = p_pos * reg_raw
    thresholded = (p_pos >= thresh).astype(float) * reg_raw
    baseline = df_test_meta["lag_1w"].values
    results = {}
    results["expected"] = evaluate_regression(y_raw_test, expected)
    results["thresholded"] = evaluate_regression(y_raw_test, thresholded)
    results["persistence"] = evaluate_regression(y_raw_test, baseline)
    mask_pos = y_raw_test > 0
    if mask_pos.sum() > 0:
        results["expected_pos"] = evaluate_regression(y_raw_test[mask_pos], expected[mask_pos])
        results["thresholded_pos"] = evaluate_regression(y_raw_test[mask_pos], thresholded[mask_pos])
        results["persistence_pos"] = evaluate_regression(y_raw_test[mask_pos], baseline[mask_pos])
    else:
        results["expected_pos"] = results["thresholded_pos"] = results["persistence_pos"] = None
    results["classifier_auc"] = roc_auc_score((y_raw_test > 0).astype(int), p_pos)
    return results, p_pos, reg_raw

# ---------------- main ----------------
def main():
    print("Loading weekly CSV...")
    df = load_weekly_df(CSV_PATH)
    print("Preparing dataset...")
    df = prepare_weekly(df)

    print("Building sequences...")
    X, prod_ids, y_bin, y_raw, dates, skus, le, feature_cols = make_sequences(df, seq_len=SEQ_LEN)
    print(f"Sequences: {len(X)}  | features per step: {len(feature_cols)}")

    if len(X) == 0:
        raise SystemExit("No sequences generated. Check input / sequence length / MIN_WEEKS_PER_SKU.")

    # split into train/val/test
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=VAL_SIZE/(1 - TEST_SIZE), random_state=RANDOM_SEED)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    prod_train, prod_val, prod_test = prod_ids[train_idx], prod_ids[val_idx], prod_ids[test_idx]
    y_bin_train, y_bin_val, y_bin_test = y_bin[train_idx], y_bin[val_idx], y_bin[test_idx]
    y_raw_train, y_raw_val, y_raw_test = y_raw[train_idx], y_raw[val_idx], y_raw[test_idx]

    # scale features
    X_train_s, X_val_s, X_test_s, scaler = scale_X(X_train, X_val, X_test)
    n_products = len(le.classes_)
    n_feats = X_train_s.shape[2]

    # classifier
    clf = build_classifier_model(SEQ_LEN, n_feats, n_products)
    clf.summary()

    sample_weight_clf = np.ones(len(y_bin_train), dtype=np.float32)
    sample_weight_clf[y_bin_train == 1] = POS_WEIGHT

    cb_clf = [callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
              callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)]

    print("Training classifier...")
    hist_clf = clf.fit({"history": X_train_s, "prod_id": prod_train},
                       y_bin_train,
                       validation_data=({"history": X_val_s, "prod_id": prod_val}, y_bin_val),
                       sample_weight=sample_weight_clf,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       callbacks=cb_clf,
                       verbose=2)

    p_val = clf.predict({"history": X_val_s, "prod_id": prod_val}).ravel()
    try:
        auc_val = roc_auc_score(y_bin_val, p_val)
    except Exception:
        auc_val = None
    print("Classifier val AUC:", auc_val)

    # regressor on positives
    pos_train_mask = y_raw_train > 0
    pos_val_mask = y_raw_val > 0

    if pos_train_mask.sum() < 10:
        print("Warning: too few positive examples for regressor training. Regressor may not learn well.")

    X_reg_train = X_train_s[pos_train_mask]
    prod_reg_train = prod_train[pos_train_mask]
    y_reg_train = np.log1p(y_raw_train[pos_train_mask])  # train on log1p

    X_reg_val = X_val_s[pos_val_mask]
    prod_reg_val = prod_val[pos_val_mask]
    y_reg_val = np.log1p(y_raw_val[pos_val_mask])

    reg = build_regressor_model(SEQ_LEN, n_feats, n_products)
    reg.summary()

    cb_reg = [callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
              callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)]

    print("Training regressor on positive weeks...")
    if len(X_reg_train) > 0:
        hist_reg = reg.fit({"history": X_reg_train, "prod_id": prod_reg_train},
                           y_reg_train,
                           validation_data=({"history": X_reg_val, "prod_id": prod_reg_val}, y_reg_val) if len(X_reg_val) > 0 else None,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           callbacks=cb_reg,
                           verbose=2)
    else:
        print("Skipping regressor training because no positive training examples.")

    # prepare test meta
    df_test_meta = pd.DataFrame({
        "product_sku": skus[test_idx],
        "week_start": dates[test_idx],
        "sold_sum": y_raw_test,
        "lag_1w": [float(x[-1, feature_cols.index("lag_1w")]) if "lag_1w" in feature_cols else 0.0 for x in X_test]
    })

    print("Predicting and evaluating on test set...")
    results, p_pos_test, reg_raw_test = predict_and_eval(clf, reg, X_test_s, prod_test, y_raw_test, df_test_meta, thresh=THRESH)

    print("\n=== Results (test) ===")
    print("Classifier AUC (test):", results.get("classifier_auc"))
    print("Persistence baseline:", results["persistence"])
    print("Expected combination (p * reg):", results["expected"])
    print("Thresholded combination (threshold=%.2f):" % THRESH, results["thresholded"])
    print("Positive-only metrics (if any):")
    print(" expected_pos:", results["expected_pos"])
    print(" thresholded_pos:", results["thresholded_pos"])
    print(" persistence_pos:", results["persistence_pos"])

    # Save models & artifacts (save both .keras and .h5 for compatibility)
    print("Saving models and artifacts...")
    keras_clf_path = SAVE_PREFIX + "_classifier.keras"
    keras_reg_path = SAVE_PREFIX + "_regressor.keras"
    h5_clf_path = SAVE_PREFIX + "_classifier.h5"
    h5_reg_path = SAVE_PREFIX + "_regressor.h5"

    # Save native keras format
    clf.save(keras_clf_path)
    reg.save(keras_reg_path)
    # Also save HDF5 versions for older scripts that expect .h5 (note: custom layers still required at load)
    try:
        clf.save(h5_clf_path)
        reg.save(h5_reg_path)
    except Exception:
        # h5 saving sometimes warns; it's optional
        print("Warning: saving .h5 failed or not recommended on this TF version; .keras files saved.")

    joblib.dump(scaler, SAVE_PREFIX + "_scaler.pkl")
    joblib.dump(le, SAVE_PREFIX + "_labelencoder.pkl")

    # Save test predictions CSV
    out_df = pd.DataFrame({
        "product_sku": skus[test_idx],
        "week_start": dates[test_idx],
        "y_true": y_raw_test,
        "p_pos": p_pos_test,
        "reg_pred_raw": reg_raw_test,
        "expected": p_pos_test * reg_raw_test,
        "thresholded": (p_pos_test >= THRESH).astype(float) * reg_raw_test,
        "lag_1w": df_test_meta["lag_1w"].values
    })
    out_df.to_csv(SAVE_PREFIX + "_test_predictions.csv", index=False)
    print("Saved test predictions to:", SAVE_PREFIX + "_test_predictions.csv")

if __name__ == "__main__":
    main()
