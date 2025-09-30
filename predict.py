#!/usr/bin/env python3
"""
predict_two_stage_weekly.py

Usage:
  python predict_two_stage_weekly.py --input Order.all.weekly_build_features.csv --out preds.csv
  python predict_two_stage_weekly.py --single-sku SKU123 --single-csv sku123_last12.csv --out single_pred.csv

Requirements:
  pip install pandas numpy joblib tensorflow scikit-learn
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope, register_keras_serializable
import tensorflow as tf

# ---------------- Config (match training) ----------------
MODEL_PREFIX = "two_stage_weekly"
CLASSIFIER_KERAS = MODEL_PREFIX + "_classifier.keras"
REGRESSOR_KERAS = MODEL_PREFIX + "_regressor.keras"
CLASSIFIER_H5 = MODEL_PREFIX + "_classifier.h5"
REGRESSOR_H5 = MODEL_PREFIX + "_regressor.h5"
SCALER_PATH = MODEL_PREFIX + "_scaler.pkl"
LABELENCODER_PATH = MODEL_PREFIX + "_labelencoder.pkl"
SEQ_LEN = 12
THRESH = 0.5
# --------------------------------------------------------

# ---------------- custom layers (same as training) ----------------
@register_keras_serializable(package="custom_layers")
class NotEqual(tf.keras.layers.Layer):
    def __init__(self, value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call(self, inputs):
        return tf.not_equal(inputs, tf.cast(self.value, inputs.dtype))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"value": self.value})
        return cfg

@register_keras_serializable(package="custom_layers")
class Any(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_any(inputs, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg

# ---------------- helpers ----------------
def find_model_path():
    """Prefer .keras; fallback to .h5 if .keras not found."""
    clf_path = CLASSIFIER_KERAS if os.path.exists(CLASSIFIER_KERAS) else (CLASSIFIER_H5 if os.path.exists(CLASSIFIER_H5) else None)
    reg_path = REGRESSOR_KERAS if os.path.exists(REGRESSOR_KERAS) else (REGRESSOR_H5 if os.path.exists(REGRESSOR_H5) else None)
    return clf_path, reg_path

def safe_load_artifacts():
    clf_path, reg_path = find_model_path()
    if clf_path is None or reg_path is None:
        raise FileNotFoundError("Classifier/regressor model files not found. Looked for .keras/.h5 variants.")
    if not os.path.exists(SCALER_PATH) or not os.path.exists(LABELENCODER_PATH):
        raise FileNotFoundError("Scaler or labelencoder artifacts not found.")

    # Use custom_object_scope so deser. finds NotEqual/Any
    with custom_object_scope({"NotEqual": NotEqual, "Any": Any}):
        clf = load_model(clf_path)
        reg = load_model(reg_path)

    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABELENCODER_PATH)
    return clf, reg, scaler, le

def build_sequences_from_weekly_csv(df, feature_cols, seq_len=SEQ_LEN):
    """Build sliding sequences (seq_len) from weekly features dataframe sorted by product_sku, week_start.
       Returns: X (N, seq_len, F), prod_list (len N), meta_df with product_sku & week_start (target week).
    """
    if "product_sku" not in df.columns or "week_start" not in df.columns:
        raise ValueError("Input weekly CSV must contain 'product_sku' and 'week_start' columns.")
    df = df.sort_values(["product_sku", "week_start"]).reset_index(drop=True)

    # ensure feature columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    groups = df.groupby("product_sku")
    X_list = []
    prod_list = []
    meta_rows = []
    for sku, g in groups:
        g2 = g.sort_values("week_start").reset_index(drop=True)
        feats = g2[feature_cols].values
        n = len(g2)
        if n < 1:
            continue
        # sliding windows where last row at i is the target-week snapshot
        for i in range(seq_len - 1, n):
            start = i - seq_len + 1
            end = i + 1
            X_list.append(feats[start:end])
            prod_list.append(sku)
            meta_rows.append({
                "product_sku": sku,
                "week_start": g2.loc[i, "week_start"]
            })
    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feature_cols))), [], pd.DataFrame(meta_rows)
    X = np.stack(X_list)
    return X, prod_list, pd.DataFrame(meta_rows)

def map_skuprod_to_ids(prod_list, label_encoder):
    """Map product SKU list to labelencoded ids. Unseen SKUs fall back to index 0."""
    classes = list(label_encoder.classes_)
    class_to_index = {c:i for i,c in enumerate(classes)}
    fallback = 0
    prod_ids = []
    unseen = set()
    for sku in prod_list:
        if sku in class_to_index:
            prod_ids.append(class_to_index[sku])
        else:
            prod_ids.append(fallback)
            unseen.add(sku)
    if unseen:
        warnings.warn(f"{len(unseen)} unseen SKUs encountered; falling back to index {fallback}. Examples: {list(unseen)[:5]}")
    return np.array(prod_ids, dtype=np.int32)

def predict_from_arrays(clf, reg, scaler, X_raw, prod_ids, meta_df, threshold=THRESH):
    """Scale and predict. Returns meta_df with added prediction columns."""
    if X_raw.shape[0] == 0:
        return meta_df
    n, T, F = X_raw.shape
    X_scaled = scaler.transform(X_raw.reshape(-1, F)).reshape(n, T, F)
    # predictions
    p_pos = clf.predict({"history": X_scaled, "prod_id": prod_ids}).ravel()
    reg_log = reg.predict({"history": X_scaled, "prod_id": prod_ids}).ravel()
    reg_raw = np.expm1(reg_log)
    reg_raw = np.maximum(0.0, reg_raw)
    expected = p_pos * reg_raw
    thresholded = (p_pos >= threshold).astype(float) * reg_raw
    out = meta_df.copy()
    out["p_pos"] = p_pos
    out["reg_pred_raw"] = reg_raw
    out["expected"] = expected
    out["thresholded"] = thresholded
    return out

# ---------------- command-line ----------------
def main():
    parser = argparse.ArgumentParser(description="Load two-stage weekly models and predict next-week demand.")
    parser.add_argument("--input", "-i", help="Input weekly features CSV (same format as training).", required=False)
    parser.add_argument("--single-sku", help="Product SKU for single prediction (use with --single-csv)", required=False)
    parser.add_argument("--single-csv", help="CSV with last SEQ_LEN weekly rows for single SKU (sorted ascending by week_start)", required=False)
    parser.add_argument("--out", "-o", help="Output CSV path (predictions). Default: preds.csv", default="preds.csv")
    parser.add_argument("--threshold", "-t", type=float, default=THRESH, help="Threshold for thresholded prediction (default 0.5)")
    args = parser.parse_args()

    clf, reg, scaler, le = safe_load_artifacts()
    print("Loaded classifier, regressor, scaler, labelencoder.")

    # feature columns MUST match training's sequence columns (in same order)
    feature_cols = [
        "sold_sum", "avg_price_filled", "orders_sum", "is_observed", "days_with_sales",
        "lag_1w", "lag_4w", "lag_12w", "roll4_mean", "roll12_mean",
        "woy_sin", "woy_cos", "month", "quarter",
        "price_change_pct", "promo_flag", "price_missing"
    ]

    if args.input:
        df = pd.read_csv(args.input, parse_dates=["week_start"])
        X_raw, prod_list, meta_df = build_sequences_from_weekly_csv(df, feature_cols, SEQ_LEN)
        print(f"Built {len(prod_list)} sequences from input file.")
        if len(prod_list) == 0:
            print("No sequences created. Exiting.")
            return
        prod_ids = map_skuprod_to_ids(prod_list, le)
        preds_df = predict_from_arrays(clf, reg, scaler, X_raw, prod_ids, meta_df, threshold=args.threshold)
        preds_df.to_csv(args.out, index=False)
        print("Saved predictions to", args.out)
        print(preds_df.head(12))

    elif args.single_sku and args.single_csv:
        df_single = pd.read_csv(args.single_csv, parse_dates=["week_start"])
        df_single = df_single.sort_values("week_start").reset_index(drop=True)
        # if fewer than SEQ_LEN rows we'll pad with zeros on top
        feats = df_single[feature_cols].values if feature_cols[0] in df_single.columns else df_single.values
        n_feats = feats.shape[1]
        if feats.shape[0] > SEQ_LEN:
            feats = feats[-SEQ_LEN:, :]
        elif feats.shape[0] < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - feats.shape[0], n_feats), dtype=float)
            feats = np.vstack([pad, feats])
        seq_raw = feats.reshape(1, SEQ_LEN, n_feats)
        sku = args.single_sku
        prod_id = map_skuprod_to_ids([sku], le)
        meta = pd.DataFrame([{"product_sku": sku, "week_start": df_single["week_start"].iloc[-1] if len(df_single)>0 else pd.NaT}])
        preds_df = predict_from_arrays(clf, reg, scaler, seq_raw, prod_id, meta, threshold=args.threshold)
        preds_df.to_csv(args.out, index=False)
        print("Saved single-SKU prediction to", args.out)
        print(preds_df.T)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
