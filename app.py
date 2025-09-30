"""
Streamlit UI for two-stage weekly predictor (mock) — updated with 'use existing preds.csv' checkbox
Behavior:
 - If "Use existing preds.csv if found" is checked and a ./preds.csv file exists, the app will load & display it.
 - Otherwise the app will generate a new preds.csv by calling the prediction logic (in-process if possible, else subprocess).
"""
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import subprocess
import os
import importlib
import io
import traceback
from pathlib import Path

# ---------- CONFIG ----------
SCRIPT_MODULE = "predict"   # python module name (predict.py)
SCRIPT_PATH = "predict.py"  # fallback subprocess script
PYTHON_EXEC = os.environ.get("PYTHON_EXECUTABLE", "python")
DEFAULT_THRESHOLD = 0.5
SEQ_LEN = 12
DEFAULT_PREDS_FILENAME = "preds.csv"
# ----------------------------

st.set_page_config(page_title="Demand Forecast Mock", layout="wide")
st.title("Demand Forecast — Week 3 Mock UI")

st.markdown(
    f"""
    **Hướng dẫn nhanh**  
    - Upload file weekly-features (đầu vào giống file dùng để huấn luyện).  
    - Chọn `Use existing `{DEFAULT_PREDS_FILENAME}` if found` để dùng file `{DEFAULT_PREDS_FILENAME}` trong thư mục hiện tại (nếu có).  
    - Nếu không có file hoặc không muốn dùng file cũ, app sẽ tạo file mới bằng cách gọi script dự đoán predict.py.
    """
)

st.subheader("Chức năng chính")
uploaded = st.file_uploader("Upload file features CSV", type=["csv"])
threshold = st.slider("Threshold dùng cho việc dự đoán", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01)
use_existing = st.checkbox(f"Sử dụng file `{DEFAULT_PREDS_FILENAME}` nếu có (tìm trong folder hiện tại)", value=True)
run_button = st.button("Run")

st.markdown("---")

# Column explanation expander
with st.expander(f"Giải thích các cột trong file `{DEFAULT_PREDS_FILENAME}` (bấm để mở)"):
    st.markdown("""
    - **product_sku** — Mã SKU sản phẩm.
    - **week_start** — Ngày bắt đầu tuần (ISO date), tức tuần mà dự đoán cho tuần kế tiếp.  
    - **p_pos** — Xác suất (0 - 1) của 'có bán > 0' trong tuần tiếp theo (classifier output).
    - **reg_pred_raw** — Dự đoán số lượng có thể bán được trong tuần tới (regressor output).
    - **expected** — Kết hợp giữa `p_pos * reg_pred_raw` dùng để biết được sản phẩm nào cần được nhập hàng nhiều hơn (chỉ số cao hơn).  
    - **thresholded** — Nếu `p_pos >= threshold` thì bằng `reg_pred_raw`, ngược lại 0 (ứng với chiến lược của threshold).
    """)
    st.caption("Lưu ý: tên cột phải khớp với file xuất từ script predict.py")

log_area = st.empty()
out_area = st.empty()


# ---------- helper functions ----------
def try_import_predict_module():
    try:
        cwd = os.getcwd()
        if cwd not in os.sys.path:
            os.sys.path.insert(0, cwd)
        mod = importlib.import_module(SCRIPT_MODULE)
        return mod
    except Exception:
        return None


def run_predict_subprocess(input_path: str, out_path: str, threshold: float = DEFAULT_THRESHOLD, timeout=180):
    cmd = [PYTHON_EXEC, SCRIPT_PATH, "--input", input_path, "--out", out_path, "--threshold", str(threshold)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), timeout=timeout)
    return proc.returncode == 0, proc.stdout, proc.stderr


def predict_inprocess_from_df(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD):
    mod = try_import_predict_module()
    if mod is None:
        raise ImportError(f"Không thể import module '{SCRIPT_MODULE}' với lỗi: {os.getcwd()}")

    clf, reg, scaler, le = mod.safe_load_artifacts()

    feature_cols = [
        "sold_sum", "avg_price_filled", "orders_sum", "is_observed", "days_with_sales",
        "lag_1w", "lag_4w", "lag_12w", "roll4_mean", "roll12_mean",
        "woy_sin", "woy_cos", "month", "quarter",
        "price_change_pct", "promo_flag", "price_missing"
    ]

    X_raw, prod_list, meta_df = mod.build_sequences_from_weekly_csv(df, feature_cols, seq_len=SEQ_LEN)
    if X_raw.shape[0] == 0:
        return pd.DataFrame()
    prod_ids = mod.map_skuprod_to_ids(prod_list, le)
    preds_df = mod.predict_from_arrays(clf, reg, scaler, X_raw, prod_ids, meta_df, threshold=threshold)
    return preds_df


def load_csv_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file, parse_dates=["week_start"])
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


# ---------- Main actions ----------
def try_use_existing_preds():
    """
    If use_existing checked and ./preds.csv exists, load and return it.
    """
    p = Path(DEFAULT_PREDS_FILENAME)
    if p.exists():
        try:
            df = pd.read_csv(p, parse_dates=["week_start"])
            return df
        except Exception:
            # try without parsing dates
            df = pd.read_csv(p)
            return df
    return None


def run_batch(uploaded_file, threshold_val, use_existing_flag):
    # if checkbox and file exists -> simply load and show
    if use_existing_flag:
        existing = try_use_existing_preds()
        if existing is not None:
            log_area.text(f"Tìm thấy file `{DEFAULT_PREDS_FILENAME}`, sử dụng file này thay vì tạo mới. Rows: {len(existing)}")
            out_area.dataframe(existing.head(200))
            st.download_button(f"Download file `{DEFAULT_PREDS_FILENAME}`", existing.to_csv(index=False).encode("utf-8"), file_name=DEFAULT_PREDS_FILENAME, mime="text/csv")
            return

    # else generate new predictions
    if uploaded_file is None:
        st.warning("Cần upload file features CSV để thực hiện dự đoán.")
        return

    df = load_csv_from_upload(uploaded_file)
    if df is None or df.shape[0] == 0:
        st.error("File CSV không hợp lệ.")
        return

    if os.path.exists(DEFAULT_PREDS_FILENAME):
        try:
            os.remove(DEFAULT_PREDS_FILENAME)
            st.info(f"Xóa file `{DEFAULT_PREDS_FILENAME}` cũ.")
        except Exception as ex:
            st.warning(f"Không thể xóa file `{DEFAULT_PREDS_FILENAME}`: {ex}")

    log = []
    log.append(f"Input rows: {len(df)}")

    # Try in-process first
    mod = try_import_predict_module()
    if mod is not None:
        try:
            preds_df = predict_inprocess_from_df(df, threshold=threshold_val)
            if preds_df is None or preds_df.shape[0] == 0:
                st.info("Không thể build sequences từ input. Lỗi file CSV.")
                log_area.text("\n".join(log))
                return

            preds_df.to_csv(DEFAULT_PREDS_FILENAME, index=False)
            out_area.dataframe(preds_df.head(200))
            st.download_button("Download file CSV", preds_df.to_csv(index=False).encode("utf-8"), file_name=DEFAULT_PREDS_FILENAME, mime="text/csv")
            log.append(f"Dự đoán thành công. Lưu tại `./{DEFAULT_PREDS_FILENAME}`")
            log_area.text("\n".join(log))
            return
        except Exception:
            log.append("Dự đoán thất bại.")
            log.append(traceback.format_exc())

    # Subprocess fallback
    log.append("Running prediction via subprocess (fallback).")
    tmpdir = tempfile.mkdtemp(prefix="pred_mock_")
    input_path = os.path.join(tmpdir, "input.csv")
    out_path = os.path.join(tmpdir, f"{DEFAULT_PREDS_FILENAME}")
    df.to_csv(input_path, index=False)
    log.append(f"Wrote temporary input to {input_path}")

    try:
        ok, sout, serr = run_predict_subprocess(input_path, out_path, threshold=threshold_val, timeout=300)
        log.append("=== subprocess stdout ===")
        log.append(sout)
        log.append("=== subprocess stderr ===")
        log.append(serr)
        if not ok:
            log_area.text("\n".join(log))
            st.error("Subprocess prediction failed — see logs above.")
            return

        if not os.path.exists(out_path):
            log.append("Subprocess finished but output file missing.")
            log_area.text("\n".join(log))
            st.error("Prediction finished but output CSV not found.")
            return

        preds_df = pd.read_csv(out_path, parse_dates=["week_start"])

        preds_df.to_csv(DEFAULT_PREDS_FILENAME, index=False)
        out_area.dataframe(preds_df.head(200))
        st.download_button("Download predictions CSV", preds_df.to_csv(index=False).encode("utf-8"), file_name=DEFAULT_PREDS_FILENAME, mime="text/csv")
        log.append(f"Subprocess prediction succeeded. Copied to ./{DEFAULT_PREDS_FILENAME}")
        log_area.text("\n".join(log))
    except subprocess.TimeoutExpired:
        log.append("Subprocess timed out.")
        log_area.text("\n".join(log))
        st.error("Prediction timed out.")
    except Exception:
        log.append("Error running subprocess:")
        log.append(traceback.format_exc())
        log_area.text("\n".join(log))
        st.error("Error running prediction (see logs).")


# Hook up UI buttons
if run_button:
    run_batch(uploaded, threshold, use_existing)
