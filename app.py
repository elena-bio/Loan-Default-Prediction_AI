# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO

st.set_page_config(page_title="Loan Default Demo", layout="wide")

st.title("Loan Default Prediction — Demo")
st.markdown(
    "Upload a CSV with the same feature columns used for training (see README). "
    "This demo loads the trained pipeline and runs the same simple feature-engineering before prediction."
)

@st.cache_data
def load_model(path="best_loan_model.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)

def extract_expected_columns_from_pipeline(pipe):
    """Try to extract the numeric and categorical column names the pipeline expects.
       Returns a set of column names (may be empty on failure)."""
    cols = []
    try:
        preproc = pipe.named_steps.get("preproc", None)
        if preproc is None:
            return set()
        # preproc.transformers_ is list of (name, transformer, columns)
        for t in preproc.transformers_:
            # t can be ('num', Pipeline(...), [col list]) or similar
            if len(t) >= 3:
                cols_part = t[2]
                # columns may be slice, list, array-like
                try:
                    for c in cols_part:
                        cols.append(c)
                except Exception:
                    # maybe a slice or other object; skip
                    pass
    except Exception:
        pass
    return set(cols)

def simple_feature_engineering(df_in, expected_numeric=None):
    """Apply same simple FE used in training: loan_to_asset, loan_amount_log, outlier flags.
       expected_numeric: iterable of numeric column names to compute outlier flags for (optional).
    """
    df = df_in.copy()

    # Ensure integers nicely parsed
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").round().astype("Int64")

    # loan_to_asset and log transform if loan_amount and asset_cost exist
    if ("loan_amount" in df.columns) and ("asset_cost" in df.columns):
        # avoid division by zero
        df["loan_to_asset"] = df["loan_amount"] / df["asset_cost"].replace(0, np.nan)
        df["loan_amount_log"] = np.log1p(df["loan_amount"].fillna(0))
    else:
        # create columns with NaN so pipeline finds them (if pipeline expects them)
        df["loan_to_asset"] = np.nan
        df["loan_amount_log"] = np.nan

    # Decide numeric columns for outlier flags
    if expected_numeric:
        num_cols = [c for c in expected_numeric if c in df.columns]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # compute IQR outlier flags for each numeric column
    for c in num_cols:
        try:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                df[f"{c}_outlier_flag"] = 0
                continue
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            df[f"{c}_outlier_flag"] = ((df[c] < low) | (df[c] > high)).astype(int)
        except Exception:
            df[f"{c}_outlier_flag"] = 0

    return df

# Load model (fail early if missing)
try:
    model = load_model("best_loan_model.joblib")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Try to get expected columns from the pipeline to help preparing input
expected_cols = extract_expected_columns_from_pipeline(model)
if expected_cols:
    st.info(f"Model pipeline expects ~{len(expected_cols)} columns (will ensure they exist before predict).")

# Sidebar: sample file option
st.sidebar.header("Demo options")
use_sample = st.sidebar.checkbox("Use sample test file from repo (test_4zJg83n.csv)", value=False)

uploaded = None
if use_sample:
    sample_path = "test_4zJg83n.csv"
    if os.path.exists(sample_path):
        uploaded = sample_path
    else:
        st.sidebar.warning("Sample test file not found in repo.")
else:
    uploaded = st.file_uploader("Upload a CSV file (test)", type=["csv"])

# Load dataframe
df = None
if uploaded:
    if isinstance(uploaded, str):
        # path to local sample
        df = pd.read_csv(uploaded)
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            st.stop()

if df is None:
    st.info("Upload a CSV to run predictions (or use the sample test file in the repo).")
else:
    st.write("### Input preview", df.head(10))

    # Apply feature engineering (use expected numeric cols if any)
    df_fe = simple_feature_engineering(df, expected_numeric=[c for c in expected_cols if c in df.columns] or None)

    # Ensure pipeline expected columns exist in df_fe (add missing columns as NaN)
    missing = [c for c in expected_cols if c not in df_fe.columns]
    if missing:
        st.warning(f"The model expects these columns but they are missing from the uploaded file: {missing}")
        # create them with NaN so pipeline can accept the dataframe
        for c in missing:
            df_fe[c] = np.nan

    # Re-order or select columns according to model if needed (not required for sklearn pipelines that select columns inside)
    # Predict
    try:
        preds = model.predict(df_fe)
        try:
            probs = model.predict_proba(df_fe)[:, 1]
        except Exception:
            probs = np.zeros(len(preds))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    out = df.copy()
    out["loan_default_pred"] = preds
    out["probability"] = probs

    st.write("### Predictions (first rows)", out.head(20))

    # Download button
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    # Optional: show basic aggregate
    st.write("### Summary")
    st.write(f"Predicted defaults: {int(out['loan_default_pred'].sum())} / {len(out)}")
    st.bar_chart(out["loan_default_pred"].value_counts().sort_index())

    # Save predictions locally (optional)
    if st.button("Save predictions to submission.csv (repo)"):
        out.to_csv("submission.csv", index=False)
        st.success("Saved to submission.csv in repository folder.")
