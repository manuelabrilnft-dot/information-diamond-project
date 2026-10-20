import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(page_title="Credit", layout="wide")

APP_DIR = Path(__file__).parent.resolve()

def find_file(*names):
    """Return first existing path among given names, checked both in CWD and app dir."""
    for n in names:
        p1 = Path(n)
        p2 = APP_DIR / n
        if p1.exists(): return p1
        if p2.exists(): return p2
    return None

@st.cache_data
def load_dataset():
    # prefer renamed 'query_result.csv'
    candidates = ["query_result.csv", "dataset.csv"]
    for c in candidates:
        p = find_file(c)
        if p:
            try:
                df = pd.read_csv(p)
                return df, p.name
            except Exception as e:
                st.error(f"Failed to read {p}: {e}")
                st.stop()
    st.error("No dataset CSV found. Add 'query_result.csv' (cleaned/encoded) next to app.py.")
    st.stop()

@st.cache_resource
def load_model():
    for name in ["model_knn.pkl", "model.pkl"]:
        p = find_file(name)
        if p:
            try:
                blob = joblib.load(p)
                if isinstance(blob, dict) and "model" in blob:
                    return {**blob, "filename": p.name}
                return {
                    "model": blob,
                    "feature_columns": None,
                    "target_column": "class_good",
                    "positive_label_name": "good",
                    "scaler": None,
                    "filename": p.name,
                }
            except Exception as e:
                st.warning(f"Model load error ({p}): {e}")
                return {}
    return {}

def list_one_hot(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    cats = [c.split(prefix + "_", 1)[1] for c in cols]
    return cols, cats

def apply_one_hot_filter(df, prefix, selected):
    if not selected: return df
    cols = [f"{prefix}_{c}" for c in selected if f"{prefix}_{c}" in df.columns]
    return df if not cols else df[df[cols].sum(axis=1) >= 1]

def to_feature_vector(row_df, feature_columns):
    return row_df.reindex(columns=feature_columns, fill_value=0)

def make_manual_input(df):
    st.markdown("#### Manual Applicant Input")
    c1, c2, c3 = st.columns(3)
    numeric = {}
    for i, col in enumerate(["credit_amount", "duration", "age", "installment_commitment"]):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            with (c1, c2, c3)[i % 3]:
                mn, mx = float(df[col].min()), float(df[col].max())
                step = max((mx - mn) / 100, 1.0)
                val = float(np.median(df[col]))
                numeric[col] = st.number_input(col, value=val, min_value=mn, max_value=mx, step=step)

    onehot_prefixes = ["purpose", "employment", "housing", "credit_history", "savings_status", "checking_status", "job"]
    choices = {}
    for p in onehot_prefixes:
        _, cats = list_one_hot(df, p)
        if cats:
            sel = st.selectbox(p.replace("_", " ").title(), ["(none)"] + cats, index=0, key=f"manual_{p}")
            choices[p] = (None if sel == "(none)" else sel)

    # Build single-row frame from column template
    single = pd.DataFrame([0] * len(df.columns), index=df.columns).T
    for k, v in numeric.items():
        if k in single.columns:
            single[k] = v
    for p, cat in choices.items():
        if cat:
            col = f"{p}_{cat}"
            if col in single.columns:
                single[col] = 1
    for tcol in ["class_good", "credit_risk"]:
        if tcol in single.columns:
            single = single.drop(columns=[tcol])
    return single

# -------- UI Header --------
st.title("Credit Risk Analysis")

# -------- Data + Model --------
df, source_name = load_dataset()
bundle = load_model()
model = bundle.get("model")
feature_columns = bundle.get("feature_columns")
target_column = bundle.get("target_column", "class_good")
model_file = bundle.get("filename")

# -------- Sidebar Filters --------
st.sidebar.header("Filters")
st.sidebar.caption(f"Source: **{source_name}** — Rows: **{len(df)}**, Cols: **{df.shape[1]}**")

for n in ["credit_amount", "duration", "age", "installment_commitment"]:
    if n in df.columns and pd.api.types.is_numeric_dtype(df[n]):
        mn, mx = float(df[n].min()), float(df[n].max())
        lo, hi = st.sidebar.slider(n.replace("_", " ").title(), mn, mx, (mn, mx))
        df = df[(df[n] >= lo) & (df[n] <= hi)]

for p in ["purpose", "employment", "housing", "credit_history", "savings_status", "checking_status", "job"]:
    _, cats = list_one_hot(df, p)
    if cats:
        sel = st.sidebar.multiselect(p.replace("_", " ").title(), cats, key=f"filter_{p}")
        df = apply_one_hot_filter(df, p, sel)

st.sidebar.success(f"Active subset: {len(df)} rows")

# -------- Tabs --------
tab_dashboard, tab_visuals, tab_predict = st.tabs(
    ["Dashboard", "Visuals", "Predict"]
)

with tab_dashboard:
    st.subheader("Overview")
    st.dataframe(df.head(500), use_container_width=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Rows (active)", len(df))
    with k2: st.metric("Columns", df.shape[1])
    if "credit_amount" in df.columns and pd.api.types.is_numeric_dtype(df["credit_amount"]):
        with k3: st.metric("Avg credit_amount", f"{df['credit_amount'].mean():.0f}")
    if "duration" in df.columns and pd.api.types.is_numeric_dtype(df["duration"]):
        with k4: st.metric("Avg duration (m)", f"{df['duration'].mean():.1f}")
    target = "class_good" if "class_good" in df.columns else ("credit_risk" if "credit_risk" in df.columns else None)
    if target:
        st.markdown("### Summary by Risk Class")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            agg = df.groupby(target)[num_cols].agg(["mean", "median", "count"]).T
            st.dataframe(agg, use_container_width=True)

with tab_visuals:
    st.subheader("Visual Insights")
    if "credit_amount" in df.columns and pd.api.types.is_numeric_dtype(df["credit_amount"]):
        st.markdown("**Credit Amount Distribution**")
        f1, a1 = plt.subplots()
        a1.hist(df["credit_amount"].dropna(), bins=30)
        a1.set_xlabel("credit_amount"); a1.set_ylabel("count")
        st.pyplot(f1)
    if "duration" in df.columns and "class_good" in df.columns:
        st.markdown("**Duration by Risk Class** (1=good, 0=bad)")
        f2, a2 = plt.subplots()
        groups = [df[df["class_good"] == t]["duration"].dropna() for t in sorted(df["class_good"].unique())]
        a2.boxplot(groups, labels=[str(t) for t in sorted(df["class_good"].unique())])
        a2.set_xlabel("Risk class"); a2.set_ylabel("duration")
        st.pyplot(f2)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        st.markdown("**Correlation Heatmap (numeric)**")
        c = num.corr(numeric_only=True)
        f3, a3 = plt.subplots()
        im = a3.imshow(c)
        a3.set_xticks(range(len(c.columns))); a3.set_xticklabels(c.columns, rotation=90, fontsize=7)
        a3.set_yticks(range(len(c.columns))); a3.set_yticklabels(c.columns, fontsize=7)
        f3.colorbar(im)
        st.pyplot(f3)

with tab_predict:
    st.subheader("Predict Credit Risk")
    if not model:
        st.info("Place **model_knn.pkl** (preferred) or **model.pkl** next to app.py to enable predictions.")
    else:
        st.caption(f"Loaded model: **{model_file}**")
        feats = feature_columns or [c for c in df.columns if c != target_column]
        mode = st.radio("Input Mode", ["Pick a row", "Manual form", "Upload CSV"], horizontal=True)

        if mode == "Pick a row":
            idx = st.number_input("Row index (from current filtered table)", 0, max(0, len(df)-1), 0)
            if len(df) > 0:
                row = df.iloc[[int(idx)]].copy()
                if target_column in row.columns:
                    row = row.drop(columns=[target_column])
                X = to_feature_vector(row, feats)
                if hasattr(model, "predict_proba"):
                    p1 = float(model.predict_proba(X)[:, 1][0]); pred = int(p1 >= 0.5)
                else:
                    pred = int(model.predict(X)[0]); p1 = float(pred)
                st.metric("Predicted class (1=good, 0=bad)", pred)
                st.write(f"Probability of class=1 (good): **{p1:.3f}**")

        elif mode == "Manual form":
            single = make_manual_input(df)
            X = to_feature_vector(single, feats)
            if hasattr(model, "predict_proba"):
                p1 = float(model.predict_proba(X)[:, 1][0]); pred = int(p1 >= 0.5)
            else:
                pred = int(model.predict(X)[0]); p1 = float(pred)
            st.metric("Predicted class (1=good, 0=bad)", pred)
            st.write(f"Probability of class=1 (good): **{p1:.3f}**")

        else:
            up = st.file_uploader("Upload CSV (columns should match training feature set)", type=["csv"])
            if up is not None:
                new_df = pd.read_csv(up)
                X2 = new_df.reindex(columns=feats, fill_value=0)
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X2)[:, 1]; pred = model.predict(X2)
                else:
                    pred = model.predict(X2); prob = np.array(pred, dtype=float)
                out = new_df.copy()
                out["pred_class"] = pred
                out["proba_good"] = prob
                st.dataframe(out.head(200), use_container_width=True)
                st.download_button("Download predictions CSV", data=out.to_csv(index=False), file_name="predictions.csv")
