import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------- Config ----------
SOURCE = "query_result.csv"     # or "query_result.csv" if that’s your file name
TARGET = "class_good"               # 1 = good, 0 = bad
ARTIFACTS_DIR = Path(".")

# ---------- Load ----------
df = pd.read_csv(SOURCE)
assert TARGET in df.columns, f"Target column '{TARGET}' not found."
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int).values

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Quick model selection (fast & deterministic) ----------
# Small validation split to choose k & weighting quickly
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

grid_k = [3, 5, 7, 9, 11]
grid_w = ["uniform", "distance"]

candidates = []
for w in grid_w:
    for k in grid_k:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights=w))
        ])
        pipe.fit(X_tr, y_tr)
        y_val_pred = pipe.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        candidates.append({"k": k, "weights": w, "val_f1": f1})

sel_df = pd.DataFrame(candidates).sort_values("val_f1", ascending=False).reset_index(drop=True)
best_k = int(sel_df.iloc[0]["k"])
best_w = sel_df.iloc[0]["weights"]

# ---------- Retrain best on full training ----------
best_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k, weights=best_w))
])
best_pipe.fit(X_train, y_train)

# ---------- Evaluate on test ----------
y_pred = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:, 1]

metrics = {
    "model": "KNeighborsClassifier",
    "params": {"n_neighbors": best_k, "weights": best_w},
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "validation_top_row": sel_df.iloc[0].to_dict(),
}

# ---------- Save artifacts ----------
(ARTIFACTS_DIR / "knn_selection_results.csv").write_text(
    sel_df.to_csv(index=False), encoding="utf-8"
)

with open(ARTIFACTS_DIR / "metrics_knn.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump({
    "model": best_pipe,
    "model_name": "KNeighborsClassifier",
    "feature_columns": X.columns.tolist(),
    "positive_label_name": "good",
    "target_column": TARGET,
}, ARTIFACTS_DIR / "model_knn.pkl")

report = f"""# Model Evaluation Report – Credit Risk (KNN)

**Dataset:** {Path(SOURCE).name}  
**Target:** {TARGET} (1 = good, 0 = bad)

## Validation Selection
Best parameters: k = {best_k}, weights = {best_w}  
Val F1 (hold-out): {sel_df.iloc[0]['val_f1']:.3f}

## Test Set Performance
- Accuracy: {metrics['accuracy']:.3f}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1-score: {metrics['f1']:.3f}
- ROC AUC: {metrics['roc_auc']:.3f}
"""
(ARTIFACTS_DIR / "Model_Evaluation_Report_KNN.md").write_text(report, encoding="utf-8")

print("Done. Artifacts written:",
      "model_knn.pkl, metrics_knn.json, knn_selection_results.csv, Model_Evaluation_Report_KNN.md")