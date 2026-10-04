import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 16

# 1. Measure data loading only.
started = time.perf_counter()
df = pd.read_csv("creditcard.csv")
data_load_seconds = time.perf_counter() - started

X = df.drop(columns="Class")
y = df["Class"]

# 2. 60% train / 20% validation / 20% final test, stratified by Class.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=SEED,
    stratify=y_trainval,
)

# 3. Train on CPU. Validation is used only for early stopping.
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=SEED,
    n_jobs=2,
    verbosity=-1,
)

started = time.perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(20, verbose=False)],
)
training_seconds = time.perf_counter() - started

# 4. Evaluate exactly once on the held-out test set.
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# 5. Warm-up outside of inference timing.
one_row = X_test.iloc[:1]
batch = X_test.iloc[:1000]
model.predict_proba(one_row)
model.predict_proba(batch)

def median_predict_seconds(data, repeats):
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(data)
        elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed))

single_seconds = median_predict_seconds(one_row, repeats=50)
batch_seconds = median_predict_seconds(batch, repeats=10)

best_iteration = int(model.best_iteration_ or model.n_estimators)

result = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "architecture": platform.machine(),
    "versions": {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
    "dataset_rows": int(len(df)),
    "fraud_rows": int(y.sum()),
    "seed": SEED,
    "split": {
        "train": int(len(X_train)),
        "validation": int(len(X_valid)),
        "test": int(len(X_test)),
    },
    "n_jobs": 2,
    "decision_threshold": 0.5,
    "data_load_seconds": float(data_load_seconds),
    "training_seconds": float(training_seconds),
    "best_iteration": best_iteration,
    "auc_roc": float(roc_auc_score(y_test, probabilities)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "latency_1_row_ms": float(single_seconds * 1000),
    "latency_repeats": 50,
    "batch_rows": int(len(batch)),
    "batch_repeats": 10,
    "batch_1000_rows_seconds": float(batch_seconds),
    "throughput_1000_rows_per_second": float(len(batch) / batch_seconds),
    "timing_summary": (
        "Median timing; warm-up excluded; predict_proba on pandas input."
    ),
}

Path("benchmark_result.json").write_text(
    json.dumps(result, indent=2, allow_nan=False),
    encoding="utf-8",
)
print(json.dumps(result, indent=2, allow_nan=False))
