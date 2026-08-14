#!/usr/bin/env python3
"""
Lab 16 - Cloud AI Environment Setup
Benchmark: LightGBM training + inference tren CPU node (AWS EC2 t3.medium).

Dataset: Credit Card Fraud Detection (mlg-ulb/creditcardfraud)
    284,807 giao dich, 492 gian lan (0.172%) - bai toan phan loai nhi phan mat can bang.

Cach chay:
    python3 benchmark.py
    python3 benchmark.py --data /duong/dan/creditcard.csv --output ket_qua.json

Ket qua duoc ghi ra benchmark_result.json.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import lightgbm as lgb

RANDOM_STATE = 42
TARGET_COLUMN = "Class"

# Cac vi tri thuong gap cua dataset sau khi giai nen bang Kaggle CLI
DEFAULT_DATA_PATHS = [
    os.path.expanduser("~/ml-benchmark/creditcard.csv"),
    os.path.expanduser("~/creditcard.csv"),
    "creditcard.csv",
    "data/creditcard.csv",
]


def find_dataset(explicit_path):
    """Tra ve duong dan dataset dau tien ton tai, hoac thoat voi huong dan tai."""
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        sys.exit("Khong tim thay file: {}".format(explicit_path))

    for candidate in DEFAULT_DATA_PATHS:
        if os.path.isfile(candidate):
            return candidate

    sys.exit(
        "Khong tim thay creditcard.csv. Da tim o:\n  "
        + "\n  ".join(DEFAULT_DATA_PATHS)
        + "\n\nTai dataset bang lenh:\n"
        "  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/"
    )


def collect_system_info():
    """Thu thap thong tin may de dua vao bao cao (CPU model, so core, RAM)."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "lightgbm_version": lgb.__version__,
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }

    # /proc chi ton tai tren Linux (EC2). Chay thu tren Windows se bo qua phan nay.
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["total_ram_gb"] = round(kb / 1024 / 1024, 2)
                    break
    except (OSError, ValueError, IndexError):
        pass

    return info


def measure_latency(model, single_row, n_warmup, n_runs):
    """Do do tre khi du doan 1 dong. Warmup truoc de loai nhieu cua lan goi dau."""
    for _ in range(n_warmup):
        model.predict(single_row)

    samples = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(single_row)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples = np.array(samples)
    return {
        "runs": n_runs,
        "mean_ms": round(float(samples.mean()), 4),
        "p50_ms": round(float(np.percentile(samples, 50)), 4),
        "p95_ms": round(float(np.percentile(samples, 95)), 4),
        "p99_ms": round(float(np.percentile(samples, 99)), 4),
        "min_ms": round(float(samples.min()), 4),
        "max_ms": round(float(samples.max()), 4),
    }


def measure_throughput(model, batch, n_warmup, n_runs):
    """Do thong luong khi du doan mot batch (mac dinh 1000 dong)."""
    batch_size = len(batch)

    for _ in range(n_warmup):
        model.predict(batch)

    samples = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(batch)
        samples.append(time.perf_counter() - start)

    samples = np.array(samples)
    mean_sec = float(samples.mean())
    return {
        "batch_size": batch_size,
        "runs": n_runs,
        "mean_batch_time_ms": round(mean_sec * 1000.0, 4),
        "rows_per_second": round(batch_size / mean_sec, 2),
        "per_row_ms": round(mean_sec * 1000.0 / batch_size, 6),
    }


def best_f1_threshold(y_true, y_proba):
    """Tim nguong cho F1 cao nhat - huu ich vi 0.5 hiem khi toi uu tren du lieu mat can bang."""
    best = {"threshold": 0.5, "f1": 0.0}
    for threshold in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": round(float(threshold), 2), "f1": round(float(f1), 6)}
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LightGBM tren dataset Credit Card Fraud (Lab 16)"
    )
    parser.add_argument("--data", default=None, help="Duong dan toi creditcard.csv")
    parser.add_argument(
        "--output", default="benchmark_result.json", help="File JSON ket qua"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Ty le tap test (mac dinh 0.2)"
    )
    parser.add_argument(
        "--latency-runs", type=int, default=1000, help="So lan do latency 1 dong"
    )
    parser.add_argument(
        "--throughput-runs", type=int, default=50, help="So lan do throughput 1000 dong"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LAB 16 - LIGHTGBM CPU BENCHMARK (Credit Card Fraud Detection)")
    print("=" * 70)

    system_info = collect_system_info()
    print("\n[MAY CHU]")
    for key, value in system_info.items():
        print("  {:<20} {}".format(key + ":", value))

    # ------------------------------------------------------------------
    # 1. Load dataset (do thoi gian)
    # ------------------------------------------------------------------
    data_path = find_dataset(args.data)
    print("\n[1/5] Dang load dataset: {}".format(data_path))

    load_start = time.perf_counter()
    df = pd.read_csv(data_path)
    load_time = time.perf_counter() - load_start

    if TARGET_COLUMN not in df.columns:
        sys.exit(
            "File thieu cot '{}'. Cac cot tim thay: {}".format(
                TARGET_COLUMN, list(df.columns)
            )
        )

    n_rows, n_cols = df.shape
    n_fraud = int(df[TARGET_COLUMN].sum())
    fraud_ratio = n_fraud / n_rows

    print("  Load xong trong {:.4f} giay".format(load_time))
    print("  Kich thuoc: {:,} dong x {} cot".format(n_rows, n_cols))
    print(
        "  Gian lan:   {:,} / {:,} ({:.3f}%)".format(n_fraud, n_rows, fraud_ratio * 100)
    )
    print("  Bo nho:     {:.1f} MB".format(df.memory_usage(deep=True).sum() / 1024**2))

    # ------------------------------------------------------------------
    # 2. Tach train / validation / test
    # ------------------------------------------------------------------
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # stratify de giu nguyen ty le gian lan cuc thap o moi tap
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )
    # Tach them tap validation rieng cho early stopping, KHONG dung tap test
    # (dung test de early stop se lam ro ri thong tin va thoi phong metric).
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    print("\n[2/5] Tach du lieu")
    print("  Train:      {:,} dong ({:,} gian lan)".format(len(X_train), int(y_train.sum())))
    print("  Validation: {:,} dong ({:,} gian lan)".format(len(X_val), int(y_val.sum())))
    print("  Test:       {:,} dong ({:,} gian lan)".format(len(X_test), int(y_test.sum())))

    # ------------------------------------------------------------------
    # 3. Huan luyen (do thoi gian)
    # ------------------------------------------------------------------
    # Dataset chi co 0.172% gian lan (315 positive tren 182k dong train). Voi ty le lech
    # nhu vay, learning_rate cao + num_leaves lon lam moi cay overfit vao nhum positive
    # nho xiu, boosting di sai huong va AUC validation TUT DAN theo so cay (do thuc nghiem:
    # lr=0.05/num_leaves=31 cho AUC 0.83 sau 300 cay, tham chi te hon 1 cay don le).
    # Bo tham so duoi day regularize manh hon nen mo hinh hoi tu dung: AUC ~0.979.
    hyperparams = {
        "objective": "binary",
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 50,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    print("\n[3/5] Huan luyen LGBMClassifier (early stopping 50 vong)...")
    model = lgb.LGBMClassifier(**hyperparams)

    train_start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.perf_counter() - train_start

    best_iteration = int(model.best_iteration_ or hyperparams["n_estimators"])
    print("  Training xong trong {:.4f} giay".format(train_time))
    print("  Best iteration: {} / {}".format(best_iteration, hyperparams["n_estimators"]))

    # ------------------------------------------------------------------
    # 4. Danh gia tren tap test
    # ------------------------------------------------------------------
    print("\n[4/5] Danh gia tren tap test...")

    eval_start = time.perf_counter()
    y_proba = model.predict_proba(X_test)[:, 1]
    eval_predict_time = time.perf_counter() - eval_start
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print("  AUC-ROC:           {:.6f}".format(metrics["auc_roc"]))
    print("  Average Precision: {:.6f}".format(metrics["average_precision"]))
    print("  Accuracy:          {:.6f}".format(metrics["accuracy"]))
    print("  F1-Score:          {:.6f}".format(metrics["f1_score"]))
    print("  Precision:         {:.6f}".format(metrics["precision"]))
    print("  Recall:            {:.6f}".format(metrics["recall"]))
    print(
        "  Confusion matrix:  TN={:,}  FP={}  FN={}  TP={}".format(tn, fp, fn, tp)
    )

    threshold_info = best_f1_threshold(y_test.to_numpy(), y_proba)
    print(
        "  Nguong F1 tot nhat: {} (F1={:.6f}) - so voi nguong mac dinh 0.5".format(
            threshold_info["threshold"], threshold_info["f1"]
        )
    )

    # ------------------------------------------------------------------
    # 5. Do inference latency va throughput
    # ------------------------------------------------------------------
    print("\n[5/5] Do inference latency va throughput...")

    single_row = X_test.iloc[[0]]
    latency = measure_latency(
        model, single_row, n_warmup=100, n_runs=args.latency_runs
    )
    print(
        "  Latency 1 dong:     mean {:.4f} ms | p50 {:.4f} ms | p95 {:.4f} ms".format(
            latency["mean_ms"], latency["p50_ms"], latency["p95_ms"]
        )
    )

    batch_1000 = X_test.iloc[:1000]
    throughput = measure_throughput(
        model, batch_1000, n_warmup=5, n_runs=args.throughput_runs
    )
    print(
        "  Throughput 1000 dong: {:.4f} ms/batch | {:,.0f} dong/giay".format(
            throughput["mean_batch_time_ms"], throughput["rows_per_second"]
        )
    )

    # ------------------------------------------------------------------
    # Xuat ket qua ra JSON
    # ------------------------------------------------------------------
    top_features = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )

    result = {
        "lab": "Lab 16 - Cloud AI Environment Setup",
        "task": "Credit Card Fraud Detection - LightGBM tren CPU",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system_info": system_info,
        "dataset": {
            "path": data_path,
            "rows": int(n_rows),
            "columns": int(n_cols),
            "fraud_count": n_fraud,
            "fraud_ratio": round(fraud_ratio, 6),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "timing": {
            "data_load_seconds": round(load_time, 4),
            "training_seconds": round(train_time, 4),
            "test_set_predict_seconds": round(eval_predict_time, 4),
        },
        "model": {
            "type": "LGBMClassifier",
            "hyperparameters": hyperparams,
            "best_iteration": best_iteration,
            "trees_built": int(model.booster_.num_trees()),
        },
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "confusion_matrix": confusion,
        "best_f1_threshold": threshold_info,
        "inference": {
            "latency_single_row": latency,
            "throughput_1000_rows": throughput,
        },
        "top_10_features": {
            name: int(score) for name, score in top_features.items()
        },
    }

    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)

    print("\n" + "=" * 70)
    print("BANG KET QUA (dien vao README)")
    print("=" * 70)
    summary_rows = [
        ("Thoi gian load data", "{:.4f} s".format(load_time)),
        ("Thoi gian training", "{:.4f} s".format(train_time)),
        ("Best iteration", str(best_iteration)),
        ("AUC-ROC", "{:.6f}".format(metrics["auc_roc"])),
        ("Accuracy", "{:.6f}".format(metrics["accuracy"])),
        ("F1-Score", "{:.6f}".format(metrics["f1_score"])),
        ("Precision", "{:.6f}".format(metrics["precision"])),
        ("Recall", "{:.6f}".format(metrics["recall"])),
        ("Inference latency (1 row)", "{:.4f} ms".format(latency["mean_ms"])),
        (
            "Inference throughput (1000 rows)",
            "{:.4f} ms ({:,.0f} rows/s)".format(
                throughput["mean_batch_time_ms"], throughput["rows_per_second"]
            ),
        ),
    ]
    for label, value in summary_rows:
        print("  {:<34} {}".format(label, value))

    print("\nDa ghi ket qua chi tiet ra: {}".format(os.path.abspath(args.output)))
    print("=" * 70)


if __name__ == "__main__":
    main()
