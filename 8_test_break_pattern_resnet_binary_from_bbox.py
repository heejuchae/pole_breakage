import os
import json
import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score


current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 경로 설정
# =========================
data_root = current_dir / "5. train_data"
train_dir = data_root / "train"
test_dir = data_root / "test"

train_seq = train_dir / "break_imgs_train.npy"
train_lab = train_dir / "break_labels_train.npy"
test_seq = test_dir / "break_imgs_test.npy"
test_lab = test_dir / "break_labels_test.npy"

# 8번 학습 코드 기준 최종 저장 모델
model_path = current_dir / "7. resnet_runs_binary" / "checkpoints" / "final_binary.keras"
# cand 버전이면 아래처럼 교체
# model_path = current_dir / "최종모델" / "파인튜닝 후" / "cand" / "final_binary.keras"

BASE_SEED = 42
CALIBRATION_RATIO = 0.15
BATCH = 16
TARGET_PRECISION = 0.80
TARGET_RECALL = 0.90


# =========================
# 데이터 로드
# =========================
def load_data():
    if not train_seq.exists() or not train_lab.exists():
        raise FileNotFoundError(f"학습 데이터 없음: {train_seq}, {train_lab}")
    if not test_seq.exists() or not test_lab.exists():
        raise FileNotFoundError(f"테스트 데이터 없음: {test_seq}, {test_lab}")

    X = np.load(train_seq).astype(np.float32)
    y = np.load(train_lab).astype(np.float32)
    X_test = np.load(test_seq).astype(np.float32)
    y_test = np.load(test_lab).astype(np.float32)

    if y.ndim != 2 or y.shape[1] < 1:
        raise ValueError(f"train label shape 이상함: {y.shape}")
    if y_test.ndim != 2 or y_test.shape[1] < 1:
        raise ValueError(f"test label shape 이상함: {y_test.shape}")

    y_cls = y[:, 0].astype(np.float32)
    y_test_cls = y_test[:, 0].astype(np.float32)
    return X, y_cls, X_test, y_test_cls


def make_ds(X: np.ndarray, y: np.ndarray):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# =========================
# threshold 선택
# =========================
def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = TARGET_PRECISION,
    target_recall: float = TARGET_RECALL,
) -> Dict:
    y_true = y_true.astype(np.int32)

    best = None
    best_key = None
    thresholds = np.arange(0.05, 0.951, 0.005)

    for th in thresholds:
        pred = (y_prob >= th).astype(np.int32)

        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        p_hit = min(prec / max(target_precision, 1e-8), 1.0)
        r_hit = min(rec / max(target_recall, 1e-8), 1.0)

        score = (
            0.30 * p_hit +
            0.70 * r_hit -
            0.03 * abs(p_hit - r_hit)
        )

        satisfied = (prec >= target_precision) and (rec >= target_recall)

        candidate = {
            "threshold": float(th),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "score": float(score),
            "satisfied": bool(satisfied),
            "p_hit": float(p_hit),
            "r_hit": float(r_hit),
        }

        key = (
            score,
            rec,
            prec,
            -th,
        )

        if (best is None) or (key > best_key):
            best = candidate
            best_key = key

    return best


# =========================
# 평가 저장
# =========================
def evaluate_and_save(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
    threshold: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_test = make_ds(X_test, y_test)
    metrics = model.evaluate(ds_test, verbose=1, return_dict=True)

    prob = model.predict(X_test, batch_size=BATCH, verbose=0).reshape(-1)
    pred = (prob >= threshold).astype(np.int32)
    y_true = y_test.astype(np.int32)

    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred).tolist()

    auc = None
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, prob))

    result = {
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "threshold_metrics": {
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        },
        "roc_auc_sklearn": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "y_true": y_true.tolist(),
                "y_prob": prob.tolist(),
                "y_pred": pred.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return result


# =========================
# main
# =========================
def main():
    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print("model_path:", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없음: {model_path}")

    X, y, X_test, y_test = load_data()
    print("X:", X.shape, "y:", y.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # 8번 학습 코드와 동일한 calibration holdout 분리
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X,
        y,
        test_size=CALIBRATION_RATIO,
        random_state=BASE_SEED,
        stratify=y.astype(int),
    )

    print("X_fit:", X_fit.shape, "y_fit:", y_fit.shape)
    print("X_cal:", X_cal.shape, "y_cal:", y_cal.shape)

    model = keras.models.load_model(str(model_path), compile=True)

    # calibration set에서 threshold 탐색
    cal_prob = model.predict(X_cal, batch_size=BATCH, verbose=0).reshape(-1)
    th_info = find_best_threshold(
        y_true=y_cal,
        y_prob=cal_prob,
        target_precision=TARGET_PRECISION,
        target_recall=TARGET_RECALL,
    )

    print("\n===== CALIBRATION THRESHOLD =====")
    print(json.dumps(th_info, ensure_ascii=False, indent=2))

    out_dir = current_dir / f"test_binary_eval_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    result = evaluate_and_save(
        model=model,
        X_test=X_test,
        y_test=y_test,
        out_dir=out_dir,
        threshold=th_info["threshold"],
    )

    with open(out_dir / "calibration_threshold.json", "w", encoding="utf-8") as f:
        json.dump(th_info, f, ensure_ascii=False, indent=2)

    print("\n===== FINAL TEST RESULT =====")
    print("save_dir:", out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
