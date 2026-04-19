import os
import json
import datetime
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# ============================================================
# 설정
# ============================================================
BASE_SEED = 42
BATCH = 16
EPOCHS = 150

CALIBRATION_RATIO = 0.15
FINAL_VAL_RATIO = 0.10

TARGET_PRECISION = 0.80
TARGET_RECALL = 0.90

HEAD_LR = 1e-3
FT_LR = 1e-5

HEAD_LR = 1e-3
FT_LR = 1e-5

HEAD_EPOCHS = 20
FT_EPOCHS = 40

DROPOUT = 0.3
USE_MIXED_PRECISION = False
FREEZE_BN_IN_FT = True
USE_AUGMENTATION = False
AUG_NOISE_STD = 0.02
AUG_BRIGHTNESS_DELTA = 0.05
AUG_CONTRAST_LOWER = 0.95
AUG_CONTRAST_UPPER = 1.05

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
data_root = current_dir / "5. train_data"
train_dir = data_root / "train"
test_dir = data_root / "test"
run_dir = current_dir / "7. resnet_runs_binary"
ckpt_dir = run_dir / "checkpoints"

# 입력 데이터
train_seq = train_dir / "break_imgs_train.npy"
train_lab = train_dir / "break_labels_train.npy"
test_seq = test_dir / "break_imgs_test.npy"
test_lab = test_dir / "break_labels_test.npy"

# 기존 bbox 모델 체크포인트
bbox_ckpt_dir = current_dir / "7. resnet_runs" / "checkpoints"
best_x_path = bbox_ckpt_dir / "best_x.keras"
best_y_path = bbox_ckpt_dir / "best_y.keras"
best_z_path = bbox_ckpt_dir / "best_z.keras"


# ============================================================
# 유틸
# ============================================================
def set_seed(seed: int = BASE_SEED) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")

def make_repeated_splits(X, y, n_splits=5, test_size=0.2, random_state=BASE_SEED):
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    splits = []
    for tr_idx, val_idx in splitter.split(X, y.astype(int)):
        splits.append((tr_idx, val_idx))
    return splits


class DelayedModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, start_epoch=5, monitor="val_precision"):
        super().__init__()
        self.filepath = filepath
        self.start_epoch = start_epoch
        self.monitor = monitor
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if (epoch + 1) < self.start_epoch:
            return

        if current > self.best:
            self.best = current
            self.model.save(self.filepath)
            print(f"\nEpoch {epoch+1}: {self.monitor} improved to {current:.5f}, saving model.")


class DelayedEarlyStopping(keras.callbacks.Callback):
    def __init__(self, start_epoch=5, patience=30, monitor="val_precision"):
        super().__init__()
        self.start_epoch = start_epoch
        self.patience = patience
        self.monitor = monitor
        self.best = -np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if (epoch + 1) < self.start_epoch:
            return

        if current > self.best:
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEpoch {epoch+1}: early stopping")
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    print("Restoring best weights.")
                self.model.stop_training = True

class ValidationConstraintMetric(keras.callbacks.Callback):
    def __init__(
        self,
        X_val,
        y_val,
        batch_size=BATCH,
        target_precision=TARGET_PRECISION,
        target_recall=TARGET_RECALL,
    ):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.target_precision = target_precision
        self.target_recall = target_recall

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_prob = self.model.predict(
            self.X_val,
            batch_size=self.batch_size,
            verbose=0
        ).reshape(-1)

        info = find_best_threshold(
            y_true=self.y_val,
            y_prob=val_prob,
            target_precision=self.target_precision,
            target_recall=self.target_recall,
        )

        logs["val_target_score"] = info["score"]
        logs["val_prec_at_target"] = info["precision"]
        logs["val_rec_at_target"] = info["recall"]
        logs["val_th_at_target"] = info["threshold"]
        logs["val_target_ok"] = float(info["satisfied"])

        print(
            f"\nEpoch {epoch+1}: "
            f"val_target_score={info['score']:.5f}, "
            f"val_prec_at_target={info['precision']:.5f}, "
            f"val_rec_at_target={info['recall']:.5f}, "
            f"val_th_at_target={info['threshold']:.2f}, "
            f"val_target_ok={int(info['satisfied'])}"
        )



# ============================================================
# 데이터 로드
# ============================================================
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not train_seq.exists() or not train_lab.exists():
        raise FileNotFoundError(f"학습 데이터 없음: {train_seq}, {train_lab}")
    if not test_seq.exists() or not test_lab.exists():
        raise FileNotFoundError(f"테스트 데이터 없음: {test_seq}, {test_lab}")

    X = np.load(train_seq).astype(np.float32)
    y = np.load(train_lab).astype(np.float32)
    X_test = np.load(test_seq).astype(np.float32)
    y_test = np.load(test_lab).astype(np.float32)

    if y.ndim != 2 or y.shape[1] < 1:
        raise ValueError(f"train label shape 이상함: {y.shape}. 첫 번째 열에 cls가 있어야 함.")
    if y_test.ndim != 2 or y_test.shape[1] < 1:
        raise ValueError(f"test label shape 이상함: {y_test.shape}. 첫 번째 열에 cls가 있어야 함.")

    # 첫 번째 열: break / normal 분류 라벨
    y_cls = y[:, 0].astype(np.float32)
    y_test_cls = y_test[:, 0].astype(np.float32)
    return X, y_cls, X_test, y_test_cls

def train_one_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    split_id: int,
) -> Dict:
    split_ckpt_dir = ckpt_dir / f"split_{split_id}"
    best_model_path = split_ckpt_dir / "best_binary.keras"

    ds_train = make_ds(X_train, y_train, training=True)
    ds_val = make_ds(X_val, y_val, training=False)

    class_weight = {
        0: 1.0,
        1: 4.0,
    }

    model = build_binary_classifier(input_shape=X_train.shape[1:])

    # =========================
    # Stage 1: head-only
    # =========================
    set_backbone_trainability(
        model,
        unfreeze_last_two_stages=False,
        freeze_bn=True,
    )
    compile_binary_model(model, lr=HEAD_LR)
    print_trainable_status(model, f"split{split_id}_stage1_head_only")

    history_head = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=HEAD_EPOCHS,
        callbacks=make_callbacks(
            X_val, y_val,
            ckpt_path=best_model_path,
            log_name=f"split_{split_id}_stage1",
        ),
        class_weight=class_weight,
        verbose=1,
    )

    if best_model_path.exists():
        model = keras.models.load_model(str(best_model_path), compile=False)

    # =========================
    # Stage 2: s4 fine-tuning
    # =========================
    set_backbone_trainability(
        model,
        unfreeze_last_two_stages=True,
        freeze_bn=FREEZE_BN_IN_FT,
    )
    compile_binary_model(model, lr=FT_LR)
    print_trainable_status(model, f"split{split_id}_stage2_s3s4_finetune")

    history_ft = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=FT_EPOCHS,
        callbacks=make_callbacks(
            X_val, y_val,
            ckpt_path=best_model_path,
            log_name=f"split_{split_id}_stage2",
        ),
        class_weight=class_weight,
        verbose=1,
    )

    best_model = keras.models.load_model(str(best_model_path), compile=True) if best_model_path.exists() else model

    val_prob = best_model.predict(X_val, batch_size=BATCH, verbose=0).reshape(-1)
    th_info = find_best_threshold(
        y_true=y_val,
        y_prob=val_prob,
        target_precision=TARGET_PRECISION,
        target_recall=TARGET_RECALL,
    )

    history_head_dict = {k: [float(v) for v in vals] for k, vals in history_head.history.items()}
    history_ft_dict = {k: [float(v) for v in vals] for k, vals in history_ft.history.items()}

    best_epoch_head = get_best_epoch_from_history(
        history_head_dict,
        monitor="val_target_score",
        start_epoch=5,
    )
    best_epoch_ft = get_best_epoch_from_history(
        history_ft_dict,
        monitor="val_target_score",
        start_epoch=5,
    )

    return {
        "split_id": split_id,
        "threshold": float(th_info["threshold"]),
        "precision": float(th_info["precision"]),
        "recall": float(th_info["recall"]),
        "f1": float(th_info["f1"]),
        "score": float(th_info["score"]),
        "satisfied": bool(th_info["satisfied"]),
        "best_epoch_head": int(best_epoch_head),
        "best_epoch_ft": int(best_epoch_ft),
        "best_model_path": str(best_model_path),
        "history_head": history_head_dict,
        "history_ft": history_ft_dict,
    }

def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    head_epochs_max: int,
    ft_epochs_max: int,
) -> keras.Model:
    final_ckpt_dir = ckpt_dir / "final_retrain"
    best_model_path = final_ckpt_dir / "best_final_binary.keras"

    ds_train = make_ds(X_train, y_train, training=True)
    ds_val = make_ds(X_val, y_val, training=False)

    class_weight = {
        0: 1.0,
        1: 4.0,
    }

    model = build_binary_classifier(input_shape=X_train.shape[1:])

    # =========================
    # Final Stage 1: head-only
    # =========================
    set_backbone_trainability(
        model,
        unfreeze_last_two_stages=False,
        freeze_bn=True,
    )
    compile_binary_model(model, lr=HEAD_LR)
    print_trainable_status(model, "final_stage1_head_only")

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=head_epochs_max,
        callbacks=make_callbacks(
            X_val, y_val,
            ckpt_path=best_model_path,
            log_name="final_stage1",
        ),
        class_weight=class_weight,
        verbose=1,
    )

    if best_model_path.exists():
        model = keras.models.load_model(str(best_model_path), compile=False)

    # =========================
    # Final Stage 2: s3+s4 fine-tuning
    # =========================
    set_backbone_trainability(
        model,
        unfreeze_last_two_stages=True,
        freeze_bn=FREEZE_BN_IN_FT,
    )
    compile_binary_model(model, lr=FT_LR)
    print_trainable_status(model, "final_stage2_s3s4_finetune")

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=ft_epochs_max,
        callbacks=make_callbacks(
            X_val, y_val,
            ckpt_path=best_model_path,
            log_name="final_stage2",
        ),
        class_weight=class_weight,
        verbose=1,
    )

    best_model = keras.models.load_model(str(best_model_path), compile=True) if best_model_path.exists() else model
    return best_model


def _augment_binary(img: tf.Tensor, label: tf.Tensor):
    img = tf.image.random_brightness(img, max_delta=AUG_BRIGHTNESS_DELTA)
    img = tf.image.random_contrast(img, lower=AUG_CONTRAST_LOWER, upper=AUG_CONTRAST_UPPER)

    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=AUG_NOISE_STD, dtype=img.dtype)
    img = img + noise

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

# ============================================================
# 데이터셋
# ============================================================
def make_ds(X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 5000), seed=BASE_SEED, reshuffle_each_iteration=True)
        if USE_AUGMENTATION:
            ds = ds.map(_augment_binary, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 기존 bbox 모델에서 feature extractor 만들기
# ============================================================
def _find_feature_layer_name(model: keras.Model) -> str:
    # 우선순위: 이름이 gap으로 끝나는 레이어 -> dropout 직전 -> 뒤에서 두 번째
    for layer in model.layers:
        if layer.name.endswith("_gap"):
            return layer.name

    for idx, layer in enumerate(model.layers):
        if isinstance(layer, layers.Dropout) and idx > 0:
            return model.layers[idx - 1].name

    if len(model.layers) < 2:
        raise ValueError(f"모델 레이어 수가 너무 적음: {model.name}")
    return model.layers[-2].name



def build_frozen_backbone(model_path: Path, prefix: str) -> keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {model_path}")

    base = keras.models.load_model(str(model_path), compile=False)
    feat_layer_name = _find_feature_layer_name(base)
    feat_out = base.get_layer(feat_layer_name).output
    extractor = keras.Model(inputs=base.input, outputs=feat_out, name=f"{prefix}_extractor")
    extractor.trainable = False
    return extractor


# ============================================================
# 분류 모델
# ============================================================
def build_binary_classifier(input_shape: Tuple[int, ...]) -> keras.Model:
    x_backbone = build_frozen_backbone(best_x_path, "x")
    y_backbone = build_frozen_backbone(best_y_path, "y")
    z_backbone = build_frozen_backbone(best_z_path, "z")

    inp = keras.Input(shape=input_shape, name="binary_input")

    fx_raw = x_backbone(inp)   # (None, 512)
    fy_raw = y_backbone(inp)   # (None, 512)
    fz_raw = z_backbone(inp)   # (None, 512)

    # 축별 feature projection
    fx = layers.Dense(256, activation="relu", name="x_proj")(fx_raw)
    fy = layers.Dense(256, activation="relu", name="y_proj")(fy_raw)
    fz = layers.Dense(256, activation="relu", name="z_proj")(fz_raw)

    # 샘플별 x/y/z 중요도 계산
    gate_feat = layers.Concatenate(name="gate_input")([fx, fy, fz])
    gate_feat = layers.Dense(64, activation="relu", name="axis_gate_fc1")(gate_feat)
    axis_gate = layers.Dense(3, activation="softmax", name="axis_gate")(gate_feat)   # (None, 3)

    # Lambda 없이 축별 feature를 stack
    fx_e = layers.Reshape((1, 256), name="x_expand")(fx)
    fy_e = layers.Reshape((1, 256), name="y_expand")(fy)
    fz_e = layers.Reshape((1, 256), name="z_expand")(fz)

    stacked_feat = layers.Concatenate(axis=1, name="stacked_xyz_feat")([fx_e, fy_e, fz_e])   # (None, 3, 256)
    axis_gate_e = layers.Reshape((3, 1), name="axis_gate_expand")(axis_gate)                   # (None, 3, 1)

    weighted_stack = layers.Multiply(name="weighted_xyz_stack")([stacked_feat, axis_gate_e])   # (None, 3, 256)
    weighted_flat = layers.Flatten(name="weighted_xyz_flat")(weighted_stack)                    # (None, 768)

    # 원본 + 가중 feature + gate 자체를 같이 사용
    feat = layers.Concatenate(name="fused_xyz_features")([
        fx_raw, fy_raw, fz_raw,   # 1536
        weighted_flat,            # 768
        axis_gate                 # 3
    ])

    feat = layers.BatchNormalization(name="cls_bn0")(feat)
    feat = layers.Dense(128, activation="relu", name="cls_fc1")(feat)
    feat = layers.Dropout(DROPOUT, name="cls_drop1")(feat)
    feat = layers.Dense(32, activation="relu", name="cls_fc2")(feat)
    feat = layers.Dropout(DROPOUT, name="cls_drop2")(feat)
    out = layers.Dense(1, activation="sigmoid", name="binary_cls")(feat)

    model = keras.Model(inp, out, name="binary_from_bbox_backbones_gated")
    return model


# ============================================================
# 콜백
# ============================================================
def make_callbacks(X_val, y_val, ckpt_path: Path, log_name: str) -> list:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = run_dir / "logs" / log_name / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    return [
        ValidationConstraintMetric(
            X_val=X_val,
            y_val=y_val,
            batch_size=BATCH,
            target_precision=TARGET_PRECISION,
            target_recall=TARGET_RECALL,
        ),
        DelayedModelCheckpoint(
            filepath=str(ckpt_path),
            start_epoch=5,
            monitor="val_target_score",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_target_score",
            mode="max",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
        DelayedEarlyStopping(
            start_epoch=5,
            patience=30,
            monitor="val_target_score",
        ),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]


# ============================================================
# 리포트 저장
# ============================================================
def save_history_plot(history: keras.callbacks.History, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history

    # loss
    fig = plt.figure(figsize=(7, 4))
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.title("Binary classifier loss")
    plt.tight_layout()
    plt.savefig(out_dir / "history_loss.png", dpi=150)
    plt.close(fig)

    # auc
    fig = plt.figure(figsize=(7, 4))
    plt.plot(hist.get("auc", []), label="train_auc")
    plt.plot(hist.get("val_auc", []), label="val_auc")
    plt.legend()
    plt.title("Binary classifier AUC")
    plt.tight_layout()
    plt.savefig(out_dir / "history_auc.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.items()}, f, ensure_ascii=False, indent=2)

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = TARGET_PRECISION,
    target_recall: float = TARGET_RECALL,
) -> Dict:
    """
    precision / recall 둘 다 반영한 단일 score로 threshold 선택.
    hard filter / fallback 없이 모든 threshold를 같은 규칙으로 평가한다.

    score 설계:
    - p_ratio = precision / target_precision
    - r_ratio = recall / target_recall
    - base = min(p_ratio, r_ratio)         # 둘 중 약한 쪽을 강하게 반영
    - bonus = 0.1 * (p_ratio + r_ratio)    # 둘 다 높은 경우 보너스
    - 최종 score = base + bonus

    즉 precision만 높거나 recall만 높은 threshold는 불리하고,
    둘 다 같이 높아야 score가 커진다.
    """
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

        # recall을 조금 더 중요하게
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
            score,   # target 0.8 / 0.9에 얼마나 잘 맞는가
            rec,     # 동점이면 recall 높은 쪽
            prec,    # 그다음 precision
            -th
        )

        if (best is None) or (key > best_key):
            best = candidate
            best_key = key

    return best


def evaluate_and_save(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                      out_dir: Path, threshold: float = 0.5) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_test = make_ds(X_test, y_test, training=False)
    metrics = model.evaluate(ds_test, verbose=1, return_dict=True)

    prob = model.predict(X_test, batch_size=BATCH, verbose=0).reshape(-1)
    pred = (prob >= threshold).astype(np.int32)
    y_true = y_test.astype(np.int32)

    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred).tolist()

    auc = None
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, prob))

    threshold_metrics = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

    result = {
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "threshold_metrics": threshold_metrics,
        "roc_auc_sklearn": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    df = {
        "y_true": y_true.tolist(),
        "y_prob": prob.tolist(),
        "y_pred": pred.tolist(),
    }
    with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(df, f, ensure_ascii=False, indent=2)

    return result


def compile_binary_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.BinaryFocalCrossentropy(
            gamma=2.0,
            from_logits=False,
            apply_class_balancing=False,
        ),
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

def set_backbone_trainability(
    model: keras.Model,
    unfreeze_last_two_stages: bool = False,
    freeze_bn: bool = True,
) -> None:
    backbone_names = {"x_extractor", "y_extractor", "z_extractor"}

    for layer in model.layers:
        if layer.name not in backbone_names:
            continue

        if not unfreeze_last_two_stages:
            layer.trainable = False
            for sublayer in layer.layers:
                sublayer.trainable = False
        else:
            layer.trainable = True
            for sublayer in layer.layers:
                sublayer.trainable = False

                if ("_s3_" in sublayer.name) or ("_s4_" in sublayer.name):
                    if freeze_bn and isinstance(sublayer, layers.BatchNormalization):
                        sublayer.trainable = False
                    else:
                        sublayer.trainable = True

def print_trainable_status(model: keras.Model, tag: str) -> None:
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    print(f"[{tag}] trainable params: {trainable:,}")
    print(f"[{tag}] non-trainable params: {non_trainable:,}")


def get_best_epoch_from_history(
    history_dict: Dict,
    monitor: str = "val_prec_at_rec90",
    start_epoch: int = 5,
) -> int:
    values = history_dict.get(monitor, [])
    if len(values) == 0:
        raise ValueError(f"{monitor} not found in history")

    begin = max(0, start_epoch - 1)
    sliced = values[begin:]
    best_local_idx = int(np.argmax(sliced))
    return begin + best_local_idx + 1

# ============================================================
# 메인
# ============================================================
def main() -> None:
    set_seed(BASE_SEED)

    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    X, y, X_test, y_test = load_data()
    print("X:", X.shape, "y:", y.shape, "class counts:", np.unique(y.astype(int), return_counts=True))
    print("X_test:", X_test.shape, "y_test:", y_test.shape, "test counts:", np.unique(y_test.astype(int), return_counts=True))

    # final threshold calibration용 holdout 분리
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X,
        y,
        test_size=CALIBRATION_RATIO,
        random_state=BASE_SEED,
        stratify=y.astype(int),
    )

    print("X_fit:", X_fit.shape, "y_fit:", y_fit.shape)
    print("X_cal:", X_cal.shape, "y_cal:", y_cal.shape)

    splits = make_repeated_splits(
        X_fit, y_fit,
        n_splits=5,
        test_size=0.2,
        random_state=BASE_SEED,
    )
    print("num_splits:", len(splits))

    split_results = []

    for split_id, (tr_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n===== Split {split_id}/{len(splits)} =====")

        X_train, X_val = X_fit[tr_idx], X_fit[val_idx]
        y_train, y_val = y_fit[tr_idx], y_fit[val_idx]

        print("X_train:", X_train.shape, "X_val:", X_val.shape)

        split_result = train_one_split(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            split_id=split_id,
        )

        split_results.append(split_result)
        print(f"[split {split_id}] threshold={split_result['threshold']:.4f}, "
              f"precision={split_result['precision']:.4f}, "
              f"recall={split_result['recall']:.4f}")

    thresholds = [r["threshold"] for r in split_results]
    precisions = [r["precision"] for r in split_results]
    recalls = [r["recall"] for r in split_results]
    best_epochs_head = [r["best_epoch_head"] for r in split_results]
    best_epochs_ft = [r["best_epoch_ft"] for r in split_results]

    median_threshold = float(np.median(thresholds))
    median_head_epoch = int(round(np.median(best_epochs_head)))
    median_ft_epoch = int(round(np.median(best_epochs_ft)))

    summary = {
        "fit_size": int(len(X_fit)),
        "calibration_size": int(len(X_cal)),
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "best_epochs_head": best_epochs_head,
        "best_epochs_ft": best_epochs_ft,
        "median_threshold": median_threshold,
        "mean_threshold": float(np.mean(thresholds)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "median_head_epoch": median_head_epoch,
        "median_ft_epoch": median_ft_epoch,
        "split_results": split_results,
        "final_val_ratio": FINAL_VAL_RATIO,
    }

    out_dir = run_dir / f"cv_summary_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


    # =========================
    # Final retrain with internal validation
    # =========================
    X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
        X_fit,
        y_fit,
        test_size=FINAL_VAL_RATIO,
        random_state=BASE_SEED,
        stratify=y_fit.astype(int),
    )

    print("\n===== FINAL RETRAIN =====")
    print("cv_median_threshold (for reference only):", median_threshold)
    print("cv_median_head_epoch (reference):", median_head_epoch)
    print("cv_median_ft_epoch (reference):", median_ft_epoch)
    print("X_final_train:", X_final_train.shape, "y_final_train:", y_final_train.shape)
    print("X_final_val:", X_final_val.shape, "y_final_val:", y_final_val.shape)

    final_model = train_final_model(
        X_train=X_final_train,
        y_train=y_final_train,
        X_val=X_final_val,
        y_val=y_final_val,
        head_epochs_max=HEAD_EPOCHS,
        ft_epochs_max=FT_EPOCHS,
    )

    final_model_path = ckpt_dir / "final_binary.keras"
    final_model.save(final_model_path)

    # =========================
    # Final threshold selection on calibration holdout
    # =========================
    cal_prob = final_model.predict(X_cal, batch_size=BATCH, verbose=0).reshape(-1)
    final_th_info = find_best_threshold(
        y_true=y_cal,
        y_prob=cal_prob,
        target_precision=TARGET_PRECISION,
        target_recall=TARGET_RECALL,
    )

    print("\n===== FINAL CALIBRATION THRESHOLD =====")
    print(json.dumps(final_th_info, ensure_ascii=False, indent=2))

    summary["final_calibration_threshold"] = float(final_th_info["threshold"])
    summary["final_calibration_precision"] = float(final_th_info["precision"])
    summary["final_calibration_recall"] = float(final_th_info["recall"])
    summary["final_calibration_f1"] = float(final_th_info["f1"])

    final_eval_dir = run_dir / f"final_eval_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    result = evaluate_and_save(
        final_model,
        X_test,
        y_test,
        final_eval_dir,
        threshold=final_th_info["threshold"],
    )

    with open(final_eval_dir / "cv_summary_used.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== FINAL TEST RESULT =====")
    print("final_model_path:", final_model_path)
    print("final_eval_dir:", final_eval_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
