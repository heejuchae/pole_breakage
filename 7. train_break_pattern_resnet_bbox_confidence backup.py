# 7. ResNet(2D)로 파단 패턴 학습 (Break / Normal) - bbox + confidence 버전
"""
- 입력: `5. train_data` 내 break_imgs_train.npy (shape: [N, 304, 19, 3]), break_labels_train.npy (cls + bbox + mask)
- ROI별(x,y,z) 독립 ResNet 회귀 모델 학습, huber_bestpair + BCE(confidence) loss
- 출력: 5*P (hc, hw, dc, dw, confidence) per box. GT 매칭된 pred에 confidence=1 학습
- 체크포인트: `7. resnet_runs/checkpoints/best_x.keras`, best_y.keras, best_z.keras

- 데이터 증강: 좌우반전(degree축), 상하반전(height축) 랜덤 적용. 학습 시에만 사용.
- 기본 동작(Windows): WSL2에서 train_break_pattern_resnet_bbox_wsl2.sh 를 실행해 GPU 사용.
- 로컬(현재 환경)에서 학습: python "7. train_break_pattern_resnet_bbox.py" --local

- 주의: 4*P 구버전 체크포인트는 호환되지 않음. 재학습 필요.
"""

import os
import sys
import subprocess
from pathlib import Path

# Windows에서 --local 없이 실행 시 WSL2 스크립트로 넘겨서 GPU 학습
_run_local = "--local" in sys.argv or sys.platform != "win32"
if _run_local:
    if "--local" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--local"]
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = Path(_script_dir).parent
    _sh_path = Path(_script_dir) / "train_break_pattern_resnet_bbox_wsl2.sh"
    if not _sh_path.exists():
        _sh_path = _project_root / "make_ai" / "train_break_pattern_resnet_bbox_wsl2.sh"
    if _sh_path.exists():
        _abs = _sh_path.resolve()
        _drive = _abs.drive
        _wsl_path = ("/mnt/" + _drive[0].lower() + str(_abs)[len(_drive):].replace("\\", "/")) if _drive else str(_abs).replace("\\", "/")
        print("WSL2에서 GPU 학습 실행:", _wsl_path)
        ret = subprocess.run(["wsl", "bash", _wsl_path], cwd=str(_project_root))
        sys.exit(ret.returncode)
    # 스크립트 없으면 로컬 학습 진행

import datetime
import json
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow import (Windows 로컬 시 CUDA PATH 추가)
if sys.platform == "win32":
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.exists(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_SEED = 42

# ============================================================================
# 1) 데이터 경로 / 로드
# ============================================================================

data_root = Path(current_dir) / "5. train_data"
train_dir = data_root / "train"
test_dir = data_root / "test"

train_seq = train_dir / "break_imgs_train.npy"
train_lab = train_dir / "break_labels_train.npy"
test_seq = test_dir / "break_imgs_test.npy"
test_lab = test_dir / "break_labels_test.npy"


def load_data():
    X = np.load(train_seq).astype(np.float32)
    y = np.load(train_lab).astype(np.float32)
    K = int((y.shape[1] - 1) // 15)
    assert 1 + 15 * K == y.shape[1], f"y.shape[1]={y.shape[1]} not 1+15*K"
    X_test = np.load(test_seq).astype(np.float32)
    y_test = np.load(test_lab).astype(np.float32)
    return X, y, X_test, y_test, K


# ============================================================================
# 2) Train/Val split
# ============================================================================

def split_train_val(X, y, test_size=0.2, random_state=BASE_SEED):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y[:, 0].astype(int))


# ============================================================================
# 3) ROI 타겟 슬라이스 / 데이터셋
# ============================================================================

def slice_roi_targets(y, roi_idx: int, K: int):
    """y: (N, 1+15K). 반환: y_cls (N,1), y_reg_roi (N, 5K) = [bbox_r, mask_r]."""
    bbox_dim = 12 * K  # 3*K*4
    mask_dim = 3 * K
    bbox_flat = y[:, 1 : 1 + bbox_dim].astype("float32")
    mask_flat = y[:, 1 + bbox_dim : 1 + bbox_dim + mask_dim].astype("float32")
    bbox = bbox_flat.reshape(-1, 3, K, 4)
    mask = mask_flat.reshape(-1, 3, K)
    bbox_r = bbox[:, roi_idx, :, :].reshape(-1, K * 4)
    mask_r = mask[:, roi_idx, :]
    y_reg_roi = np.concatenate([bbox_r, mask_r], axis=1).astype("float32")
    return y[:, 0:1].astype("float32"), y_reg_roi


BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE
# 학습 시 좌우/상하 flip 증강 사용 여부. False로 두면 증강 비활성화.
USE_AUGMENTATION = True

AUG_NOISE_STD = 0.02
AUG_BRIGHTNESS_DELTA = 0.05
AUG_CONTRAST_LOWER = 0.95
AUG_CONTRAST_UPPER = 1.05
AUG_SHIFT_RATIO_H = 0.03   # 높이축 최대 3%
AUG_SHIFT_RATIO_W = 0.03   # degree축 최대 3%

def _augment_flip(img: tf.Tensor, y_reg: tf.Tensor, K: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    추천 증강만 적용:
    1) 좌우/상하 flip
    2) 약한 brightness/contrast
    3) 약한 gaussian noise
    4) 작은 translation

    bbox 형식: [hc, hw, dc, dw]
    - horizontal flip: dc -> 1 - dc
    - vertical flip: hc -> 1 - hc
    - translation:
        hc -> hc + dy
        dc -> dc + dx
    """
    bbox_dim = K * 4
    bbox = tf.reshape(y_reg[:bbox_dim], (K, 4))   # [K,4]
    mask = y_reg[bbox_dim:]                       # [K]

    # -------------------------
    # 1) Flip
    # -------------------------
    # u1 = tf.random.uniform([], 0, 1)
    # do_lr = u1 > 0.5
    # img = tf.cond(do_lr, lambda: tf.image.flip_left_right(img), lambda: img)
    # dc_new = tf.cond(do_lr, lambda: 1.0 - bbox[:, 2], lambda: bbox[:, 2])
    # bbox = tf.concat([bbox[:, :2], tf.reshape(dc_new, (-1, 1)), bbox[:, 3:]], axis=1)

    # u2 = tf.random.uniform([], 0, 1)
    # do_ud = u2 > 0.5
    # img = tf.cond(do_ud, lambda: tf.image.flip_up_down(img), lambda: img)
    # hc_new = tf.cond(do_ud, lambda: 1.0 - bbox[:, 0], lambda: bbox[:, 0])
    # bbox = tf.concat([tf.reshape(hc_new, (-1, 1)), bbox[:, 1:]], axis=1)

    # # -------------------------
    # # 2) 약한 brightness / contrast
    # # -------------------------
    img = tf.image.random_brightness(img, max_delta=AUG_BRIGHTNESS_DELTA)
    img = tf.image.random_contrast(img, lower=AUG_CONTRAST_LOWER, upper=AUG_CONTRAST_UPPER)

    # # -------------------------
    # # 3) 약한 Gaussian noise
    # # -------------------------
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=AUG_NOISE_STD, dtype=img.dtype)
    img = img + noise

    # # -------------------------
    # # 4) 작은 translation
    # # -------------------------
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    max_dy = tf.cast(tf.round(tf.cast(h, tf.float32) * AUG_SHIFT_RATIO_H), tf.int32)
    max_dx = tf.cast(tf.round(tf.cast(w, tf.float32) * AUG_SHIFT_RATIO_W), tf.int32)

    dy = tf.random.uniform([], minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32)
    dx = tf.random.uniform([], minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32)

    img = tf.roll(img, shift=dy, axis=0)
    img = tf.roll(img, shift=dx, axis=1)

    dy_norm = tf.cast(dy, tf.float32) / tf.cast(h, tf.float32)
    dx_norm = tf.cast(dx, tf.float32) / tf.cast(w, tf.float32)

    hc = bbox[:, 0] + dy_norm
    hw = bbox[:, 1]
    dc = bbox[:, 2] + dx_norm
    dw = bbox[:, 3]

    # bbox 중심값 clip
    hc = tf.clip_by_value(hc, 0.0, 1.0)
    dc = tf.clip_by_value(dc, 0.0, 1.0)

    bbox = tf.stack([hc, hw, dc, dw], axis=1)

    # 이미지 값 범위 보정
    img = tf.clip_by_value(img, 0.0, 1.0)

    y_reg = tf.concat([tf.reshape(bbox, [-1]), mask], axis=0)
    return img, y_reg


def make_ds_roi(X, y, roi_idx: int, K: int, training: bool, seed: int):
    _, y_reg = slice_roi_targets(y, roi_idx=roi_idx, K=K)
    ds = tf.data.Dataset.from_tensor_slices((X, y_reg.astype("float32")))
    if training:
        ds = ds.shuffle(min(len(X), 5000), seed=seed, reshuffle_each_iteration=True)
        if USE_AUGMENTATION:
            ds = ds.map(lambda img, lbl: _augment_flip(img, lbl, K), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)


# ============================================================================
# 4) ResNet18-like 모델 (회귀 출력 5*P: hc, hw, dc, dw, confidence)
# ============================================================================

def basic_block(x, filters, stride=(1, 1), prefix="bb"):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.ReLU(name=f"{prefix}_relu1")(x)
    x = layers.Conv2D(filters, 3, strides=(1, 1), padding="same", use_bias=False, name=f"{prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    if shortcut.shape[-1] != filters or stride != (1, 1):
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, name=f"{prefix}_proj_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{prefix}_proj_bn")(shortcut)
    x = layers.Add(name=f"{prefix}_add")([x, shortcut])
    x = layers.ReLU(name=f"{prefix}_out")(x)
    return x


def make_stage(x, filters, blocks, first_stride=(1, 1), prefix="stage"):
    x = basic_block(x, filters, stride=first_stride, prefix=f"{prefix}_b0")
    for i in range(1, blocks):
        x = basic_block(x, filters, stride=(1, 1), prefix=f"{prefix}_b{i}")
    return x


def build_resnet18_like(input_shape, pred_num, name="resnet18_like_reg", dropout=0.3):
    inp = keras.Input(shape=input_shape, name=f"{name}_input")
    x = layers.Conv2D(64, 7, strides=(2, 1), padding="same", use_bias=False, name=f"{name}_stem_conv")(inp)
    x = layers.BatchNormalization(name=f"{name}_stem_bn")(x)
    x = layers.ReLU(name=f"{name}_stem_relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=(2, 1), padding="same", name=f"{name}_stem_pool")(x)
    x = make_stage(x, 64, blocks=2, first_stride=(1, 1), prefix=f"{name}_s1")
    x = make_stage(x, 128, blocks=2, first_stride=(2, 1), prefix=f"{name}_s2")
    x = make_stage(x, 256, blocks=2, first_stride=(2, 1), prefix=f"{name}_s3")
    x = make_stage(x, 512, blocks=2, first_stride=(2, 1), prefix=f"{name}_s4")
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    reg_out = layers.Dense(5 * pred_num, activation="sigmoid", name="reg")(x)  # 5*P: hc, hw, dc, dw, conf
    return keras.Model(inp, reg_out, name=name)


# ============================================================================
# 5) IoU / Loss / Metric
# ============================================================================

def iou_2d_from_center_width(pred, true, eps=1e-7):
    """pred, true: (..., 4) [hc, hw, dc, dw]. returns IoU (...,)."""
    pred = tf.cast(pred, tf.float32)
    true = tf.cast(true, tf.float32)
    phc, phw, pdc, pdw = tf.unstack(pred, axis=-1)
    thc, thw, tdc, tdw = tf.unstack(true, axis=-1)
    ph1 = phc - 0.5 * phw
    ph2 = phc + 0.5 * phw
    pd1 = pdc - 0.5 * pdw
    pd2 = pdc + 0.5 * pdw
    th1 = thc - 0.5 * thw
    th2 = thc + 0.5 * thw
    td1 = tdc - 0.5 * tdw
    td2 = tdc + 0.5 * tdw
    ih1 = tf.maximum(ph1, th1)
    ih2 = tf.minimum(ph2, th2)
    id1 = tf.maximum(pd1, td1)
    id2 = tf.minimum(pd2, td2)
    inter_h = tf.maximum(0.0, ih2 - ih1)
    inter_d = tf.maximum(0.0, id2 - id1)
    inter = inter_h * inter_d
    area_p = tf.maximum(0.0, ph2 - ph1) * tf.maximum(0.0, pd2 - pd1)
    area_t = tf.maximum(0.0, th2 - th1) * tf.maximum(0.0, td2 - td1)
    union = area_p + area_t - inter
    return inter / (union + eps)


# conf_loss: bbox 비중 확대를 위해 기본 0.3. IoU threshold 미달 시 conf_loss 미적용.
CONF_WEIGHT = 0.0
IOU_THRESHOLD_FOR_CONF = 0.3


def huber_bestpair_loss(P: int, K: int, delta=0.05, conf_weight=CONF_WEIGHT, iou_threshold_for_conf=IOU_THRESHOLD_FOR_CONF):
    huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def loss(y_true, y_pred):
        pred = tf.reshape(y_pred, [-1, P, 5])
        pred_bbox = pred[:, :, :4]
        pred_conf = pred[:, :, 4]
        gt_flat = y_true[:, : 4 * K]
        m_flat = y_true[:, 4 * K :]
        gt = tf.reshape(gt_flat, [-1, K, 4])
        m = tf.cast(m_flat > 0.5, tf.float32)
        iou_mat = iou_2d_from_center_width(pred_bbox[:, :, None, :], gt[:, None, :, :])
        neg_inf = tf.constant(-1e9, dtype=iou_mat.dtype)
        iou_masked = tf.where(m[:, None, :] > 0, iou_mat, neg_inf)
        B = tf.shape(pred_bbox)[0]
        flat = tf.reshape(iou_masked, [B, -1])
        arg = tf.argmax(flat, axis=-1, output_type=tf.int32)
        gt_idx = arg % K
        pred_idx = arg // K
        pred_sel = tf.gather(pred_bbox, pred_idx, batch_dims=1)
        gt_sel = tf.gather(gt, gt_idx, batch_dims=1)
        per_dim = huber(gt_sel, pred_sel)
        bbox_loss_per_sample = tf.reduce_mean(per_dim, axis=-1)
        has_gt = tf.cast(tf.reduce_any(m > 0, axis=1), tf.float32)
        denom = tf.reduce_sum(has_gt) + 1e-7
        bbox_loss = tf.reduce_sum(bbox_loss_per_sample * has_gt) / denom

        # best-pair IoU per sample: IoU threshold 이상일 때만 conf_loss 적용
        best_iou_per_sample = tf.reduce_max(iou_masked, axis=[1, 2])
        conf_ok = tf.cast(
            (best_iou_per_sample >= iou_threshold_for_conf) & (has_gt > 0), tf.float32
        )
        conf_target = tf.one_hot(pred_idx, depth=P, dtype=tf.float32)
        conf_loss_per_sample = bce(conf_target, pred_conf)
        denom_conf = tf.reduce_sum(conf_ok) + 1e-7
        conf_loss = tf.reduce_sum(conf_loss_per_sample * conf_ok) / denom_conf

        return bbox_loss + conf_weight * conf_loss

    return loss


def bbox_iou_metric_maxPK(P: int, K: int, eps=1e-8):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        gt = tf.reshape(y_true[:, : 4 * K], [-1, K, 4])
        m = tf.reshape(y_true[:, 4 * K : 5 * K], [-1, K])
        pred_full = tf.reshape(y_pred, [-1, P, 5])
        pred = pred_full[:, :, :4]
        iou_mat = iou_2d_from_center_width(pred[:, :, None, :], gt[:, None, :, :], eps=eps)
        neg_inf = tf.constant(-1e9, dtype=iou_mat.dtype)
        iou_masked = tf.where(m[:, None, :] > 0.5, iou_mat, neg_inf)
        best_iou = tf.reduce_max(iou_masked, axis=[1, 2])
        has_gt = tf.cast(tf.reduce_any(m > 0.5, axis=1), tf.float32)
        best_iou = tf.where(has_gt > 0, best_iou, tf.zeros_like(best_iou))
        denom = tf.reduce_sum(has_gt) + eps
        return tf.reduce_sum(best_iou * has_gt) / denom

    metric.__name__ = "bbox_iou"
    return metric


# ============================================================================
# 6) 콜백 / 학습 루프
# ============================================================================

run_dir = Path(current_dir) / "7. resnet_runs"
ckpt_dir = run_dir / "checkpoints"
MONITOR = "val_bbox_iou"
MODE = "max"


def make_callbacks(axis: str):
    best_ckpt_path = ckpt_dir / f"best_{axis}.keras"
    log_dir = run_dir / "logs" / axis / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    cb_best = keras.callbacks.ModelCheckpoint(
        filepath=str(best_ckpt_path),
        monitor=MONITOR,
        mode=MODE,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    cb_tb = keras.callbacks.TensorBoard(log_dir=str(log_dir))
    cb_rlr = keras.callbacks.ReduceLROnPlateau(
        monitor=MONITOR, mode=MODE, factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    cb_es = keras.callbacks.EarlyStopping(
        monitor=MONITOR, mode=MODE, patience=50, restore_best_weights=True, verbose=1
    )
    return [cb_best, cb_tb, cb_rlr, cb_es]


def build_and_compile_model(axis: str, input_shape, P: int, K: int):
    m = build_resnet18_like(input_shape=input_shape, pred_num=P, name=f"resnet18_like_reg_{axis}")
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=huber_bestpair_loss(P=P, K=K, delta=0.05, conf_weight=CONF_WEIGHT, iou_threshold_for_conf=IOU_THRESHOLD_FOR_CONF),
        metrics=[bbox_iou_metric_maxPK(P=P, K=K)],
        jit_compile=False,
    )
    return m


# ============================================================================
# 7) 평가: eval_bbox_roi_bestpair
# ============================================================================

def to_corners_np(x):
    hc, hw, dc, dw = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    hmin = hc - 0.5 * hw
    hmax = hc + 0.5 * hw
    dmin = dc - 0.5 * dw
    dmax = dc + 0.5 * dw
    return hmin, hmax, dmin, dmax


def iou_matrix_np(pred, gt, eps=1e-8):
    """pred: (N,P,4), gt: (N,K,4). return iou: (N,P,K)."""
    phmin, phmax, pdmin, pdmax = to_corners_np(pred[:, :, None, :])
    thmin, thmax, tdmin, tdmax = to_corners_np(gt[:, None, :, :])
    ihmin = np.maximum(phmin, thmin)
    ihmax = np.minimum(phmax, thmax)
    idmin = np.maximum(pdmin, tdmin)
    idmax = np.minimum(pdmax, tdmax)
    inter_h = np.maximum(0.0, ihmax - ihmin)
    inter_d = np.maximum(0.0, idmax - idmin)
    inter = inter_h * inter_d
    area_p = np.maximum(0.0, phmax - phmin) * np.maximum(0.0, pdmax - pdmin)
    area_t = np.maximum(0.0, thmax - thmin) * np.maximum(0.0, tdmax - tdmin)
    union = area_p + area_t - inter + eps
    return inter / union


def eval_bbox_roi_bestpair(model, X, y, roi_idx: int, K: int, P: int, batch=32, save_dir=None, prefix=""):
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    _, y_reg = slice_roi_targets(y, roi_idx=roi_idx, K=K)
    gt_bbox = y_reg[:, : 4 * K].reshape(-1, K, 4)
    gt_mask = y_reg[:, 4 * K : 5 * K].reshape(-1, K)
    has_gt = gt_mask.sum(axis=1) > 0

    pred = model.predict(X, batch_size=batch, verbose=0)
    pred_full = pred.reshape(-1, P, 5)  # 5*P: hc, hw, dc, dw, conf
    pred_bbox = pred_full[:, :, :4]

    iou = iou_matrix_np(pred_bbox, gt_bbox)
    iou_masked = np.where(gt_mask[:, None, :] > 0.5, iou, -1e9)
    flat = iou_masked.reshape(-1, P * K)
    arg = flat.argmax(axis=1)
    best_iou = flat[np.arange(len(flat)), arg]
    pred_idx = arg // K
    gt_idx = arg % K

    best_iou_valid = best_iou[has_gt]
    mean_best_iou = float(best_iou_valid.mean()) if best_iou_valid.size else np.nan

    pred_sel = pred_bbox[np.arange(len(pred_bbox)), pred_idx]
    gt_sel = gt_bbox[np.arange(len(gt_bbox)), gt_idx]
    err = (pred_sel - gt_sel)[has_gt]
    rmse = np.sqrt(np.mean(err ** 2, axis=0)) if err.size else np.array([np.nan] * 4)

    print(f"[ROI {roi_idx}] mean Best-IoU(max(PxK)) = {mean_best_iou:.4f}  (valid samples={has_gt.sum()}/{len(has_gt)})")
    print(f"[ROI {roi_idx}] RMSE(best-pair) hc/hw/dc/dw = {[float(x) for x in rmse]}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(best_iou_valid, bins=30)
    axes[0].set_title(f"ROI {roi_idx} Best-IoU (max(PxK))")
    axes[0].set_xlabel("Best IoU")
    axes[1].scatter(gt_sel[has_gt, 0], pred_sel[has_gt, 0], s=6)
    axes[1].set_title("True vs Pred (hc)")
    axes[2].scatter(gt_sel[has_gt, 2], pred_sel[has_gt, 2], s=6)
    axes[2].set_title("True vs Pred (dc)")
    fig.suptitle(f"ROI {roi_idx} | mean Best-IoU={mean_best_iou:.4f} | RMSE hc/hw/dc/dw={[float(x) for x in rmse]}", y=1.02)
    fig.tight_layout()

    if save_dir is not None:
        out_png = save_dir / f"{prefix}roi{roi_idx}_bestpair_summary.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print("Saved:", out_png)
        df = pd.DataFrame({
            "idx": np.arange(len(best_iou)),
            "has_gt": has_gt.astype(int),
            "best_iou": best_iou,
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "gt_hc": gt_sel[:, 0], "gt_hw": gt_sel[:, 1], "gt_dc": gt_sel[:, 2], "gt_dw": gt_sel[:, 3],
            "pred_hc": pred_sel[:, 0], "pred_hw": pred_sel[:, 1], "pred_dc": pred_sel[:, 2], "pred_dw": pred_sel[:, 3],
        })
        df.to_csv(save_dir / f"{prefix}roi{roi_idx}_bestpair_rows.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)

    return {"mean_best_iou": mean_best_iou, "rmse": rmse, "best_iou": best_iou, "pred_idx": pred_idx, "gt_idx": gt_idx, "has_gt": has_gt}


def load_best_or_current(best_ckpt_path, fallback_model, P: int, K: int):
    if Path(best_ckpt_path).exists():
        print("Loading best model:", best_ckpt_path)
        m = keras.models.load_model(str(best_ckpt_path), compile=False)
        m.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=huber_bestpair_loss(P=P, K=K, delta=0.05, conf_weight=CONF_WEIGHT, iou_threshold_for_conf=IOU_THRESHOLD_FOR_CONF),
            metrics=[bbox_iou_metric_maxPK(P=P, K=K)],
            jit_compile=False,
        )
        return m
    return fallback_model


# ============================================================================
# 8) main
# ============================================================================

def main():
    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    if not train_seq.exists() or not train_lab.exists():
        raise FileNotFoundError(f"학습 데이터 없음: {train_seq}, {train_lab}")

    X, y, X_test, y_test, K = load_data()
    print("X:", X.shape, "y:", y.shape, "cls counts:", np.unique(y[:, 0].astype(int), return_counts=True))
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    X_train, X_val, y_train, y_val = split_train_val(X, y)
    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    P = 3
    EPOCHS = 300

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_x = build_resnet18_like(input_shape=X_train.shape[1:], pred_num=P, name="resnet18_like_reg_x")
    model_y = build_resnet18_like(input_shape=X_train.shape[1:], pred_num=P, name="resnet18_like_reg_y")
    model_z = build_resnet18_like(input_shape=X_train.shape[1:], pred_num=P, name="resnet18_like_reg_z")

    histories = {}
    for axis, roi_idx in [("x", 0), ("y", 1), ("z", 2)]:
        seed = BASE_SEED + roi_idx
        tf.keras.utils.set_random_seed(seed)
        ds_train = make_ds_roi(X_train, y_train, roi_idx=roi_idx, K=K, training=True, seed=seed)
        ds_val = make_ds_roi(X_val, y_val, roi_idx=roi_idx, K=K, training=False, seed=seed)
        model = build_and_compile_model(axis, X_train.shape[1:], P=P, K=K)
        callbacks = make_callbacks(axis)
        history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=1)
        histories[axis] = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # 테스트 평가
    seed_test = BASE_SEED + 999
    best_x = load_best_or_current(ckpt_dir / "best_x.keras", model_x, P=P, K=K)
    best_y = load_best_or_current(ckpt_dir / "best_y.keras", model_y, P=P, K=K)
    best_z = load_best_or_current(ckpt_dir / "best_z.keras", model_z, P=P, K=K)

    for name, m, roi_idx in [("X", best_x, 0), ("Y", best_y, 1), ("Z", best_z, 2)]:
        ds_t = make_ds_roi(X_test, y_test, roi_idx=roi_idx, K=K, training=False, seed=seed_test)
        res = m.evaluate(ds_t, verbose=1)
        print(f"== Test {name} == ", dict(zip(m.metrics_names, res)))

    out_dir = Path(current_dir) / f"eval_bestpair_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("save to:", out_dir)
    res_x = eval_bbox_roi_bestpair(best_x, X_test, y_test, roi_idx=0, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="x_")
    res_y = eval_bbox_roi_bestpair(best_y, X_test, y_test, roi_idx=1, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="y_")
    res_z = eval_bbox_roi_bestpair(best_z, X_test, y_test, roi_idx=2, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="z_")

    # 학습 히스토리 저장
    with open(run_dir / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)
    print("학습 완료. 체크포인트:", ckpt_dir)
    print("평가 결과 디렉터리:", out_dir)


if __name__ == "__main__":
    main()
