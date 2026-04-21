import os
import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 경로 설정
# =========================
data_root = current_dir / "5. train_data"
train_dir = data_root / "train"
test_dir = data_root / "test"

test_seq = test_dir / "break_imgs_test.npy"
test_lab = test_dir / "break_labels_test.npy"

# 여기만 바꾸면 됨
model_dir = current_dir / "최종모델" / "파인튜닝 전"

best_x_path = model_dir / "best_x.keras"
best_y_path = model_dir / "best_y.keras"
best_z_path = model_dir / "best_z.keras"

BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE
BASE_SEED = 42

# 기존 학습 코드와 동일
CONF_WEIGHT = 0.0
IOU_THRESHOLD_FOR_CONF = 0.3


# =========================
# 데이터 로드
# =========================
def load_test_data():
    X_test = np.load(test_seq).astype(np.float32)
    y_test = np.load(test_lab).astype(np.float32)
    K = int((y_test.shape[1] - 1) // 15)
    assert 1 + 15 * K == y_test.shape[1], f"y_test.shape[1]={y_test.shape[1]} not 1+15*K"
    return X_test, y_test, K


def slice_roi_targets(y, roi_idx: int, K: int):
    bbox_dim = 12 * K
    mask_dim = 3 * K

    bbox_flat = y[:, 1:1 + bbox_dim].astype("float32")
    mask_flat = y[:, 1 + bbox_dim:1 + bbox_dim + mask_dim].astype("float32")

    bbox = bbox_flat.reshape(-1, 3, K, 4)
    mask = mask_flat.reshape(-1, 3, K)

    bbox_r = bbox[:, roi_idx, :, :].reshape(-1, K * 4)
    mask_r = mask[:, roi_idx, :]
    y_reg_roi = np.concatenate([bbox_r, mask_r], axis=1).astype("float32")

    return y[:, 0:1].astype("float32"), y_reg_roi


def make_ds_roi(X, y, roi_idx: int, K: int):
    _, y_reg = slice_roi_targets(y, roi_idx=roi_idx, K=K)
    ds = tf.data.Dataset.from_tensor_slices((X, y_reg.astype("float32")))
    return ds.batch(BATCH).prefetch(AUTOTUNE)


# =========================
# 모델 로딩용 custom objects
# =========================
def iou_2d_from_center_width(pred, true, eps=1e-7):
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


def huber_bestpair_loss(P: int, K: int, delta=0.05, conf_weight=CONF_WEIGHT, iou_threshold_for_conf=IOU_THRESHOLD_FOR_CONF):
    huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def loss(y_true, y_pred):
        pred = tf.reshape(y_pred, [-1, P, 5])
        pred_bbox = pred[:, :, :4]
        pred_conf = pred[:, :, 4]

        gt_flat = y_true[:, :4 * K]
        m_flat = y_true[:, 4 * K:]

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

        best_iou_per_sample = tf.reduce_max(iou_masked, axis=[1, 2])
        conf_ok = tf.cast(
            (best_iou_per_sample >= iou_threshold_for_conf) & (has_gt > 0),
            tf.float32,
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

        gt = tf.reshape(y_true[:, :4 * K], [-1, K, 4])
        m = tf.reshape(y_true[:, 4 * K:5 * K], [-1, K])

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


# =========================
# 상세 평가
# =========================
def to_corners_np(x):
    hc, hw, dc, dw = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    hmin = hc - 0.5 * hw
    hmax = hc + 0.5 * hw
    dmin = dc - 0.5 * dw
    dmax = dc + 0.5 * dw
    return hmin, hmax, dmin, dmax


def iou_matrix_np(pred, gt, eps=1e-8):
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
    gt_bbox = y_reg[:, :4 * K].reshape(-1, K, 4)
    gt_mask = y_reg[:, 4 * K:5 * K].reshape(-1, K)
    has_gt = gt_mask.sum(axis=1) > 0

    pred = model.predict(X, batch_size=batch, verbose=0)
    pred_full = pred.reshape(-1, P, 5)
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

    print(f"[ROI {roi_idx}] mean Best-IoU = {mean_best_iou:.4f}")
    print(f"[ROI {roi_idx}] RMSE hc/hw/dc/dw = {[float(x) for x in rmse]}")

    if save_dir is not None:
        df = pd.DataFrame({
            "idx": np.arange(len(best_iou)),
            "has_gt": has_gt.astype(int),
            "best_iou": best_iou,
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "gt_hc": gt_sel[:, 0],
            "gt_hw": gt_sel[:, 1],
            "gt_dc": gt_sel[:, 2],
            "gt_dw": gt_sel[:, 3],
            "pred_hc": pred_sel[:, 0],
            "pred_hw": pred_sel[:, 1],
            "pred_dc": pred_sel[:, 2],
            "pred_dw": pred_sel[:, 3],
        })
        df.to_csv(save_dir / f"{prefix}roi{roi_idx}_bestpair_rows.csv", index=False, encoding="utf-8-sig")

    return {
        "mean_best_iou": mean_best_iou,
        "rmse": rmse,
    }


def load_model_with_custom_objects(model_path: Path, P: int, K: int):
    custom_objects = {
        "loss": huber_bestpair_loss(P=P, K=K, delta=0.05),
        "bbox_iou": bbox_iou_metric_maxPK(P=P, K=K),
    }

    model = keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=False,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=huber_bestpair_loss(P=P, K=K, delta=0.05),
        metrics=[bbox_iou_metric_maxPK(P=P, K=K)],
        jit_compile=False,
    )
    return model


def main():
    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    if not best_x_path.exists() or not best_y_path.exists() or not best_z_path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없음:\n{best_x_path}\n{best_y_path}\n{best_z_path}"
        )

    X_test, y_test, K = load_test_data()
    print("X_test:", X_test.shape, "y_test:", y_test.shape, "K:", K)

    P = 3  # 기존 학습 코드와 동일

    best_x = load_model_with_custom_objects(best_x_path, P=P, K=K)
    best_y = load_model_with_custom_objects(best_y_path, P=P, K=K)
    best_z = load_model_with_custom_objects(best_z_path, P=P, K=K)

    for name, model, roi_idx in [("X", best_x, 0), ("Y", best_y, 1), ("Z", best_z, 2)]:
        ds_test = make_ds_roi(X_test, y_test, roi_idx=roi_idx, K=K)
        res = model.evaluate(ds_test, verbose=1)
        print(f"== Test {name} ==")
        print(dict(zip(model.metrics_names, res)))

    out_dir = current_dir / f"eval_final_only_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("save to:", out_dir)

    eval_bbox_roi_bestpair(best_x, X_test, y_test, roi_idx=0, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="x_")
    eval_bbox_roi_bestpair(best_y, X_test, y_test, roi_idx=1, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="y_")
    eval_bbox_roi_bestpair(best_z, X_test, y_test, roi_idx=2, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="z_")


if __name__ == "__main__":
    main()