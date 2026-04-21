#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
README용 대표 예시 그림 생성
- Stage 1: contour 기반 + X/Y/Z 세 축 모두 GT bbox / Pred bbox 표시
- Stage 2: TP / FN(없으면 FP) 예시를 4열 레이아웃으로 저장
  * 1~3열: X/Y/Z contour
  * 4열: GT / Pred / Prob / Margin / stage1 IoU 요약
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


BASE_SEED = 42
BATCH = 16
TARGET_PRECISION = 0.80
TARGET_RECALL = 0.90

ROI_NAMES = {0: "x", 1: "y", 2: "z"}
ROI_TITLES = {0: "X-axis", 1: "Y-axis", 2: "Z-axis"}

CONF_WEIGHT = 0.0
IOU_THRESHOLD_FOR_CONF = 0.3


def set_seed(seed: int = BASE_SEED) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slice_roi_targets(y: np.ndarray, roi_idx: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
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


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = TARGET_PRECISION,
    target_recall: float = TARGET_RECALL,
) -> Dict:
    y_true = y_true.astype(np.int32)
    thresholds = np.arange(0.05, 0.951, 0.005)

    best = None
    best_key = None

    for th in thresholds:
        pred = (y_prob >= th).astype(np.int32)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        p_hit = min(prec / max(target_precision, 1e-8), 1.0)
        r_hit = min(rec / max(target_recall, 1e-8), 1.0)
        score = 0.30 * p_hit + 0.70 * r_hit - 0.03 * abs(p_hit - r_hit)

        candidate = {
            "threshold": float(th),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "score": float(score),
        }
        key = (score, rec, prec, -th)

        if best is None or key > best_key:
            best = candidate
            best_key = key

    return best


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
        conf_ok = tf.cast((best_iou_per_sample >= iou_threshold_for_conf) & (has_gt > 0), tf.float32)

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

        iou_mat = iou_2d_from_center_width(pred[:, :, None, :], gt[:, None, :, :])
        iou_masked = tf.where(m[:, None, :] > 0, iou_mat, tf.zeros_like(iou_mat))
        max_iou = tf.reduce_max(iou_masked, axis=[1, 2])

        valid = tf.cast(tf.reduce_any(m > 0, axis=1), tf.float32)
        return tf.reduce_sum(max_iou * valid) / (tf.reduce_sum(valid) + eps)

    return metric


def load_bbox_model(model_path: Path, K: int) -> keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"bbox 모델을 찾을 수 없습니다: {model_path}")

    temp_model = keras.models.load_model(str(model_path), compile=False)
    out_dim = int(temp_model.output_shape[-1])
    if out_dim % 5 != 0:
        raise ValueError(f"bbox 출력 차원이 5의 배수가 아닙니다: {out_dim}")

    P = out_dim // 5
    custom_objects = {
        "loss": huber_bestpair_loss(P=P, K=K),
        "metric": bbox_iou_metric_maxPK(P=P, K=K),
    }
    return keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)


def norm_bbox_to_pixel(bbox: np.ndarray, h: int, w: int) -> Tuple[float, float, float, float]:
    hc, hw, dc, dw = [float(x) for x in bbox]
    y1 = (hc - hw / 2.0) * h
    y2 = (hc + hw / 2.0) * h
    x1 = (dc - dw / 2.0) * w
    x2 = (dc + dw / 2.0) * w

    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)
    return x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)


def get_plot_extent(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = int(channel.shape[0])
    w = int(channel.shape[1])
    degree_axis = np.linspace(90.0, 180.0, w, dtype=np.float32)
    height_axis = np.linspace(0.0, 1.5, h, dtype=np.float32)
    return degree_axis, height_axis


def pick_contour_norm(channel: np.ndarray):
    ch = np.asarray(channel, dtype=np.float32)
    finite = ch[np.isfinite(ch)]
    if finite.size == 0:
        return None

    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        return mcolors.Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6)

    vcenter = float(np.median(finite))
    if vmin < vcenter < vmax:
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def draw_contour_panel(ax, channel: np.ndarray, title: str = ""):
    channel = np.asarray(channel, dtype=np.float32)
    degree_axis, height_axis = get_plot_extent(channel)
    norm = pick_contour_norm(channel)

    if np.allclose(np.nanmax(channel), np.nanmin(channel)):
        ax.imshow(
            channel,
            cmap="RdBu_r",
            aspect="auto",
            origin="lower",
            extent=[degree_axis[0], degree_axis[-1], height_axis[0], height_axis[-1]],
        )
    else:
        levels = np.linspace(float(np.nanmin(channel)), float(np.nanmax(channel)), 25)
        if np.unique(levels).size < 2:
            levels = 25
        ax.contourf(
            degree_axis,
            height_axis,
            channel,
            levels=levels,
            cmap="RdBu_r",
            norm=norm,
        )

    if title:
        ax.set_title(title, fontsize=11, pad=6, fontweight="semibold")
    ax.set_xlabel("Degree", fontsize=9)
    ax.set_ylabel("Height (m)", fontsize=9)
    ax.tick_params(labelsize=8, pad=1.5)
    ax.set_xlim(float(degree_axis[0]), float(degree_axis[-1]))
    ax.set_ylim(float(height_axis[0]), float(height_axis[-1]))
    ax.grid(alpha=0.15)


def choose_best_pred_and_gt(pred_out: np.ndarray, y_reg_roi: np.ndarray, K: int):
    P = pred_out.shape[0] // 5
    pred = pred_out.reshape(P, 5)
    pred_bbox = pred[:, :4]

    gt_bbox = y_reg_roi[:4 * K].reshape(K, 4)
    mask = y_reg_roi[4 * K:5 * K] > 0.5
    valid_gt = gt_bbox[mask]
    if len(valid_gt) == 0:
        raise ValueError("유효 GT bbox가 없습니다.")

    pred_tf = tf.convert_to_tensor(pred_bbox[:, None, :], dtype=tf.float32)
    gt_tf = tf.convert_to_tensor(valid_gt[None, :, :], dtype=tf.float32)
    iou_mat = iou_2d_from_center_width(pred_tf, gt_tf).numpy()

    flat_idx = int(np.argmax(iou_mat))
    pred_idx, gt_idx = np.unravel_index(flat_idx, iou_mat.shape)
    best_iou = float(iou_mat[pred_idx, gt_idx])

    return pred_bbox[pred_idx], valid_gt[gt_idx], best_iou


def collect_stage1_bbox_info_for_samples(
    X_samples: np.ndarray,
    y_samples_full: np.ndarray,
    K: int,
    model_dir: Path,
) -> Dict[int, Dict]:
    sample_infos: Dict[int, Dict] = {}
    if len(X_samples) == 0:
        return sample_infos

    for roi_idx in (0, 1, 2):
        model_path = model_dir / f"best_{ROI_NAMES[roi_idx]}.keras"
        model = load_bbox_model(model_path, K=K)
        _, y_reg_roi = slice_roi_targets(y_samples_full, roi_idx=roi_idx, K=K)
        pred = model.predict(X_samples, batch_size=32, verbose=0)

        for local_idx in range(len(X_samples)):
            mask = y_reg_roi[local_idx, 4 * K:5 * K] > 0.5
            if not np.any(mask):
                continue
            try:
                pred_bbox, gt_bbox, best_iou = choose_best_pred_and_gt(pred[local_idx], y_reg_roi[local_idx], K)
            except ValueError:
                continue

            sample_infos.setdefault(local_idx, {})
            sample_infos[local_idx][roi_idx] = {
                "pred_bbox": pred_bbox.tolist(),
                "gt_bbox": gt_bbox.tolist(),
                "iou": float(best_iou),
            }

    return sample_infos


def save_stage1_example(
    X_test: np.ndarray,
    y_test: np.ndarray,
    K: int,
    model_dir: Path,
    out_dir: Path,
) -> Dict:
    ensure_dir(out_dir)

    roi_targets = {}
    roi_preds = {}

    for roi_idx in (0, 1, 2):
        model_path = model_dir / f"best_{ROI_NAMES[roi_idx]}.keras"
        model = load_bbox_model(model_path, K=K)
        _, y_reg_roi = slice_roi_targets(y_test, roi_idx=roi_idx, K=K)
        roi_targets[roi_idx] = y_reg_roi
        roi_preds[roi_idx] = model.predict(X_test, batch_size=32, verbose=0)

    best_sample = None

    for i in range(len(X_test)):
        if int(y_test[i, 0]) != 1:
            continue

        per_roi = {}
        valid_roi_count = 0
        iou_sum = 0.0

        for roi_idx in (0, 1, 2):
            y_reg_roi = roi_targets[roi_idx][i]
            mask = y_reg_roi[4 * K:5 * K] > 0.5
            if not np.any(mask):
                continue

            try:
                pred_bbox, gt_bbox, best_iou = choose_best_pred_and_gt(roi_preds[roi_idx][i], y_reg_roi, K)
            except ValueError:
                continue

            per_roi[roi_idx] = {
                "pred_bbox": pred_bbox.tolist(),
                "gt_bbox": gt_bbox.tolist(),
                "iou": float(best_iou),
            }
            valid_roi_count += 1
            iou_sum += float(best_iou)

        if valid_roi_count == 0:
            continue

        mean_iou = iou_sum / valid_roi_count
        candidate = {
            "sample_index": i,
            "valid_roi_count": valid_roi_count,
            "mean_iou": mean_iou,
            "per_roi": per_roi,
        }

        key = (valid_roi_count, mean_iou)
        if best_sample is None or key > (best_sample["valid_roi_count"], best_sample["mean_iou"]):
            best_sample = candidate

    if best_sample is None:
        raise RuntimeError("1차 학습 대표 예시를 찾지 못했습니다.")

    sample_index = best_sample["sample_index"]
    img = X_test[sample_index]

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)

    for roi_idx, ax in enumerate(axes):
        channel = img[:, :, roi_idx]
        draw_contour_panel(ax, channel, ROI_TITLES[roi_idx])

        roi_info = best_sample["per_roi"].get(roi_idx)
        if roi_info is not None:
            h, w = channel.shape
            pred_bbox = np.array(roi_info["pred_bbox"], dtype=np.float32)
            gt_bbox = np.array(roi_info["gt_bbox"], dtype=np.float32)

            gx, gy, gw, gh = norm_bbox_to_pixel(gt_bbox, h, w)
            px, py, pw, ph = norm_bbox_to_pixel(pred_bbox, h, w)

            degree_axis, height_axis = get_plot_extent(channel)
            x_scale = (degree_axis[-1] - degree_axis[0]) / w
            y_scale = (height_axis[-1] - height_axis[0]) / h

            ax.add_patch(Rectangle(
                (degree_axis[0] + gx * x_scale, height_axis[0] + gy * y_scale),
                gw * x_scale, gh * y_scale,
                fill=False, edgecolor="lime", linewidth=2.2, label="GT bbox"
            ))
            ax.add_patch(Rectangle(
                (degree_axis[0] + px * x_scale, height_axis[0] + py * y_scale),
                pw * x_scale, ph * y_scale,
                fill=False, edgecolor="red", linewidth=2.2, linestyle="--", label="Pred bbox"
            ))
            ax.set_title(ROI_TITLES[roi_idx], fontsize=11, pad=6, fontweight="semibold")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

            degree_axis, height_axis = get_plot_extent(channel)
            ax.text(
                degree_axis[0] + 4.0,
                height_axis[-1] - 0.10,
                f"IoU = {roi_info['iou']:.4f}",
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="white",
                    edgecolor="#666666",
                    alpha=0.92,
                ),
            )

    fig.suptitle(
        f"Stage 1 Example",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    save_path = out_dir / "stage1_bbox_example.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    best_sample["image_path"] = str(save_path)
    return best_sample


def load_binary_data(project_root: Path):
    train_dir = project_root / "5. train_data" / "train"
    test_dir = project_root / "5. train_data" / "test"

    X_train = np.load(train_dir / "break_imgs_train.npy").astype(np.float32)
    y_train_full = np.load(train_dir / "break_labels_train.npy").astype(np.float32)
    X_test = np.load(test_dir / "break_imgs_test.npy").astype(np.float32)
    y_test_full = np.load(test_dir / "break_labels_test.npy").astype(np.float32)

    y_train = y_train_full[:, 0].astype(np.float32)
    y_test = y_test_full[:, 0].astype(np.float32)
    return X_train, y_train, X_test, y_test, y_test_full


def load_binary_model(project_root: Path) -> keras.Model:
    candidates = [
        project_root / "7. resnet_runs_binary" / "checkpoints" / "final_binary.keras",
        project_root / "7. resnet_runs_binary" / "checkpoints" / "final_retrain" / "best_final_binary.keras",
    ]
    for path in candidates:
        if path.exists():
            return keras.models.load_model(str(path), compile=False)
    raise FileNotFoundError("2차 분류 모델(final_binary.keras 또는 best_final_binary.keras)을 찾지 못했습니다.")


def estimate_threshold(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    _, X_cal, _, y_cal = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=BASE_SEED,
        stratify=y_train.astype(int),
    )
    prob_cal = model.predict(X_cal, batch_size=BATCH, verbose=0).reshape(-1)
    return find_best_threshold(y_true=y_cal, y_prob=prob_cal)


def save_stage2_example(
    project_root: Path,
    out_dir: Path,
    threshold: Optional[float] = None,
) -> Dict:
    ensure_dir(out_dir)

    X_train, y_train, X_test, y_test, y_test_full = load_binary_data(project_root)
    K = int((y_test_full.shape[1] - 1) // 15)
    model = load_binary_model(project_root)

    if threshold is None:
        th_info = estimate_threshold(model, X_train, y_train)
        threshold = float(th_info["threshold"])

    prob = model.predict(X_test, batch_size=BATCH, verbose=0).reshape(-1)
    pred = (prob >= threshold).astype(np.int32)
    y_true = y_test.astype(np.int32)

    tp_idx = np.where((y_true == 1) & (pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (pred == 1))[0]

    chosen_cases = []
    if len(tp_idx) > 0:
        chosen_cases.append(("TP example", int(tp_idx[np.argmax(prob[tp_idx])])))
    if len(fn_idx) > 0:
        chosen_cases.append(("FN example", int(fn_idx[np.argmax(prob[fn_idx])])))
    elif len(fp_idx) > 0:
        chosen_cases.append(("FP example", int(fp_idx[np.argmax(prob[fp_idx])])))
    if not chosen_cases:
        chosen_cases.append(("Example", int(np.argmax(prob))))

    selected_indices = [idx for _, idx in chosen_cases]
    bbox_infos = collect_stage1_bbox_info_for_samples(
        X_samples=X_test[selected_indices],
        y_samples_full=y_test_full[selected_indices],
        K=K,
        model_dir=project_root / "7. resnet_runs" / "checkpoints",
    )

    n_rows = len(chosen_cases)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(18.5, 5.1 * n_rows),
        gridspec_kw={"width_ratios": [1.18, 1.18, 1.18, 1.00]},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    summary_cases = []

    for row_idx, (case_name, sample_idx) in enumerate(chosen_cases):
        img = X_test[sample_idx]
        local_bbox_info = bbox_infos.get(row_idx, {})

        for ch in range(3):
            ax = axes[row_idx, ch]
            channel = img[:, :, ch]
            draw_contour_panel(ax, channel, ROI_TITLES[ch])

            roi_info = local_bbox_info.get(ch)
            if roi_info is not None:
                h, w = channel.shape
                pred_bbox = np.array(roi_info["pred_bbox"], dtype=np.float32)
                gt_bbox = np.array(roi_info["gt_bbox"], dtype=np.float32)

                gx, gy, gw, gh = norm_bbox_to_pixel(gt_bbox, h, w)
                px, py, pw, ph = norm_bbox_to_pixel(pred_bbox, h, w)

                degree_axis, height_axis = get_plot_extent(channel)
                x_scale = (degree_axis[-1] - degree_axis[0]) / w
                y_scale = (height_axis[-1] - height_axis[0]) / h

                ax.add_patch(Rectangle(
                    (degree_axis[0] + gx * x_scale, height_axis[0] + gy * y_scale),
                    gw * x_scale, gh * y_scale,
                    fill=False, edgecolor="lime", linewidth=2.0, label="GT bbox"
                ))
                ax.add_patch(Rectangle(
                    (degree_axis[0] + px * x_scale, height_axis[0] + py * y_scale),
                    pw * x_scale, ph * y_scale,
                    fill=False, edgecolor="red", linewidth=2.0, linestyle="--", label="Pred bbox"
                ))
                ax.set_title(ROI_TITLES[ch], fontsize=12, pad=8, fontweight="semibold")
                ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

        info_ax = axes[row_idx, 3]
        info_ax.axis("off")
        info_ax.text(
            0.06, 0.92, case_name,
            ha="left", va="bottom",
            fontsize=18, fontweight="bold",
            transform=info_ax.transAxes,
        )
        margin = float(prob[sample_idx] - threshold)
        margin_sign = "+" if margin >= 0 else ""

        lines = []
        for ch in range(3):
            if ch in local_bbox_info:
                lines.append(f"{ROI_TITLES[ch]} IoU  {local_bbox_info[ch]['iou']:.4f}")
            else:
                lines.append(f"{ROI_TITLES[ch]} IoU  -")

        gt_text = "Break" if int(y_true[sample_idx]) == 1 else "Normal"
        pred_text = "Break" if int(pred[sample_idx]) == 1 else "Normal"

        panel_text = "\n".join([
            "",
            f"GT        {gt_text}",
            f"Pred      {pred_text}",
            f"Prob      {float(prob[sample_idx]):.4f}",
            f"Threshold {float(threshold):.3f}",
            f"Margin    {margin_sign}{margin:.4f}",
            "",
            *lines,
        ])

        info_ax.text(
            0.06, 0.50, panel_text,
            ha="left", va="center", fontsize=15,
            linespacing=1.40,
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor="#f8f8f8",
                edgecolor="#9a9a9a",
                linewidth=1.4,
            ),
            transform=info_ax.transAxes,
        )

        summary_cases.append({
            "case_name": case_name,
            "sample_index": int(sample_idx),
            "gt_label": int(y_true[sample_idx]),
            "pred_label": int(pred[sample_idx]),
            "pred_prob": float(prob[sample_idx]),
            "margin_from_threshold": float(margin),
            "bbox_info": local_bbox_info,
        })

    fig.suptitle(
        "Stage 2 Example",
        fontsize=20,
        fontweight="bold",
        y=1.03,
    )

    save_path = out_dir / "stage2_classification_example.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "threshold": float(threshold),
        "cases": summary_cases,
        "image_path": str(save_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="README용 1차/2차 대표 예시 그림 생성")
    parser.add_argument("--project-dir", type=str, default=".", help="프로젝트 루트 경로")
    parser.add_argument("--out-dir", type=str, default="readme_examples", help="그림 저장 폴더")
    parser.add_argument("--threshold", type=float, default=None, help="2차 분류 threshold. 없으면 calibration으로 추정")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(BASE_SEED)

    project_root = Path(args.project_dir).resolve()
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    test_dir = project_root / "5. train_data" / "test"
    X_test = np.load(test_dir / "break_imgs_test.npy").astype(np.float32)
    y_test = np.load(test_dir / "break_labels_test.npy").astype(np.float32)
    K = int((y_test.shape[1] - 1) // 15)

    stage1 = save_stage1_example(
        X_test=X_test,
        y_test=y_test,
        K=K,
        model_dir=project_root / "7. resnet_runs" / "checkpoints",
        out_dir=out_dir,
    )

    stage2 = save_stage2_example(
        project_root=project_root,
        out_dir=out_dir,
        threshold=args.threshold,
    )

    summary = {
        "stage1_example": stage1,
        "stage2_example": stage2,
    }

    with open(out_dir / "readme_examples_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[완료] README 예시 그림 저장")
    print(f"- stage1: {stage1['image_path']}")
    print(f"- stage2: {stage2['image_path']}")
    print(f"- summary: {out_dir / 'readme_examples_summary.json'}")


if __name__ == "__main__":
    main()
