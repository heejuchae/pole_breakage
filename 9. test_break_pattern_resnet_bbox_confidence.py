import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

import tensorflow as tf
from tensorflow import keras

# =============================
# 0) 너가 바꿔야 하는 경로 2개
# =============================
PROJECT_ROOT = Path("/home/heejuchae/work/make_ai")  # make_ai 루트
OUT_DIR = PROJECT_ROOT / "eval_bestpair_20260405-212820"  # 로그에 나온 폴더로 변경

# 체크포인트
CKPT_DIR = PROJECT_ROOT / "7. resnet_runs" / "checkpoints"
BEST_X = CKPT_DIR / "best_x.keras"
BEST_Y = CKPT_DIR / "best_y.keras"
BEST_Z = CKPT_DIR / "best_z.keras"

# 테스트 데이터 (네 학습 스크립트와 동일)
TEST_X = PROJECT_ROOT / "5. train_data" / "test" / "break_imgs_test.npy"
TEST_Y = PROJECT_ROOT / "5. train_data" / "test" / "break_labels_test.npy"

# =============================
# 1) 유틸: bbox 변환 / IoU
# =============================
def center_to_corners(b):
    # b: (...,4) [hc, hw, dc, dw], normalized (0..1)
    hc, hw, dc, dw = b[...,0], b[...,1], b[...,2], b[...,3]
    h1 = hc - 0.5*hw
    h2 = hc + 0.5*hw
    d1 = dc - 0.5*dw
    d2 = dc + 0.5*dw
    return h1, h2, d1, d2

def iou_2d(pred, true, eps=1e-9):
    # pred, true: (...,4) [hc, hw, dc, dw]
    ph1, ph2, pd1, pd2 = center_to_corners(pred)
    th1, th2, td1, td2 = center_to_corners(true)

    ih1 = np.maximum(ph1, th1)
    ih2 = np.minimum(ph2, th2)
    id1 = np.maximum(pd1, td1)
    id2 = np.minimum(pd2, td2)

    inter_h = np.maximum(0.0, ih2 - ih1)
    inter_d = np.maximum(0.0, id2 - id1)
    inter = inter_h * inter_d

    area_p = np.maximum(0.0, ph2-ph1) * np.maximum(0.0, pd2-pd1)
    area_t = np.maximum(0.0, th2-th1) * np.maximum(0.0, td2-td1)
    union = area_p + area_t - inter
    return inter / (union + eps)

def draw_box(ax, bbox_norm, H, W, edgecolor="r", lw=2, label=None):
    hc, hw, dc, dw = bbox_norm
    h1 = (hc - 0.5*hw) * H
    h2 = (hc + 0.5*hw) * H
    d1 = (dc - 0.5*dw) * W
    d2 = (dc + 0.5*dw) * W

    # clip
    h1, h2 = np.clip([h1, h2], 0, H)
    d1, d2 = np.clip([d1, d2], 0, W)

    rect = Rectangle((d1, h1), (d2-d1), (h2-h1), fill=False, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(rect)
    if label is not None:
        ax.text(d1, max(0, h1-3), label, fontsize=9, color=edgecolor)

# =============================
# 2) y에서 ROI 타겟 슬라이스 (네 스크립트와 동일 구조)
# =============================
def infer_K(y):
    # y: (N, 1+15K)
    K = int((y.shape[1] - 1) // 15)
    assert 1 + 15*K == y.shape[1]
    return K

def slice_roi_targets(y, roi_idx: int, K: int):
    bbox_dim = 12 * K       # 3*K*4
    mask_dim = 3 * K
    bbox_flat = y[:, 1 : 1 + bbox_dim].astype("float32")
    mask_flat = y[:, 1 + bbox_dim : 1 + bbox_dim + mask_dim].astype("float32")

    bbox = bbox_flat.reshape(-1, 3, K, 4)
    mask = mask_flat.reshape(-1, 3, K)

    bbox_r = bbox[:, roi_idx, :, :]          # (N,K,4)
    mask_r = mask[:, roi_idx, :]             # (N,K)
    return bbox_r, mask_r

# =============================
# 3) 샘플별로: GT(K개) vs Pred(P개) + bestpair 표시
# =============================
def visualize_samples(X, y, models, P=3, max_show=12, pick="worst"):
    """
    pick: "worst" / "best" / "random"
    저장: OUT_DIR/overlay_vis_TIMESTAMP/
    """
    H, W = X.shape[1], X.shape[2]
    K = infer_K(y)

    # 결과 저장 폴더
    save_dir = OUT_DIR / f"overlay_vis_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # OUT_DIR에 eval csv가 있으면 그걸 이용해서 worst/best를 고르고,
    # 없으면 그냥 random으로 뽑음.
    csvs = {
        "x": OUT_DIR / "x_roi0_bestpair_rows.csv",
        "y": OUT_DIR / "y_roi1_bestpair_rows.csv",
        "z": OUT_DIR / "z_roi2_bestpair_rows.csv",
    }

    # 기준은 x의 best_iou로 샘플 정렬 (원하면 y/z로 바꿔도 됨)
    if csvs["x"].exists():
        dfx = pd.read_csv(csvs["x"])
        valid = dfx[dfx["has_gt"] == 1].copy()
        if len(valid) == 0:
            idxs = np.random.choice(len(X), size=min(max_show, len(X)), replace=False)
        else:
            valid = valid.sort_values("best_iou", ascending=(pick=="worst"))
            idxs = valid["idx"].values[:min(max_show, len(valid))]
    else:
        idxs = np.random.choice(len(X), size=min(max_show, len(X)), replace=False)

    # ROI별 GT 준비
    gt = {}
    for axis, roi_idx in [("x",0),("y",1),("z",2)]:
        gt_bbox, gt_mask = slice_roi_targets(y, roi_idx, K)
        gt[axis] = (gt_bbox, gt_mask)

    for n, idx in enumerate(idxs):
        img = X[idx]  # (H,W,3)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"sample idx={idx}  (pick={pick})", y=1.02)

        for ax_i, (axis, roi_idx) in enumerate([("x",0),("y",1),("z",2)]):
            ax = axes[ax_i]
            ax.imshow(img)  # origin='upper' 기본값: 네 hc flip 규칙과 잘 맞음
            ax.set_title(axis.upper())
            ax.set_axis_off()

            gt_bbox, gt_mask = gt[axis]
            gt_boxes = gt_bbox[idx]            # (K,4)
            gt_valid = gt_mask[idx] > 0.5      # (K,)

            # pred: (5*P,) -> (P,5)
            pred = models[axis].predict(img[None, ...], verbose=0)[0].reshape(P, 5)
            pred_boxes = pred[:, :4]

            # --------
            # best-pair 찾기 (pred P vs GT K 중 IoU 최대)
            # --------
            if gt_valid.any():
                gt_list = gt_boxes[gt_valid]  # (K',4)
                # iou_mat: (P,K')
                iou_mat = np.stack([iou_2d(pred_boxes[p], gt_list) for p in range(P)], axis=0)
                p_idx, g_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                best_iou = float(iou_mat[p_idx, g_idx])
            else:
                p_idx, g_idx, best_iou = None, None, None

            # --------
            # GT 박스들(빨강)
            # --------
            for k_i, b in enumerate(gt_boxes):
                if gt_valid[k_i]:
                    draw_box(ax, b, H, W, edgecolor="r", lw=2, label=f"GT{k_i}")

            # --------
            # Pred 박스들(파랑): 전부 그리되 best pred는 두껍게
            # --------
            for p in range(P):
                lw = 3 if (p_idx is not None and p == p_idx) else 1.5
                label = f"P{p}"
                draw_box(ax, pred_boxes[p], H, W, edgecolor="b", lw=lw, label=label)

            if best_iou is not None:
                ax.text(2, 12, f"bestIoU={best_iou:.3f}  bestP={p_idx}", color="yellow",
                        fontsize=10, bbox=dict(facecolor="black", alpha=0.4, pad=2))

        out_png = save_dir / f"overlay_{n:03d}_idx{idx}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("✅ saved overlay images to:", save_dir)
    print("Tip: 파일탐색기/VSCode에서 해당 폴더 열어서 쭉 넘겨보면 됨.")

# =============================
# 4) main
# =============================
def main():
    X = np.load(TEST_X).astype(np.float32)
    y = np.load(TEST_Y).astype(np.float32)

    print("X_test:", X.shape, "y_test:", y.shape)

    # 모델 로드 (compile은 필요없음: predict만 할 거라서)
    models = {
        "x": keras.models.load_model(BEST_X, compile=False),
        "y": keras.models.load_model(BEST_Y, compile=False),
        "z": keras.models.load_model(BEST_Z, compile=False),
    }

    # worst 케이스부터 눈으로 보면 빠르게 문제 찾음
    visualize_samples(X, y, models, P=3, max_show=15, pick="worst")

    # 잘 되는 케이스도 확인하고 싶으면:
    # visualize_samples(X, y, models, P=3, max_show=15, pick="best")

if __name__ == "__main__":
    main()
