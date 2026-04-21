
# ===== cell 1 =====
import os, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

# ✅ 여기만 수정
# 예) PROJECT_DIR = r"C:\Users\PC\Desktop\파단 데이터 측정"
PROJECT_DIR = os.getcwd()

current_dir = PROJECT_DIR
print("current_dir:", current_dir)

MERGE_ROOT = Path(current_dir) / "4. merge_data"
EDIT_ROOT  = Path(current_dir) / "9. edit_data"
BREAK_JSON_ROOT = EDIT_ROOT / "break"

# ===== cell 2 =====
def load_crop_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """크롭된 CSV 파일 로드."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print("❌ read_csv failed:", csv_path)
        print("   err =", repr(e))
        return None

def infer_filled_degree_window(degree_series: pd.Series, step: float = 5.0, span: float = 90.0):
    """
    실제로 데이터가 가장 많이 채워진 90도 구간을 찾는다.
    예:
      30~120, 50~140, 330~60 같은 경우를 자동으로 찾음

    return
    ------
    start_deg : float
        원본 degree 기준 시작 각도
    target_bins : np.ndarray
        시작점부터 90도까지의 5도 bin (총 19개)
    """
    deg = np.asarray(degree_series, dtype=np.float32)
    deg = np.round(deg / step) * step
    deg = np.mod(deg, 360.0).astype(np.float32)

    all_bins = np.arange(0.0, 360.0, step, dtype=np.float32)   # 0,5,10,...355
    uniq, counts = np.unique(deg, return_counts=True)
    count_map = {float(u): int(c) for u, c in zip(uniq, counts)}

    n_bins = int(span / step) + 1   # 90도면 19개
    best = None

    for start in all_bins:
        bins = np.array([(start + step * i) % 360.0 for i in range(n_bins)], dtype=np.float32)
        coverage = sum(1 for b in bins if float(b) in count_map)
        total_count = sum(count_map.get(float(b), 0) for b in bins)

        score = (coverage, total_count)

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "start_deg": float(start),
                "bins": bins,
                "coverage": int(coverage),
                "total_count": int(total_count),
            }

    return best["start_deg"], best["bins"]

def has_any_zero_column(img: np.ndarray, zero_eps: float = 1e-6) -> bool:
    """
    img: (H, W, 3)
    한 column 전체가 모든 채널에서 0이면 True
    """
    abs_img = np.abs(img)
    zero_col_mask = np.all(abs_img <= zero_eps, axis=(0, 2))  # (W,)
    return bool(zero_col_mask.any())

def prepare_sequence_from_csv(
    csv_path: str,
    sort_by: str = 'height',
    feature_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
    max_height: Optional[int] = None
) -> Optional[Tuple[np.ndarray, Dict]]:
    """CSV 파일에서 시퀀스 데이터 생성 (height, degree, x, y, z 포함, 각각 0~1 정규화)."""
    df = load_crop_csv(csv_path)
    if df is None:
        return None
    
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None

    # 정렬 : 전주 높이 정렬이 기본
    if sort_by == 'height':
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    elif sort_by == 'degree':
        df = df.sort_values(['degree', 'height']).reset_index(drop=True)
    else:
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)

    # 파일별 min/max
    if feature_min_max is None:
        feature_min_max = {
            'height': (df['height'].min(), df['height'].max()),
            'degree': (df['degree'].min(), df['degree'].max()),
            'x_value': (df['x_value'].min(), df['x_value'].max()),
            'y_value': (df['y_value'].min(), df['y_value'].max()),
            'z_value': (df['z_value'].min(), df['z_value'].max()),
        }

    # 0~1 정규화
    value_cols = ['x_value', 'y_value', 'z_value']
    for col in value_cols:
        vmin, vmax = feature_min_max[col]
        if vmax > vmin:
            df[col] = (df[col].astype(np.float32) - vmin) / (vmax - vmin)
        else:
            df[col] = 0.0

    # height/degree "값 그대로"를 축으로 grid 생성 (pivot 사용)
    heights = np.sort(df['height'].unique())
    degrees = np.arange(90.0, 180.0 + 5.0, 5.0, dtype=np.float32)  # 19개 고정
    H, W = len(heights), len(degrees)

    # 1) degree를 5도 bin으로 스냅
    df['degree'] = (np.round(df['degree'] / 5.0) * 5.0).astype(np.float32)
    df['degree'] = np.mod(df['degree'], 360.0).astype(np.float32)

    # 2) 실제로 채워진 90도 구간 자동 탐색
    window_start_deg, source_bins = infer_filled_degree_window(df['degree'], step=5.0, span=90.0)

    # 3) 찾은 90도 구간을 canonical 90~180으로 정렬
    df['degree_aligned'] = (((df['degree'] - window_start_deg) % 360.0) + 90.0).astype(np.float32)

    # 4) 혹시 범위 밖 값 제거
    df = df[(df['degree_aligned'] >= 90.0) & (df['degree_aligned'] <= 180.0)].copy()

    # 5) canonical degree 축
    degrees = np.arange(90.0, 180.0 + 5.0, 5.0, dtype=np.float32)

    # degree 정규화 범위도 canonical 축으로 맞춤
    feature_min_max['degree'] = (90.0, 180.0)

    def make_grid(col: str) -> np.ndarray:
        g = (df.pivot_table(index='height', columns='degree_aligned', values=col, aggfunc='mean')
            .reindex(index=heights, columns=degrees)
            .to_numpy(dtype=np.float32))
        return np.nan_to_num(g, nan=0.0)

    x_grid = make_grid('x_value')
    y_grid = make_grid('y_value')
    z_grid = make_grid('z_value')

    img = np.stack([x_grid, y_grid, z_grid], axis=-1).astype(np.float32)

    # 완전 0 열이 하나라도 있으면 버림
    if has_any_zero_column(img, zero_eps=1e-6):
        return None

    metadata = {
        'grid_shape': img.shape,
        'num_points': int(len(df)),
        'original_length': int(len(df)),
        'unique_heights': int(H),
        'unique_degrees': int(W),
        'height_values': heights.tolist(),
        'degree_values': degrees.tolist(),
        'source_degree_start': float(window_start_deg),
        'source_degree_bins': source_bins.tolist(),
        'feature_min_max': {k: list(v) for k, v in feature_min_max.items()},
    }


    return img, metadata

# ===== cell 3 =====
#데이터 특성확인

csv_path = Path(current_dir) / "4. merge_data" / "break" / "강원동해-202209" / "0621R481" / "0621R481_2_OUT_processed.csv"
csv_path = str(csv_path)

df = load_crop_csv(csv_path)

if df is None:
    print("❌ CSV 로드 실패 (파일 경로/인코딩/빈 파일 확인)")
else:
    print("✅ df.shape (rows, cols):", df.shape)
    print("\n✅ columns:", list(df.columns))

    print("\n✅ dtypes:")
    print(df.dtypes)

    print("\n✅ head(5):")
    print(df.head())

    print("\n✅ tail(5):")
    print(df.tail())

    # 결측치 개수
    print("\n✅ missing values per column:")
    print(df.isna().sum())

    # 필요한 컬럼만 quick view
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    present = [c for c in required_cols if c in df.columns]
    if present:
        print("\n✅ required cols stats:")
        print(df[present].describe())
    else:
        print("\n⚠️ required cols not found in this CSV")

# 디버그 시각화는 필요할 때만 수동 실행
if False:
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_xyz_contours(img, meta, break_point=None, n_levels=20):
        degrees = np.arange(90.0, 180.0 + 5.0, 5.0, dtype=float)
        heights = np.array(meta["height_values"], dtype=float)

        D, H = np.meshgrid(degrees, heights)

        titles = ["X Value Contour", "Y Value Contour", "Z Value Contour"]
        cbar_labels = ["X Value (m)", "Y Value (m)", "Z Value (m)"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        for ch in range(3):
            ax = axes[ch]
            Z = img[..., ch]
            vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
            levels = np.linspace(vmin, vmax, 20)

            cf = ax.contourf(D, H, Z, levels=levels, cmap="hot")
            ax.contour(D, H, Z, levels=levels, linewidths=0.6)

            cbar = fig.colorbar(cf, ax=ax)
            cbar.set_label(cbar_labels[ch])

            ax.set_title(titles[ch])
            ax.set_xlabel("Degree (°)")
            ax.set_ylabel("Height (m)")

        plt.show()

    result = prepare_sequence_from_csv(csv_path)
    img, meta = result
    plot_xyz_contours(img, meta, break_point=(123.8, 0.939))

# ===== cell 5 =====
# PROJECT_DIR : 각 지역
# pole_dir : 각 전신주 번호

# 예시 break 폴더안의 csv파일을 모두 긁어 모은다.


from pathlib import Path
from typing import List, Tuple
import os

PROJECT_DIR = os.getcwd()
current_dir = PROJECT_DIR

def collect_all_crop_files(data_dir: str, is_break: bool) -> List[Tuple[str, str, str, int]]:
    """
    9. edit_data/{break|normal} 아래에서 *_roi_info.json 파일 수집.
    반환: (json_path, project_name, poleid, label)
    """
    base_dir = Path(current_dir) / data_dir
    base_dir = base_dir / ("break" if is_break else "normal")

    out = []

    for project_dir in base_dir.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name

        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            poleid = pole_dir.name

            patterns = ["*_OUT_processed.csv"]

            for pat in patterns:
                for json_file in pole_dir.glob(pat):
                    out.append((str(json_file), project_name, poleid, 1 if is_break else 0))

    return out

# ===== cell 6 =====
def parse_roi_bbox(roi_info: dict, k: int):
    """
    return: List[[hc, hw, dc, dw], ...]
    (구버전 삭제 기준: roi_{k}_regions만 처리)
    """
    out = []
    regions = roi_info.get(f"roi_{k}_regions")
    if not isinstance(regions, list):
        return out

    for r in regions:
        hmin = r.get("height_min")
        hmax = r.get("height_max")
        dmin = r.get("degree_min")
        dmax = r.get("degree_max")
        if None in (hmin, hmax, dmin, dmax):
            continue
        try:
            hmin, hmax, dmin, dmax = map(float, (hmin, hmax, dmin, dmax))
        except Exception:
            continue

        hc = (hmin + hmax) / 2.0
        hw = (hmax - hmin)
        dc = (dmin + dmax) / 2.0
        dw = (dmax - dmin)
        out.append([hc, hw, dc, dw])

    return out

# ===== cell 7 =====
import re
from pathlib import Path
from typing import Optional

def get_sample_id_from_csv(csv_path: str) -> Optional[str]:
    # 0621R481_2_OUT_processed.csv -> 0621R481_2
    name = Path(csv_path).name
    m = re.match(r"(.+)_OUT_processed\.csv$", name)
    return m.group(1) if m else None

# ===== cell 8 =====
BREAK_JSON_ROOT = Path(current_dir) / "9. edit_data" / "break"

def match_roi_json_from_csv(csv_path: str) -> Optional[str]:
    p = Path(csv_path)
    sample_id = get_sample_id_from_csv(csv_path)
    if sample_id is None:
        return None
    
    # csv: .../merge_data/break/<project>/<poleid>/<sample_id>_OUT_processed.csv
    # json: .../edit_data/break/<project>/<poleid>/<sample_id>_OUT_processed_roi_info.json
    project = p.parents[1].name   # <project>
    poleid  = p.parents[0].name   # <poleid>

    roi_json = Path(current_dir) / "9. edit_data" / "break" / project / poleid / f"{sample_id}_OUT_processed_roi_info.json"
    return str(roi_json) if roi_json.exists() else None

def load_roi_info_json(json_path: str) -> Optional[Dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ===== cell 9 =====
def expand_rois_from_roi_info(roi_info: dict):
    out = []
    for k in (0, 1, 2):
        for b in parse_roi_bbox(roi_info, k):  # ✅ 항상 list
            out.append((k, b))
    return out

def normalize_bbox_center_width(bbox, feature_min_max, source_degree_start: Optional[float] = None):
    hc, hw, dc, dw = bbox

    h_min, h_max = feature_min_max['height']
    d_min, d_max = feature_min_max['degree']

    if source_degree_start is not None:
        d1 = dc - 0.5 * dw
        d2 = dc + 0.5 * dw

        d1 = (((d1 - source_degree_start) % 360.0) + 90.0)
        d2 = (((d2 - source_degree_start) % 360.0) + 90.0)

        if d2 < d1:
            d1, d2 = d2, d1

        dc = (d1 + d2) / 2.0
        dw = d2 - d1

    h_rng = (h_max - h_min)
    d_rng = (d_max - d_min)

    if h_rng <= 0 or d_rng <= 0:
        return None

    hc_n = (hc - h_min) / h_rng
    hw_n = hw / h_rng
    dc_n = (dc - d_min) / d_rng
    dw_n = dw / d_rng

    hc_n = float(np.clip(hc_n, 0.0, 1.0))
    dc_n = float(np.clip(dc_n, 0.0, 1.0))
    hw_n = float(np.clip(hw_n, 0.0, 1.0))
    dw_n = float(np.clip(dw_n, 0.0, 1.0))

    return [hc_n, hw_n, dc_n, dw_n]

# ===== cell 11 =====
def _check_sync(stage, csv_path, imgs, labels, metadata_list):
    if not (len(imgs) == len(labels) == len(metadata_list)):
        print("\n[SYNC ERROR]", stage)
        print(" csv_path:", csv_path)
        print(" lens:", len(imgs), len(labels), len(metadata_list))
        raise AssertionError("len(imgs)!=len(labels)!=len(metadata_list)")

# ===== cell 12 =====
PROJECT_DIR = os.getcwd()
current_dir = PROJECT_DIR
img_height = 304

from scipy.ndimage import zoom

def resize_img_height(img: np.ndarray, target_h: int = 304) -> np.ndarray:
    h, w, c = img.shape
    if h == target_h:
        return img
    zoom_h = target_h / h
    return zoom(img, (zoom_h, 1.0, 1.0), order=1).astype(np.float32)  # W/C는 그대로

def process_cropped_data(
    data_dir: str = "9. edit_data",
    output_dir: str = "5. train_data",
    sort_by: str = 'height',
    min_points: int = 200,
    max_points: int = 400,
):
    
    import csv

    """edit_data에서 데이터를 처리하여 시퀀스 데이터 준비."""
    # 모든 csv파일을 불러옴

    output_path = Path(current_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print("파단 데이터 파일 수집 중...")
    crop_files   = collect_all_crop_files(data_dir, True)
    normal_files = collect_all_crop_files(data_dir, False)

    if normal_files:
        unique_files = len({f[0] for f in normal_files})
        print(f"  정상 데이터 파일: {unique_files}개")
        print(f"  정상 데이터 샘플: {len(normal_files)}개")

    if not crop_files and not normal_files:
        print("처리할 파일을 찾을 수 없습니다.")
        return

    print(f"\n총 {len(crop_files)}개의 파단 파일, {len(normal_files)}개의 정상 샘플 발견")
    print("\n각 파일별 min/max로 정규화합니다...")

    imgs = []
    labels_cls = []      # ✅ 기존 0/1 분류용
    labels = []          # ✅ (cls, bh, bd) 멀티라벨
    metadata_list = []
    failed_files = []
    break_image_set = set()   # ✅ 파단 '이미지(파일)' 기준 집계용

    K = 10   # roi별 최대 bbox 개수
    LABEL_DIM = 1 + (3 * K * 4) + (3 * K)  # cls + bbox_tensor + mask_tensor

    # ---- 파단 처리 ----
    if crop_files:

        for csv_path, project_name, poleid, label in tqdm(crop_files, desc="파단 데이터 처리"):

            roi_bboxes = []

            result = prepare_sequence_from_csv(
                csv_path=csv_path,
                sort_by=sort_by,
                feature_min_max=None
            )

            if result is None:
                continue

            img, metadata = result

            if metadata['original_length'] < min_points or metadata['original_length'] > max_points:
                continue

            img = resize_img_height(img, target_h=304)

            roi_json_path = match_roi_json_from_csv(csv_path)

            roi_info = load_roi_info_json(roi_json_path) if roi_json_path else None

            roi_dict = {0: [], 1: [], 2: []}

            roi_bboxes = expand_rois_from_roi_info(roi_info)  # [(roi_idx, bbox), ...] bbox는 [hc, hw, dc, dw] 형태라고 가정

            for roi_idx, bbox in roi_bboxes:  # [(0,bbox0),(1,bbox1),(2,bbox2)]
                bbox_n = normalize_bbox_center_width(
                    bbox,
                    metadata['feature_min_max'],
                    source_degree_start=metadata.get('source_degree_start')
                )
                if bbox_n is None:
                    continue
                roi_dict[roi_idx].append(bbox_n)

            # 하나도 없으면 스킵
            total = sum(len(v) for v in roi_dict.values())
            if total == 0:
                continue

            # ✅ 고정 텐서로 저장: (3, K, 4) + mask(3, K)
            bbox_tensor = np.zeros((3, K, 4), dtype=np.float32) 
            mask_tensor = np.zeros((3, K), dtype=np.float32)

            for r in (0, 1, 2):
                bxs = roi_dict[r][:K]  # K 초과는 컷
                if len(bxs) > 0:
                    bbox_tensor[r, :len(bxs), :] = np.array(bxs, dtype=np.float32)
                    mask_tensor[r, :len(bxs)] = 1.0

            # ✅ 여기서 “이미지당 1번만” append
            imgs.append(img)
            labels_cls.append(label)

            y_vec = np.zeros((LABEL_DIM,), dtype=np.float32)
            y_vec[0] = float(label)

            off_bbox = 1
            bbox_flat = bbox_tensor.reshape(-1).astype(np.float32)   # (3*K*4,)
            mask_flat = mask_tensor.reshape(-1).astype(np.float32)   # (3*K,)

            y_vec[off_bbox : off_bbox + bbox_flat.size] = bbox_flat
            y_vec[off_bbox + bbox_flat.size : off_bbox + bbox_flat.size + mask_flat.size] = mask_flat

            labels.append(y_vec)

            md = dict(metadata)
            md.update({
                "csv_path": str(csv_path),
                "project_name": project_name,
                "poleid": poleid,
                "img_height": img_height,
                "bbox_K": K,
                "bbox_total": int(total),
                "roi_bbox_counts": {r: len(roi_dict[r]) for r in (0, 1, 2)},
            })
            metadata_list.append(md)
            _check_sync("break_append", csv_path, imgs, labels, metadata_list)
            
            break_image_set.add(str(csv_path))  # 이미지 기준 집계(원하면 유지)

    break_sample_count = len(break_image_set)
    max_normal_samples = break_sample_count * 10
    print(f"\n파단 샘플 수: {break_sample_count}개")
    print(f"정상 샘플 최대 수: {max_normal_samples}개 (파단 샘플의 10배)")


    # ---- 정상 처리 ----
    if normal_files:
        normal_kept = 0
        for csv_path, project_name, poleid, label in tqdm(normal_files, desc="정상 데이터 처리"):
            if normal_kept >= max_normal_samples:
                break

            result = prepare_sequence_from_csv(
                csv_path=csv_path,
                sort_by=sort_by,
                feature_min_max=None,
            )

            if result is None:
                continue

            img, metadata = result
            if not (min_points <= metadata["original_length"] <= max_points):
                continue

            img = resize_img_height(img, target_h=304)

            bbox_tensor = np.zeros((3, K, 4), dtype=np.float32)
            mask_tensor = np.zeros((3, K), dtype=np.float32)

            y_vec = np.zeros((LABEL_DIM,), dtype=np.float32)
            y_vec[0] = 0.0  # label(정상)

            off = 1
            bbox_flat = bbox_tensor.reshape(-1)
            mask_flat = mask_tensor.reshape(-1)

            y_vec[off:off + bbox_flat.size] = bbox_flat
            y_vec[off + bbox_flat.size: off + bbox_flat.size + mask_flat.size] = mask_flat

            imgs.append(img)
            labels_cls.append(0)
            labels.append(y_vec)

            md = dict(metadata)  # ✅ 정상도 metadata 복제해서 1:1 유지
            md.update({
                "csv_path": str(csv_path),
                "project_name": project_name,
                "poleid": poleid,
                "img_height": img_height,
                "bbox_K": K,
                "bbox_total": 0,
                "roi_bbox_counts": {0: 0, 1: 0, 2: 0},
            })
            metadata_list.append(md)

            _check_sync("normal_append", csv_path, imgs, labels, metadata_list)
            
            normal_kept += 1

        print(f"  최종 정상 샘플 수: {len([1 for c in labels_cls if c == 0])}개")

    if not imgs:
        print("생성된 시퀀스 데이터가 없습니다.")
        return
    
    shapes = {}
    for im in imgs:
        shapes[im.shape] = shapes.get(im.shape, 0) + 1

    print("서로 다른 img.shape 개수:", len(shapes))
    for s, cnt in sorted(shapes.items(), key=lambda x: -x[1])[:10]:
        print(s, cnt)

    assert len(imgs) == len(labels) == len(metadata_list)    
    
    X = np.array(imgs, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    assert np.isfinite(y).all()

    print(f"\n시퀀스 데이터 생성 완료:")
    print(f"  총 샘플 수: {len(X)}")
    print(f"  시퀀스 형태: {X.shape}")
    print(f"  실패한 파일: {len(failed_files)}개")



    print("\n[DEBUG] Break ROI/BBox 통계:")

    # 9:1 분할
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.1, random_state=42, stratify=y[:, 0]
    )

    print(f"\n데이터 분할 (학습:테스트 = 9:1):")
    print(f"  학습 샘플 수: {len(X_train)}개")
    print(f"  테스트 샘플 수: {len(X_test)}개")

    # =========================
    # index mapping 생성
    # =========================
    metadata_df = pd.DataFrame(metadata_list).copy()

    train_map_df = metadata_df.iloc[train_indices].copy().reset_index(drop=True)
    train_map_df["idx"] = np.arange(len(train_map_df))
    train_map_df["split"] = "train"
    train_map_df["original_global_idx"] = train_indices

    test_map_df = metadata_df.iloc[test_indices].copy().reset_index(drop=True)
    test_map_df["idx"] = np.arange(len(test_map_df))
    test_map_df["split"] = "test"
    test_map_df["original_global_idx"] = test_indices


    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    np.save(train_dir / "break_imgs_train.npy", X_train)
    np.save(train_dir / "break_labels_train.npy", y_train)

    np.save(test_dir / "break_imgs_test.npy", X_test)
    np.save(test_dir / "break_labels_test.npy", y_test)

    np.save(output_path / "break_imgs.npy", X)
    np.save(output_path / "break_labels.npy", y)

    # =========================
    # index mapping csv 저장
    # =========================
    train_map_df.to_csv(train_dir / "break_index_map_train.csv", index=False, encoding="utf-8-sig")
    test_map_df.to_csv(test_dir / "break_index_map_test.csv", index=False, encoding="utf-8-sig")
    metadata_df.to_csv(output_path / "break_index_map_all.csv", index=False, encoding="utf-8-sig")

    metadata_file = output_path / "break_imgs_metadata.json"
    metadata_dict = {
        'total_samples': len(metadata_list),
        'img_height': img_height,
        'feature_names': ['height', 'degree', 'x_value', 'y_value', 'z_value'],
        'normalization_method': 'per_file',
        'sort_by': sort_by,
        'min_points': min_points,
        'samples': metadata_list,
        'failed_files': failed_files,
        'data_shape': list(X.shape),
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)

    print("\n데이터 저장 완료:")
    print(f"  학습 이미지: {train_dir / 'break_imgs_train.npy'}")
    print(f"  학습 라벨: {train_dir / 'break_labels_train.npy'}")
    print(f"  학습 매핑: {train_dir / 'break_index_map_train.csv'}")

    print(f"  테스트 이미지: {test_dir / 'break_imgs_test.npy'}")
    print(f"  테스트 라벨: {test_dir / 'break_labels_test.npy'}")
    print(f"  테스트 매핑: {test_dir / 'break_index_map_test.csv'}")

    print(f"  전체 이미지: {output_path / 'break_imgs.npy'}")
    print(f"  전체 라벨: {output_path / 'break_labels.npy'}")
    print(f"  전체 매핑: {output_path / 'break_index_map_all.csv'}")
    print(f"  메타데이터: {metadata_file}")

    break_samples = [i for i, l in enumerate(labels_cls) if l == 1]
    normal_samples = [i for i, l in enumerate(labels_cls) if l == 0]
    print("\n데이터 통계:")
    print(f"  총 샘플 수: {len(imgs)}개")
    print(f"    - 파단 샘플: {len(break_samples)}개 (라벨 1)")
    print(f"    - 정상 샘플: {len(normal_samples)}개 (라벨 0)")
    print(f"  파단 샘플 비율: {len(break_samples)/len(imgs)*100:.2f}%")



# ===== cell 13 =====
import json

#bbox를 제대로 인식하고 있는 지 확인하는 코드

with open("9. edit_data/break/강원인제-202306/6562G821/6562G821_2_OUT_processed_roi_info.json","r",encoding="utf-8") as f:
    roi_info = json.load(f)

for r in [0,1,2]:
    key = f"roi_{r}_regions"
    print(key, len(roi_info.get(key, [])))

# ===== cell 14 =====
process_cropped_data(
    data_dir="4. merge_data",
    output_dir="5. train_data",
    sort_by="height",
    min_points=200,
    max_points=400
)

# ===== cell 15 =====
train_seq = Path(current_dir) / "5. train_data" / "train" / "break_imgs_train.npy"
train_lab = Path(current_dir) / "5. train_data" / "train" / "break_labels_train.npy"

X = np.load(train_seq)
y = np.load(train_lab)

bs = 32
Xb, yb = X[:bs], y[:bs]

print("X batch shape:", Xb.shape, Xb.dtype)
print("y batch shape:", yb.shape, yb.dtype)

# ✅ 1) cls 분포 (이게 제일 중요)
cls = yb[:, 0].astype(int)
print("cls counts (in batch):", np.unique(cls, return_counts=True))
print("cls ratio (batch):", cls.mean())

# ✅ 2) bbox 값 범위/이상치 체크
K = 10

# 1) bbox / mask 분리
bbox_flat = yb[:, 1 : 1 + 3*K*4]                 # (B, 120)
mask_flat = yb[:, 1 + 3*K*4 : 1 + 3*K*4 + 3*K]   # (B, 30)

bbox = bbox_flat.reshape(bs, 3, K, 4)            # (B, 3, 10, 4)
mask = mask_flat.reshape(bs, 3, K)               # (B, 3, 10)

print("bbox shape:", bbox.shape, "mask shape:", mask.shape)

# 2) bbox 범위 체크는 bbox 전체에서
print("bbox min/max (batch):", bbox.min(), bbox.max())
print("bbox NaN:", int(np.isnan(bbox).sum()), "Inf:", int(np.isinf(bbox).sum()))
oob = (bbox < 0.0) | (bbox > 1.0)
print("bbox out-of-bound count:", int(oob.sum()))

# 3) 실제 bbox 존재 여부는 mask로 보는 게 정답
print("non-empty bbox per channel (mask sum):", mask.sum(axis=(0,2)))   # (3,)
print("non-empty bbox per sample (mask sum first 10):", mask.sum(axis=(1,2))[:10])

# 4) 샘플 하나 보기
i = 0
print("\nSample[0] cls:", int(yb[i,0]))
print("Sample[0] mask (roi x/y/z):\n", mask[i])
print("Sample[0] bbox (roi x/y/z, first 3 boxes each):\n", bbox[i, :, :3, :])
print("bbox min/max (batch):", bbox.min(), bbox.max())

# NaN/inf 체크
nan_cnt = np.isnan(bbox).sum()
inf_cnt = np.isinf(bbox).sum()
print("bbox NaN count:", int(nan_cnt), "Inf count:", int(inf_cnt))

# [0,1] 범위 밖(oob) 몇 개인지
oob = (bbox < 0.0) | (bbox > 1.0)
print("bbox out-of-bound count:", int(oob.sum()))

# ✅ 3) "bbox가 실제로 채워져 있나?" 확인
# bbox shape: (B, 3, K, 4)
# 마지막 축 4개가 [hc, hw, dc, dw] 이므로
# hw, dw는 마지막 축에서 뽑아야 함
hw = bbox[:, :, :, 1]
dw = bbox[:, :, :, 3]
non_empty = (hw > 0) & (dw > 0)

# print("non-empty bbox per channel in batch (counts):", non_empty.sum(axis=(0, 2)))   # (3,)
# print("non-empty bbox per sample in batch (first 10):", non_empty.sum(axis=(1, 2))[:10])

# # ✅ mask 기준과 같이 비교
# print("mask-based non-empty per channel:", mask.sum(axis=(0, 2)))
# print("mask-based non-empty per sample (first 10):", mask.sum(axis=(1, 2))[:10])

# # ✅ 실제 존재하는 bbox만 따로 보기
# valid_bbox = bbox[mask > 0]   # (num_valid_boxes, 4)
# print("valid bbox count:", len(valid_bbox))

# if len(valid_bbox) > 0:
#     print("valid bbox min/max:", valid_bbox.min(), valid_bbox.max())
#     print("valid hw min/max:", valid_bbox[:, 1].min(), valid_bbox[:, 1].max())
#     print("valid dw min/max:", valid_bbox[:, 3].min(), valid_bbox[:, 3].max())
#     print("valid bbox first 10:\n", valid_bbox[:10])
# else:
#     print("no valid bbox in this batch")

# ✅ 4) 샘플 하나 보기
i = 0
print("\nSample[0] cls:", int(yb[i,0]))
print("Sample[0] bbox (roi x/y/z):\n", bbox[i])