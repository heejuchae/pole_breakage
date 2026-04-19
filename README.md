# 파단 패턴 ResNet 파이프라인

CSV·편집 라벨에서 학습용 배열을 만들고, **축별 bbox+confidence 회귀**를 학습한 뒤, 그 가중치를 특징 추출기로 써서 **Break / Normal 이진 분류**를 학습하는 흐름입니다.

## 전체 구성 (실행 순서)

| 단계 | 스크립트 | 한 줄 요약 |
|------|-----------|------------|
| 1 | `6. training_data_bbox_extracted.py` | `4. merge_data`, `9. edit_data` → `5. train_data/`에 `npy` 생성 |
| 2 | `7. train_break_pattern_resnet_bbox_confidence.py` | 동일 2D 입력으로 **x/y/z 축별** bbox+confidence ResNet 학습 |
| 3 | `8. train_break_pattern_resnet_binary_from_bbox.py` | 7번 체크포인트 3개를 **고정 백본**으로 이진 분류기 학습 |

구버전·실험용 코드는 로컬 `_archive/`에 둘 수 있습니다 (Git 비포함).

---

## `7. train_break_pattern_resnet_bbox_confidence.py` (bbox + confidence)

### 역할

**파단 여부와 무관하게**, 2D 패턴 이미지 한 장(`304×19×3` 등, `6`번에서 만든 형식)을 보고 **각 ROI 축(x, y, z)마다** “상자가 어디 있는지”와 “그 예측을 얼마나 믿을지”를 동시에 맞추는 **회귀(+신뢰도) 단계**입니다.  
즉, **클래스 Break/Normal을 직접 예측하는 스크립트가 아니라**, 이후 단계나 후처리에서 쓰일 **공간 위치 표현**을 학습합니다.

### 입력·출력

- **입력 디렉터리**: `5. train_data/train`, `5. train_data/test`
  - `break_imgs_train.npy`, `break_imgs_test.npy`: 샘플 수 × 높이 × 폭 × 채널
  - `break_labels_train.npy`, `break_labels_test.npy`: 첫 열은 클래스 등 메타, 나머지는 **bbox·마스크가 `K`개 ROI × 축별로 펼쳐진 벡터** (`y` 길이 `1 + 15×K` 구조, 스크립트 내 `slice_roi_targets` 참고)
- **모델**: 축마다 **독립된 ResNet18 유사** 네트워크 3개 (`model_x`, `model_y`, `model_z`).
- **헤드 출력 차원**: ROI마다 **5개 값 × 예측 슬롯 수 `P`** — `(hc, hw, dc, dw)` 정규화 박스 + **confidence** 한 채널.
- **손실**: bbox 쪽은 GT 박스와의 매칭에 기반한 **Huber(best-pair)** 스타일; confidence는 **BCE**. 코드 상단 `CONF_WEIGHT`, `IOU_THRESHOLD_FOR_CONF`로 confidence 항 가중·IoU 조건을 조절합니다.
- **증강**: `USE_AUGMENTATION`이 켜지면 학습 시에만 밝기·대비·가우시안 노이즈·작은 이동 등(좌우/상하 플립 관련 코드는 주석 처리된 구간이 있음).

### 저장되는 것

- **`7. resnet_runs/checkpoints/`**
  - `best_x.keras`, `best_y.keras`, `best_z.keras` — 축별 최적 가중치
- 학습 로그·그래프 등은 같은 `7. resnet_runs/` 트리 아래에 쌓입니다(`.gitignore`로 원격 미포함).

### 실행 환경 메모

- **Windows**: 기본 경로는 WSL2로 GPU 학습을 넘기도록 되어 있고, **로컬에서 직접 돌리려면** `--local`을 주거나, Linux/WSL에서는 보통 로컬 경로로 바로 학습합니다.
- **구버전 체크포인트**: docstring대로 **4×P 형태의 옛 체크포인트와는 호환되지 않습니다.** 반드시 이 스크립트로 다시 학습해야 합니다.

---

## `8. train_break_pattern_resnet_binary_from_bbox.py` (이진 분류, bbox 특징 기반)

### 역할

**7번에서 이미 학습된 x/y/z 세 모델**을 “특징 추출기”로만 쓰고, 그 위에 **Break vs Normal 이진 분류 헤드**를 얹어 학습합니다.  
입력 이미지는 7번과 **동일한 `npy`**, 라벨은 **`y[:, 0]`의 클래스**만 사용합니다.

### 7번과의 관계

1. **`7. resnet_runs/checkpoints/best_{x,y,z}.keras`가 반드시 존재**해야 합니다. 없으면 `build_frozen_backbone`에서 `FileNotFoundError`가 납니다.
2. 각 체크포인트 전체 모델에서 **GlobalAveragePooling 직전(또는 `_gap` 레이어) 출력**까지를 잘라 **동결 서브모델**로 만듭니다(`trainable=False`).
3. 세 축의 512차원 특징을 **투영(256) → 축 게이트(softmax 3)** 로 가중합한 뒤, 원본 512×3·가중 특징·게이트를 **concat**하여 MLP로 **sigmoid 1출력** 분류기를 구성합니다(`build_binary_classifier`).

### 학습 절차 요약

- **교차 검증 느낌의 split**: `StratifiedShuffleSplit`으로 여러 번 나눠 각 split마다 `split_{id}/best_binary.keras` 등을 저장.
- **각 split 내 2단계**: (1) 백본 동결·헤드만 `HEAD_LR`로 짧게, (2) 마지막 스테이지 일부만 풀고 `FT_LR`로 파인튜닝(`set_backbone_trainability`).
- **클래스 가중**: break 쪽에 더 큰 가중(`class_weight` 1 vs 4).
- **검증 목표**: `TARGET_PRECISION`, `TARGET_RECALL`에 맞춰 threshold를 찾고, 그에 맞는 `val_target_score` 등으로 체크포인트·얼리스탑(`DelayedModelCheckpoint`, `DelayedEarlyStopping`, `ValidationConstraintMetric`).

### 저장되는 것

- **`7. resnet_runs_binary/`** (이름은 `7.`으로 시작하지만 내용은 8번 전용 실행 결과)
  - `checkpoints/split_*/best_binary.keras`, `final_retrain/best_final_binary.keras` 등
  - TensorBoard 로그, 히스토리 플롯·JSON

역시 `.gitignore`에 있어 Git에는 올리지 않습니다.

---

## 환경

- **Python**: 3.12 권장 (`requirements.txt` 주석 기준)
- **설치**

```bash
pip install -r requirements.txt
```

- **GPU**: TensorFlow 2.16+에 맞는 CUDA 설치 ([TensorFlow 설치 가이드](https://www.tensorflow.org/install))

## 데이터 디렉터리 (Git 제외)

- `4. merge_data/`, `9. edit_data/` — `6`번 입력
- `5. train_data/` — `6` 실행 후 생성; **`7`, `8`의 공통 입력**

## 실행 예시 (저장소 루트에서)

```bash
python "6. training_data_bbox_extracted.py"
python "7. train_break_pattern_resnet_bbox_confidence.py"
python "8. train_break_pattern_resnet_binary_from_bbox.py"
```

- Windows에서 7번만 **WSL 위임** 동작이 있을 수 있으니, 로컬 GPU로 돌릴 때는 `--local` 또는 Linux/WSL 환경을 사용합니다.
