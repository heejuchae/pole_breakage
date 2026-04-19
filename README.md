# 파단 패턴 ResNet 파이프라인

CSV·편집 라벨에서 학습용 배열을 만들고, bbox+confidence ResNet을 학습한 뒤, bbox 특징을 이용해 이진 분류 모델을 학습하는 스크립트 모음입니다.

## 구성 (실행 순서)

| 단계 | 스크립트 | 역할 |
|------|-----------|------|
| 1 | `6. training_data_bbox_extracted.py` | `4. merge_data`, `9. edit_data` 기반으로 `5. train_data/`에 학습용 `npy` 등 생성 |
| 2 | `7. train_break_pattern_resnet_bbox_confidence.py` | `5. train_data`의 시퀀스·라벨로 ROI별(x,y,z) bbox+confidence 학습 → `7. resnet_runs/checkpoints/*.keras` |
| 3 | `8. train_break_pattern_resnet_binary_from_bbox.py` | 7번 체크포인트와 동일 데이터로 이진 헤드 학습 → `7. resnet_runs_binary/` |

### 보조

- `9. test_break_pattern_resnet_bbox_confidence.py`: 7번 모델 관련 테스트용 (백업본 `* backup.py`는 로컬 전용, Git 제외)

구버전·실험용 코드는 로컬의 `_archive/`에 두었습니다 (Git에는 포함하지 않음).

## 환경

- **Python**: 3.12 권장 (`requirements.txt` 기준)
- **설치**: 저장소 루트에서

```bash
pip install -r requirements.txt
```

- **GPU**: TensorFlow 2.16+ CUDA 환경에 맞게 설치 ([TensorFlow 공식 가이드](https://www.tensorflow.org/install) 참고)

## 데이터 디렉터리 (Git 제외)

이 저장소에는 **대용량 `npy`·병합 CSV 등이 올라가지 않습니다** (`.gitignore` 참고). 로컬에서 다음을 준비합니다.

- `4. merge_data/` — 병합된 측정 데이터
- `9. edit_data/` — 편집·브레이크 JSON 등 (`6`번 스크립트의 `BREAK_JSON_ROOT` 등 경로와 일치해야 함)
- `5. train_data/` — `6` 실행 후 생성되는 학습·테스트용 배열 (`7`, `8`의 입력)

## 실행 예시

저장소 루트를 작업 디렉터리로 둔 뒤 순서대로 실행합니다.

```bash
python "6. training_data_bbox_extracted.py"
python "7. train_break_pattern_resnet_bbox_confidence.py"
python "8. train_break_pattern_resnet_binary_from_bbox.py"
```

- **Windows에서 7번**: 스크립트 상단 주석대로, 기본은 WSL2로 GPU 학습을 넘기고, 로컬에서 돌리려면 `--local`을 붙이거나 비 Windows 환경에서 실행합니다.
- **Linux / WSL**: 일반적으로 `--local` 없이도 로컬 학습 경로로 동작합니다.

## 산출물 (로컬)

- `5. train_data/train|test/*.npy` — 학습 입력
- `7. resnet_runs/checkpoints/best_x.keras`, `best_y.keras`, `best_z.keras` — bbox+confidence
- `7. resnet_runs_binary/` — 이진 모델 학습 로그·체크포인트

이 경로들은 `.gitignore`로 원격에 올리지 않습니다.
