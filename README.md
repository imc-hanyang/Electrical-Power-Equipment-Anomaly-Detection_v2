# Electrical Power Equipment Anomaly Detection (5-Fold OOF 소프트보팅)

Electrical Power Equipment 데이터셋을 기반으로 crop 이미지의 이상(결함) 여부를 판정하는 추론 파이프라인.
**ViT-B/16 + CLAdapter** 단일 아키텍처를 **Group 5-Fold 교차검증**으로 학습하고,
5개 fold 모델의 이상확률을 **균등 평균(소프트보팅)** 한 뒤, OOF에서 고정한 **운영 임계값**을
적용해 정상/이상을 판정합니다.

> **최소 실행:** 체크포인트를 받고 `dataset/`에 이미지를 넣은 뒤 `bash wire_inference.sh` 한 줄.

---

## 1. 설치

```bash
conda create -n opgw python=3.10 -y
conda activate opgw
pip install -r requirements.txt
```

## 2. 체크포인트 다운로드

체크포인트는 용량 문제로 git에 포함되어 있지 않습니다.

**자동 다운로드 (gdown):**

```bash
python download_checkpoints.py                # 권장 모델(loss2) 자동 다운로드
python download_checkpoints.py --target all   # loss1 + loss2 모두
```

또는 아래 Google Drive 링크에서 수동으로 받아 `checkpoints/` 에 배치하세요.

| 항목 | Google Drive |
|---|---|
| 전선 loss 1 (oof_loss1_k5) | [다운로드](https://drive.google.com/drive/folders/1hUmM1vhGOQGj9Nc-jifHWo4ervCsqcjX?usp=drive_link) |
| 전선 loss 2 (oof_loss2_k5 · **권장**) | [다운로드](https://drive.google.com/drive/folders/1SbrD7ToTKjAkefPDiNef43i21Hrv98p9?usp=drive_link) |

배치 경로:

```
checkpoints/
└── oof_loss2_k5/          # 권장 모델 (loss×2)
    ├── fold_0.pth ~ fold_4.pth
    └── operating_rule.json   # 고정 임계값 · 체크포인트 무결성 해시
```

## 3. 데이터셋 구성

```
dataset/
├── Anomaly/   # 이상 이미지   ← Anomaly/ + Normal/ 가 있으면 성능(P/R/F1/AUROC) 자동 계산
└── Normal/    # 정상 이미지
```

모델이 정상 동작하려면 **이미지가 결함 영역 기준으로 정밀하게 crop**되어야 합니다.
전체 사진을 그대로 입력하면 성능이 크게 저하됩니다.

| 정상 (Normal) | 이상 (Anomaly) |
|:---:|:---:|
| <img src="normal.png" width="200"/> | <img src="anomaly.png" width="200"/> |

---

## 4. 추론 (한 줄 실행)

```bash
bash wire_inference.sh
```

- `./dataset` 추론 → `./predictions` 저장
- `dataset/Anomaly` + `dataset/Normal` 이 있으면 **Precision / Recall / F1 / AUROC 자동 계산**

### 옵션 (선택 — 전부 기본값 있음)

```bash
bash wire_inference.sh --input-dir /path/to/images --output-dir /path/to/out
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--input-dir` | 테스트 이미지 폴더 | `./dataset` |
| `--output-dir` | 결과 저장 폴더 | `./predictions` |
| `--loss` | 체크포인트 세트 (1 / 2) | `2` (권장) |
| `--threshold` | 판정 임계값 | `operating_rule.json` 자동 |
| `--device` | `cuda:0` / `cpu` | 자동 감지 |

## 5. 출력

| 파일 | 내용 |
|---|---|
| `predictions/predictions.csv` | 이미지별 fold별 점수 · 소프트보팅 점수 · 판정(Anomaly/Normal) |
| `predictions/summary.json` | 요약 + (GT 있으면) TP·FN·FP·TN · Precision/Recall/F1/AUROC |

GT(Anomaly/Normal 폴더)가 있으면 콘솔에 성능표가 함께 출력됩니다.

## 프로젝트 구조

```
.
├── run_wire.py            # 추론 본체 (5-Fold 소프트보팅 + 성능 자동계산)
├── wire_inference.sh      # zero-arg 진입점 (→ run_wire.py)
├── download_checkpoints.py
├── requirements.txt
├── inference/
│   ├── model_runtime.py   # 모델 로드 · 전처리
│   ├── infer_soft_voting.py
│   └── inference.py
├── src/                   # ViT-B + CLAdapter 모델 정의
├── oof/                   # OOF 폴드 구성 · 임계값 산정 (재현용)
└── checkpoints/
    └── oof_loss2_k5/      # 5-fold 체크포인트 + operating_rule.json
```

## 성능

성능은 테스트셋에서 `bash wire_inference.sh` 실행 시 자동 산출됩니다
(`dataset/Anomaly` + `dataset/Normal` 필요).

OOF 운영점 (`operating_rule.json`, loss×2 기준): `threshold ≈ 0.0224`, OOF AUROC `0.932`, OOF AP `0.588`.

> ⚠️ Precision/Recall은 테스트셋의 **이상 비율(prevalence)** 에 따라 달라집니다.
> 균형(1:1) 셋과 실제 분포(이상 희소)의 Precision은 크게 다르므로, **실 분포 기준**으로 해석하세요.
