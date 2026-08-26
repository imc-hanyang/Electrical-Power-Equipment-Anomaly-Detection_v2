# Loss×1/2 Group 5-Fold OOF + Soft Voting 전달본

이 폴더는 공개 GitHub 저장소를 2026-08-20에 새로 clone한 뒤, 현재 보고 중인
Loss×1/2 Group 5-Fold OOF/Soft Voting을 실행하는 데 필요한 코드와 모델만 추가한
동료 전달용 패키지입니다. 기존의 다른 실험 코드와 산출물은 포함하지 않았습니다.

## 포함 내용

- 공개 GitHub 원본 코드: `imc-hanyang/Electrical-Power-Equipment-Anomaly-Detection`
- 실제 학습에 사용한 모델 코드 변경분: `src/`
- Group 5-Fold 분할·OOF 예측·공통 임계값 산정: `oof/`
- Loss×1 Stage 2 모델 5개: `checkpoints/oof_loss1_k5/fold_0.pth` ~ `fold_4.pth`
- Loss×2 Stage 2 모델 5개: `checkpoints/oof_loss2_k5/fold_0.pth` ~ `fold_4.pth`
- 조건별 OOF 공통 임계값: 각 체크포인트 폴더의 `operating_rule.json`
- 신규 이미지 폴더 추론: `inference/`
- 당시 사용한 분할 manifest: `oof/reference_manifests/`

학습 이미지 자체와 230장 벤치마크 이미지는 포함하지 않습니다.

## 1. 환경 설치

```bash
cd /path/to/8-20-oof-share

conda create -n kepco-oof python=3.10 -y
conda activate kepco-oof
pip install -r requirements.txt
```

CUDA 12.1용 PyTorch가 설치되도록 `requirements.txt`에 PyTorch index가 지정되어
있습니다. 다른 CUDA 버전을 사용하는 서버에서는 PyTorch만 해당 환경에 맞춰
설치하십시오.

## 2. 신규 이미지 추론

입력 폴더 아래의 이미지 파일을 재귀적으로 찾습니다. 정답 라벨이나 클래스별
하위 폴더는 필요하지 않습니다.

```bash
cd /path/to/8-20-oof-share

bash inference/run_inference.sh \
  /path/to/input_images \
  /path/to/inference_results \
  cuda:0
```

위의 기존 3개 인자 명령은 기본값인 `Loss×2`를 실행합니다. Loss 조건을 명시하려면
마지막 인자에 `1` 또는 `2`를 지정합니다.

```bash
# Loss×1
bash inference/run_inference.sh /path/to/input_images /path/to/result_loss1 cuda:0 1

# Loss×2
bash inference/run_inference.sh /path/to/input_images /path/to/result_loss2 cuda:0 2
```

동일한 명령을 Python으로 실행하려면:

```bash
python -m inference.infer_soft_voting \
  --input-dir /path/to/input_images \
  --output-dir /path/to/inference_results \
  --loss 2 \
  --device cuda:0 \
  --copy-images
```

출력:

```text
inference_results/
├── predictions.csv   # 모델별 점수, 평균점수, 임계값, 최종 판정
├── summary.json
├── Normal/           # --copy-images 사용 시 정상 판정 이미지
└── Anomaly/          # --copy-images 사용 시 이상 판정 이미지
```

추론 규칙은 다음과 같습니다. 임계값은 Loss 조건별로 다릅니다.

```text
5개 Fold 모델의 이상 확률
          ↓
다섯 확률의 동일 가중 평균(Soft Voting)
          ↓
Loss×1: 평균점수 ≥ 0.013258215  → 이상
Loss×2: 평균점수 ≥ 0.0223751217 → 이상
```

## 3. OOF 개념과 공통 임계값

각 학습 이미지는 자신을 학습하지 않은 Fold 모델로 정확히 한 번 예측합니다.

```text
Model 0: F1+F2+F3+F4 학습 → F0 예측
Model 1: F0+F2+F3+F4 학습 → F1 예측
Model 2: F0+F1+F3+F4 학습 → F2 예측
Model 3: F0+F1+F2+F4 학습 → F3 예측
Model 4: F0+F1+F2+F3 학습 → F4 예측
```

다섯 Fold의 OOF 점수는 평균하지 않고 세로로 이어 붙여 Train 29,724장의 고유한
미관측 예측을 만듭니다. 이 통합 OOF에서 Precision의 95% Wilson 하한이 90%
이상인 임계값 중 Recall이 가장 높은 값을 고정했습니다. Loss×1은
`0.013258215`, Loss×2는 `0.0223751217`입니다.

## 4. OOF 예측과 임계값 재산정

원본 학습 이미지가 동일한 위치에 있을 때 다음 순서로 재현할 수 있습니다.

```bash
python -m oof.predict_oof \
  --manifest-dir oof/reference_manifests \
  --data-root /path/to/data_train \
  --checkpoint-dir checkpoints/oof_loss1_k5 \
  --output-dir outputs/oof_predictions_loss1 \
  --device cuda:0

python -m oof.select_common_threshold \
  --predictions-dir outputs/oof_predictions_loss1 \
  --checkpoint-dir checkpoints/oof_loss1_k5 \
  --output outputs/recomputed_operating_rule_loss1.json
```

Loss×2는 위 명령의 `loss1`을 `loss2`로 바꾸면 됩니다.

재산정 결과의 threshold와 체크포인트 SHA256이 제공된
`operating_rule.json`과 일치하는지 확인하십시오.

## 5. Group 5-Fold manifest를 새로 만드는 경우

입력 CSV 필수 열은 `image_path`, `label`, `group`입니다. `group`은 동일 원본에서
생성된 Crop들이 서로 다른 Fold로 나뉘지 않도록 하는 원본 식별자입니다.

```bash
python -m oof.build_group_folds \
  --source-csv /path/to/source_train.csv \
  --output-dir outputs/new_manifests \
  --folds 5 \
  --outer-seed 1721 \
  --inner-seed-base 9100
```

## 6. Loss×1/2 Group 5-Fold 재학습

```bash
export DATA_ROOT=/path/to/data_train
export MANIFEST_DIR=/path/to/manifests
export OUTPUT_DIR=/path/to/train_outputs
export GPU=0
export PYTHON=python
export LOSS_MULTIPLIER=1  # 1 또는 2

bash oof/train_group5fold.sh
```

학습 조건:

- 모델: ViT-B/16 + CLAdapter, mean pooling, 입력 224×224
- 정상 Loss 가중치: 1.0
- 이상 Loss 가중치: `LOSS_MULTIPLIER`로 지정한 1.0 또는 2.0
- 외부 분할: 원본 `group` 기준 5-Fold
- 체크포인트 선택: 각 Fold 내부 Validation의 이상 클래스 AP
- Stage 1: backbone 고정, 40 epoch
- Stage 2: 전체 fine-tuning, 40 epoch

## 7. 제공 체크포인트 검증

```bash
sha256sum checkpoints/oof_loss1_k5/fold_*.pth
sha256sum checkpoints/oof_loss2_k5/fold_*.pth
```

정답 해시는 각 체크포인트 폴더의 `operating_rule.json`에 있습니다. 추론 프로그램도
실행 전에 같은 검사를 자동으로 수행합니다.

## 주의사항

- Loss×1의 230장 진단 결과는 `TP=98, FN=17, FP=17, TN=98`,
  Precision/Recall/F1 모두 85.22%입니다.
- Loss×2의 230장 진단 결과는 `TP=99, FN=16, FP=11, TN=104`,
  Precision=90.00%, Recall=86.09%, F1=88.00%입니다.
- 위 230장은 과거 Validation으로 사용된 진단 벤치마크이며 완전한 blind Test가
  아닙니다.
- 모델 파일은 각각 약 373MB로 GitHub의 일반 파일 제한을 넘습니다. 코드만 Git에
  올리고 모델은 Git LFS 또는 별도 공유 스토리지를 사용하는 것을 권장합니다.
