# OOF Loss×2 Group 5-Fold checkpoints

`fold_0.pth`부터 `fold_4.pth`까지는 각기 다른 외부 Fold를 보지 않고 학습한
ViT-B/16 + CLAdapter Stage 2 체크포인트입니다. 추론 시 다섯 모델의 이상 확률을
동일 가중치로 평균하고 `operating_rule.json`의 공통 임계값을 적용합니다.

각 체크포인트의 SHA256은 `operating_rule.json`에 기록되어 있으며 추론 코드가
실행 전에 자동 검증합니다.

각 `.pth` 파일은 약 373MB이므로 일반 GitHub에는 커밋할 수 없습니다. 동료에게
폴더를 직접 전달하거나 Git LFS/별도 스토리지를 사용해야 합니다.

