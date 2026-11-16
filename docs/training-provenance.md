# 신규 학습 provenance

공식 초기 가중치: [Ultralytics assets YOLO26s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt).

SHA256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.

1. `smoke-yolo26s-v1`: 24 train / 8 validation 이미지, 1 epoch, batch8, imgsz640. 신규 환경·GPU·NaN·경로 점검 완료. 이 checkpoint는 운영 registry에 등록하지 않았다. 이후 시각 검토에서 same-site family를 추가 병합했으므로 smoke manifest와 본 학습 manifest는 다르다.
2. `ppe-yolo26s-v2`: 수정된 group split, 30 epochs, seed42, imgsz640, batch16, AdamW, lr0 .001, patience8, workers0, CUDA AMP, close_mosaic3. 초기 가중치부터 새 학습했으며 smoke 또는 기존 프로젝트 checkpoint를 이어받지 않았다.

본 학습 시작 2026-11-16 18:14:22 KST, 종료 18:42:01 KST. desktop RTX5070Ti에서 수행했다. 전체 결과는 `outputs/training/ppe-yolo26s-v2`에 있다. Ultralytics validation fitness 기준 best.pt를 선택했고 test로 checkpoint를 선택하지 않았다.

선택한 가중치 SHA256:

`03c3efe5908fd7e25c21bbc0b92200787bdff7c961995c2a03d36c29afcf81dd`

registry 사본: `models/ppe-yolo26s-v2.pt` (Git 제외).

학습 manifest SHA256:

| manifest | SHA256 |
|---|---|
| train | `c5eb2b777338dce36fcb403f627cf8ddadc99b080eb5c11881cf6b47e1e13f39` |
| validation | `61febcca19901bb92a59656d2efddf9b0fc67f5f7bc6ad271bd9a3f4e59a7ec7` |
| test | `d283172ec73be227debd3cc8cd992bcf3650f045054bd04bebb73c3e5707f31c` |

실행 설정·패키지 버전·가중치/manifest hash는 `models/ppe-yolo26s-v2.provenance.json`과 model registry에 Git으로 기록했다. 학습·평가 결과는 데이터 구성 및 하드웨어에 의존한다. YOLO26n 후보와 export는 실행 도구를 제공하지만 신규 n 모델 학습·비교나 ONNX/TensorRT 측정은 수행하지 않았다.
