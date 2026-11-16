# 신규 학습 provenance

## 사람 전용 재학습 및 후보 평가 (2026-11-16)

기존 train 현장 편중을 줄이기 위해 그룹별 상한 180장, seed43으로 403장을 선택했다. 원본 train/validation/test 간 이미지를 이동하지 않았다. 첫 통합 후보 `ppe-yolo26s-balanced-v3`는 12epoch에서 조기 종료됐고 test mAP50 .7242 / mAP50–95 .3718로 기존 통합 모델보다 낮아 채택하지 않았다.

이어 공식 초기 모델에서 새로 학습한 사람 전용 `person-yolo26s-v4`는 분류 출력만 학습했으나 박스 정확도가 낮았다. 채택한 `person-yolo26s-v5`는 backbone/neck/분류 특징은 동결하고 두 head의 사람 분류 출력과 box regression을 학습했다. 403장, 8epoch, 640, batch16, AdamW lr .0001, seed44, GPU0, AMP, mosaic0이다. 객체 클래스는 실제 person 하나이며 원본 사람 라벨을 유지했다. 파라미터 변경량은 provenance에 기록해 단순 클래스 이름 변경과 구별한다.

v5 SHA256: `315fe5bb703f3761523333aa6bef56b2899c92779f7f9a720c75ee6618580f2b`. 공식 초기 SHA256: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`. `models/person-yolo26s-v5.provenance.json`, `.evaluation.json`, `data/manifests/person-yolo26s-v5/`에 학습·평가·입력 hash를 보존한다. 크레인 영상은 어떤 학습에도 사용하지 않았다. 여러 후보를 확인한 개발 test는 blind 현장 평가가 아니다.

재현: `python -m scripts.train_balanced`, `python -m scripts.train_person --run-id person-yolo26s-v5 --train-box-heads`, `python -m scripts.evaluate_candidate person-yolo26s-v5`. 실행 이력이 존재하면 다른 run-id를 사용한다. 현재 등록 모델은 사람 v5와 보호구 v2이며, 공식 초기 모델 또는 탈락 후보를 운영에 등록하지 않는다.

## 최초 통합 모델 학습

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
