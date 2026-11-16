# Industrial Safety YOLO

Windows용 독립 산업안전 영상 분석 프로그램입니다. 4채널 캡처, YOLO 검출, Ultralytics ByteTrack, PPE 상태, 다각형 위험구역, 사람–지게차 근접 규칙, SQLite 이벤트·증거, 한국어 HTML 대시보드를 새로 구현했습니다.

**현재 상태: 개발 검증용. 현장 인수 통과 아님.** 새로 학습한 모델은 `person`, `helmet`, `safety_vest` 3종만 지원합니다. 지게차 학습 데이터가 불완전하여 지게차 검출 모델은 없습니다. 해당 추적·근접 규칙 구현을 실물 검출 완료로 간주하지 마세요. 실제 C340I·NVR 계정, 원본 동영상, 이벤트 정답, 테스트 노트북 검증도 남아 있습니다.

## 실행

PowerShell에서 프로젝트 루트로 이동합니다.

```powershell
.\scripts\setup.ps1
Copy-Item .env.example .env
# .env에 CAMERA_1_RTSP ~ CAMERA_4_RTSP를 입력합니다.
# configs/cameras.local.yaml에서 해당 enabled 값을 true로 바꿉니다.
.\scripts\start.ps1
```

대시보드: <http://127.0.0.1:8765>. 중지: `.\scripts\stop.ps1`. 진단: `.\scripts\diagnose.ps1`.
이미 생성된 `.venv`가 있으면 설치를 반복할 필요 없이 start를 실행합니다. `frontend/dashboard.html` 단독 실행은 서버 연결 안내만 제공합니다.

현재 전달된 폴더에는 보유 test 사진으로 만든 전송 시험 영상도 있습니다. `.\scripts\start.ps1 -Demo`로 실제 신규 모델의 4분할 분석을 볼 수 있습니다. 화면에 DEMO로 표시되며 실물 카메라 영상이나 사건 정확도 시험이 아닙니다. 모드 변경 전에는 `stop.ps1`을 실행하세요.

로컬 영상은 `configs/cameras.local-video.example.yaml`을 `configs/cameras.local.yaml`로 복사해 파일 경로를 지정합니다. RTSP 자격증명은 환경변수로만 읽으며 API에는 반환하지 않습니다. `stream1`은 카메라의 메인 스트림입니다.

## 재현

```powershell
.venv\Scripts\python.exe -m scripts.audit_data
.venv\Scripts\python.exe -m scripts.prepare_data
.venv\Scripts\python.exe -m scripts.download_weights --model yolo26s.pt
.venv\Scripts\python.exe -m scripts.train --smoke --epochs 1 --batch 8 --run-id smoke-new
.venv\Scripts\python.exe -m scripts.train --epochs 30 --batch 16 --run-id ppe-new
.venv\Scripts\python.exe -m scripts.evaluate ppe-new --register
.venv\Scripts\python.exe -m scripts.evaluate_slices ppe-new
.venv\Scripts\python.exe -m pytest -q
```

데이터 위치가 바뀌면 `audit_data --sources <PPE폴더> <물류폴더> <현장사진폴더>`를 사용합니다. 원본은 수정하지 않습니다. 원본·파생 데이터, 가중치, `.venv`, DB·증거·로그는 Git에 넣지 않습니다. 모델 registry는 새 학습 run, 초기 가중치 hash, 데이터 manifest hash와 평가 결과를 보존합니다. Git checkout만으로 가중치 파일이 생기지는 않으며 재학습 또는 이 프로젝트의 검증된 가중치 별도 전달이 필요합니다.

## 주요 도구

| 목적 | 명령 |
|---|---|
| 1/2/4채널 성능 | `python -m scripts.benchmark --channels 4 --microbatch 4 --seconds 1800 --cameras configs/cameras.local.yaml` (batch1 기준선은 `--microbatch 1`) |
| 2시간 시험 | 위 명령의 `--seconds 7200` |
| 모든 프레임 오프라인 처리 | `python -m scripts.replay video1.mp4 video2.mp4 --output outputs/replay-001` |
| 사건 단위 평가 | `python -m scripts.evaluate_events truth.json predicted.json --camera-hours 8 --output outputs/event-metrics.json` |
| 선택 export | `python -m scripts.export_model ppe-new --format onnx` 또는 대상 노트북에서 `--format engine` |
| 전송 시험 자료 생성 | `python -m scripts.make_demo_videos` (DEMO 정지 사진 시퀀스; 실제 동영상 아님) |

항상 `.venv\Scripts\python.exe`를 사용하세요. 선택 export에는 별도 ONNX/TensorRT 의존성이 필요하고, 동등성 검사에 통과해도 운영 provider로 자동 전환하지 않습니다.

상세: [납품 및 인수 조건](docs/delivery-report.md), [검증 결과](docs/verification.md), [4채널 실측](docs/benchmark-4ch.md), [학습 및 평가](docs/model-evaluation.md), [데이터 감사](docs/dataset-audit.md), [운영](docs/operation.md), [노트북 설치](docs/laptop-install.md), [카메라 네트워크](docs/c340i-network.md).
