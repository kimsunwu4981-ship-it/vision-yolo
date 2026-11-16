# 테스트 노트북 설치

사용자 표기: **라이젠 9 AI 392 / RTX 5070 Ti Laptop VRAM 12GB / RAM 32GB**. 정확한 CPU 이름·OS·전력 제한·드라이버는 장비에서 확인해야 하며 임의로 다른 CPU 모델로 바꾸지 않았다.

1. Python 3.11 64-bit와 NVIDIA 드라이버를 장비에 준비한다. `.venv`를 다른 PC에서 복사하지 않는다.
2. 프로젝트를 쓰기 가능한 로컬 폴더에 checkout한다. 대량 데이터·학습을 OneDrive 동기화 대상 밖에서 수행하면 I/O 변동을 줄일 수 있다.
3. PowerShell에서 `scripts/setup.ps1`을 실행한다. CUDA wheel과 의존성 lock을 설치하고 진단 JSON을 만든다.
4. 이 프로젝트에서 생성한 가중치를 `models/`에 전달하고 registry SHA256과 일치시킨다. 가중치를 전달하지 않으면 원본 데이터 경로를 설정해 새 학습 절차를 재현한다.
5. `configs/cameras.local.yaml` 및 `.env`를 설정하고 `scripts/start.ps1`을 실행한다.
6. `scripts/diagnose.ps1` 결과의 torch CUDA runtime, 실제 CUDA tensor device, 지원 arch, GPU 이름·메모리를 확인한다.
7. 전원 연결·전력 profile·온도·드라이버·codec·입력 해상도·imgsz를 기록한다. 1/2/4채널, 30분, 2시간 시험을 같은 조건으로 수행한다.

CPU fallback은 기능 점검용이며 성능 저하로 표시한다. GPU 사용 가능 여부만으로 통과하지 않는다. 정상 목표는 채널당 새 추론 15FPS, 미리보기 10–15FPS, host→browser p95 500ms 이하, VRAM 9GB 이하, 시스템 RAM 20GB 이하이다. 지게차 모델이 추가되면 전체 모델 비용을 다시 측정해야 한다.

초기 backend는 PyTorch CUDA이다. TensorRT 엔진을 다른 GPU에서 복사하지 말고 대상 노트북에서 export 및 동등성 검사를 수행한다. optional export 의존성 설치와 운영 backend 전환은 기본 설치에 포함하지 않는다.
