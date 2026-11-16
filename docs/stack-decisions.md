# 검증한 기술 구성

2026-11-16 개발 PC에서 Python 3.11.9, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Ultralytics 8.4.215를 새 `.venv`에 설치했다. GPU는 **데스크톱 RTX 5070 Ti 약16GB**, 드라이버 610.88이다. 사용자 테스트 노트북의 12GB GPU와 다른 장비다. `sm_120` 지원, CUDA tensor 행렬 곱과 YOLO 학습을 실제 실행했다. `nvidia-smi`의 CUDA UMD 13.3은 드라이버 지원 정보이며 이 프로젝트 wheel의 CUDA 런타임은 **12.8**이다.

최신 문서의 모델/추적 기본값이 바뀔 수 있어 설치된 라이브러리를 확인했다. 이 버전 BYTETracker 생성자는 `BYTETracker(args)`이며 손실 프레임 속성은 `max_frames_lost`다. tracker YAML의 0.25/0.1/0.25 confidence, 0.8 match, fuse_score=True를 명시적으로 사용한다. per-tracker ID 발급과 camera/session/class pool 분리를 함께 사용한다.

| 라이브러리 | 용도 | 라이선스(배포본 확인 기준) |
|---|---|---|
| Ultralytics | YOLO26·내장 ByteTrack | AGPL-3.0 / 별도 Enterprise 선택지 |
| PyTorch / torchvision | CUDA 추론·학습 | BSD 계열 |
| PyAV / FFmpeg | PTS를 보존하는 입력·H.264 클립 | PyAV BSD; 번들 FFmpeg 구성에 따른 LGPL/GPL 별도 확인 |
| OpenCV | 박스·다각형·JPEG | Apache-2.0 |
| FastAPI / Uvicorn / Pydantic | API·검증·WebSocket | MIT / BSD / MIT |
| SQLite | 이벤트 WAL 저장 | public domain |
| NumPy / SciPy / psutil / PyYAML | 수치·매칭·자원·설정 | BSD / BSD / BSD / MIT |

설치된 전체 버전은 `requirements.lock.txt`, CUDA wheel 버전과 index는 `setup.ps1`에 고정한다. 기본 서버는 외부 플랫폼 연동·클라우드 업로드·NPU를 사용하지 않는다. 데이터 다운로드는 하지 않았고 공식 초기 가중치와 오픈소스 패키지만 받았다.

참고한 공식 문서:

- [YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [내장 추적](https://docs.ultralytics.com/modes/track/)
- [PyTorch 설치](https://pytorch.org/get-started/locally/)와 [버전별 wheel](https://pytorch.org/get-started/previous-versions/)
- [Ultralytics export](https://docs.ultralytics.com/modes/export/)
- [Ultralytics 라이선스](https://www.ultralytics.com/license)

Context7·Playwright MCP·GitHub MCP 연결은 필요하지 않았다. 공식 문서, 로컬 파일·터미널을 사용했다. CUA 브라우저 호스트는 unavailable이었고 native Edge 접근은 승인 시간 초과였다. 실제 클릭 UI 검증은 완료하지 않았다.
