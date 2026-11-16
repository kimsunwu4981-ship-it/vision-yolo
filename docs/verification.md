# 검증 상태

이 문서는 소프트웨어 개발 검증과 실제 안전 시스템 인수를 구분한다. **현장 인수 미완료**다.

## 실행하여 통과한 항목

- Python 컴파일, 내장 JavaScript `node --check`, `pip check`.
- 단위/계약/실제 codec 테스트 31개: 좌표·letterbox/crop/회전 복원, bbox/다각형 검증, 최신 프레임 슬롯 한도, camera/session/class별 ByteTrack 분리, 시간 TTL, 미관측 상태, PPE 일대일 할당·손에 든 안전모·어두운 장면 unknown, 구역 진입/이탈/재진입/관측 체류, 운전자 임의 제외 방지, 사건별 독립 타이머, 증거 JPEG/H.264와 가변 PTS, SQLite 잠금 복구, 디스크 부족, 큐 포화, token/Origin 인증, 중복 API worker의 GPU·카메라 소유 차단.
- 생성한 H.264 실제 파일 디코딩, 파일 seek/loop 시 세션 갱신, 한 입력 파일 오류의 격리.
- 실제 새 학습 모델 + 4개 생성 영상 + HTTP 서버: 4채널 MJPEG와 frame key, 실제 CUDA 모델, 미지원 지게차/미승인 상태 ready503, 한 채널 stop과 다른3채널 처리 지속, 재시작 세션 갱신, ROI 저장/조회, 실제 모델 관측에 따른 zone 이벤트, acknowledge와 위험 해제 분리, 증거 HTTP 다운로드·MP4 실제 디코딩, WebSocket 재연결. `outputs/http-integration/result.json`.
- 4,789개 사용 이미지와 파생 라벨 hash 재검증. train/validation/test group·exact hash 교차 중복 없음. 완전한 source site/time 독립성을 뜻하지 않음.
- 공식 초기 가중치 smoke GPU 학습, 30 epoch 본 학습, validation/test bbox 평가, 작은 객체 평가·실패 사례 저장, validation-only threshold 탐색.
- 깨끗한 소스 archive + 새 `.venv`: 고정 의존성 설치, 데이터 3,174/930/685 재구성, tests, 별도 CPU smoke 학습·validation. CPU smoke checkpoint는 운영 registry로 승격하지 않음.
- 4채널 생성 H.264 DEMO의 30분 연속 처리 완료. 저장 오류/드롭0, 파이프라인 오류 없음. 평균 추론24.17–24.21FPS, 미리보기10.29–10.49FPS. 최소 구간·초기 지연·공유 시스템 RAM은 정상 목표 미달이므로 인수 통과가 아님. `docs/benchmark-4ch.md` 참조.
- 최종 replay: DEMO4개 각200프레임, 총800프레임 전수 처리와0드롭을 확인했다. `outputs/replay-demo-final/`에 inputs_manifest.json/run_manifest.json/detections.jsonl/tracks.jsonl/events.jsonl/performance.csv가 있다. 이벤트 시간 규칙은 source PTS 기반 가상 관측 시계를 사용하고 실제 decode 시각도 별도 기록한다. 사건 정확도 평가가 아니다.
- 최종 PowerShell start/stop: `start.ps1 -NoBrowser -Demo` 실제 백그라운드 실행, 소유 instance 일치, HTML200, CUDA4채널, 중복 start의 기존 프로세스 재사용, `stop.ps1` 정상 종료를 확인했다. 마지막 snapshot 추론24.63–24.83FPS/미리보기10.53–10.60FPS, 파이프라인·저장 오류 없음. `outputs/start-script-state.json`은 짧은 기능 확인이며 추가30분 시험이 아니다. 검증 서버는 종료했다.

## 발견하여 수정한 오류

- 설치된 Ultralytics ByteTrack 생성자/손실 프레임 속성 차이: 설치본 API에 맞추고 실제 tracker 계약 테스트 추가.
- 초기 가변 프레임 간격 MP4 증거의 encoder timestamp 충돌: stream/codec time base를 1/1000, B-frame0으로 고정하고 실제 디코딩 회귀 검증.
- 최신 preview slot에 덮여 이벤트 전이가 누락될 수 있던 구조: 이벤트 DB 저장을 추론 경로의 별도 bounded 큐로 분리, 이벤트 증거도 별도 bounded 인코딩 작업으로 보존.
- 원본 파일명만으로 나뉜 같은 옥상 현장: Video/KakaoTalk를 같은 group으로 합친 후 공식 초기값부터 본 학습.
- 전역 Ultralytics 설정 쓰기 시도: 프로젝트 `.cache/ultralytics`로 강제. 기존 전역 설정 쓰기는 권한 차단되었고 복사하지 않음.
- PowerShell의 pip 버전 알림이 NativeCommandError로 기록되어 설치 wrapper가 exit1을 반환한 현상: 설치 자체와 pip check는 성공. setup에 version-check 비활성 및 명시적 성공 exit를 적용.
- 기본 microbatch4로 변경한 직후의 최종 HTTP 검사에서 한 카메라 stop 후1.5초 내 나머지 채널 진행 assertion 실패. batch1–4와1:1/16:9/4:3 입력을 시작 시 워밍업하도록 보완한 뒤 같은 기준을 통과했다. 나머지 세 채널13→48/49/50프레임, 관측 age32–47ms였다. 실패 로그는 `outputs/http-integration/failed-stop-isolation.*`, 수정 후9개 검사 결과는 `result.json`에 보존했다. 30분 run은 이 워밍업 보완 이전 소스이며 최종 코드의30분 재시험으로 소급하지 않는다.
- Windows PowerShell5의 기본 ANSI JSON 읽기가 한글 프로젝트 경로 비교를 실패시킴: 시작 스크립트의 로컬 JSON/환경파일 읽기를 UTF-8로 명시하고 동일 프로젝트 service 확인을 실제 재검증했다.

## 미실시 또는 인수 근거로 사용할 수 없는 항목

| 항목 | 상태 / 이유 |
|---|---|
| C340I 4대·NVR 실제 RTSP, 인증 실패/네트워크 재연결 | 장치 주소·계정 없음. 파일 오류/loop 검사는 RTSP 시험을 대체하지 않음 |
| 원본 현장 연속 영상 4개 | 사진만 제공됨. 생성한 DEMO는 전송·기능·부하 fixture |
| 신규 지게차 모델 | 원천 ZIP0바이트 + 임시 다운로드뿐. 지게차 규칙 단위 테스트는 검출 정확도가 아님 |
| 노트북 성능 | 해당 장비 없음. desktop RTX5070Ti16GB 결과와 구분 |
| 브라우저 실제 클릭, 확대 시 3채널 유지, 고DPI ROI, 재연결 시각검증 | CUA browser host unavailable, native Edge 접근 승인 시간 초과. API/좌표 코드 검증만 완료 |
| 실제 느린 브라우저 장시간 정체 | bounded 전송 구조 구현, 실제 browser 부하 미실시 |
| host 네트워크 수신→브라우저 표시 p95 | 미측정. decode ready→JPEG만 측정 |
| 사건 precision/recall·오경보/카메라시간·ID switch/fragmentation·야간·가림 | 정답 영상 없음. 평가 도구 제공, 수치 미생성 |
| 2시간 실물4채널 연속 운용 | 미실시 |
| GPU OOM을 실제로 발생시키는 장애 시험 | CPU 경로 실제 실행, fallback 코드 구현. 물리 GPU 장애 주입은 미실시 |
| ONNX/TensorRT export 및 동등성 | 도구 제공, 실행 미실시. 현재 PyTorch CUDA만 검증 |

외부 browser 제어 실패를 숨기거나 mock/API 테스트를 UI 클릭 통과로 바꾸지 않았다. 모델이 없는 class는 missing_classes에 남고 health readiness를 제한한다. 정확도/성능 초기 목표는 조정하지 않았다.
