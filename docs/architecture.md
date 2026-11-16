# 구조와 계약

캡처 스레드 4개 → 카메라별 최신 슬롯 1개 → 단일 GPU round-robin 스케줄러(batch1 기준선 측정 후 기본 최대4) → YOLO 원본 좌표 검출 → camera/session/class별 Ultralytics BYTETracker → PPE·구역·근접 규칙 → 제한 크기 저장 큐. 이미 도착한 프레임만 묶고 batch 대기는 추가하지 않는다. 미리보기 인코더는 별도 스레드에서 처리하고 API는 완성된 JPEG만 공유한다. 확대·브라우저 탭 추가는 모델·RTSP 연결을 추가하지 않는다.

`app/capture.py`의 PyAV는 RTSP TCP, 4초 연결/읽기 timeout, 지수 재연결을 사용한다. 재연결·파일 loop·seek는 새 session UUID와 빈 최신 슬롯을 만든다. 파일은 원본 PTS를 따라 재생한다. PyAV 소프트웨어 디코딩이 기준선이며 하드웨어 디코딩·H.265 최적화는 미검증이다.

프레임은 camera/session/frame, 원본 크기, source PTS, UTC 수신시각, monotonic 수신시각, 변환 메타데이터를 가진다. Ultralytics `.boxes.xyxy`는 이미 원본 픽셀이므로 재복원하지 않는다. 일반 homography 변환과 letterbox 역변환 도구는 단위 검증한다. 임의 crop·회전 전처리는 기본 경로에 적용하지 않는다.

추적은 내장 BYTETracker 하나만 사용한다. 프레임 드롭에 따른 TTL 오류를 피하기 위해 실제 관측시각으로 pool을 만료시킨다. camera/session/class가 다른 객체는 같은 pool에 들어가지 않는다. 관측하지 않은 Kalman 예측을 규칙 입력이나 현재 프레임 박스로 표시하지 않는다. 첫 관측으로 아직 활성화되지 않은 검출은 track_id 없이 표시한다.

PPE는 머리/상체 영역·크기·거리 기반 Hungarian 일대일 할당을 사용한다. 사람이 너무 작거나 화면 끝에 걸치거나 어둡거나 다른 사람과 크게 겹치면 unknown이다. 장시간의 보이는 영역에서 부재가 지속되어야 missing 후보가 된다. 이 가시성 판단은 bbox 기반 근사이고 실제 가림 판별을 보장하지 않는다. 현장 평가가 필요하다.

구역은 원본 영상 정규화 다각형, foot 또는 center anchor, 경계 margin, 진입/해제 확인 시간을 가진다. 진입·이탈 transition과 관측 체류시간을 event basis에 기록한다. 관측 공백 시간은 합산하지 않는다. 근접은 사람 높이로 정규화한 영상 거리/겹침을 기준으로 한다. driver exclusion은 항상 비활성이다. 바닥 4점 이상과 검증 표기가 있는 homography가 설정된 경우에만 지상 거리(m)를 사용한다.

이벤트 pending→active→resolved와 acknowledged는 별개다. key는 camera/session/person/risk/zone/config revision을 포함하며 사람별 독립 cooldown을 가진다. 미관측·연결 단절·재시작은 unknown으로 기록하고 안전 해제로 취급하지 않는다. 이전 세션의 미해제 이벤트는 DB에 남아 운영자가 판단한다.

SQLite WAL 저장 큐 128, 증거 인코딩 대기 16, 카메라별 JPEG pre-roll 90개/3초, 전체 JPEG 64MB, post clip 8건/각16MB를 기본 한도로 사용한다. 전체 원본 프레임을 누적하지 않는다. 클립은 관측 프리뷰의 실제 시점으로 H.264 MP4를 만든다. 저장 큐 포화·디스크 부족은 진단에 노출한다. 정리는 소유 마커가 있는 증거 루트 안의 DB 등록 파일만 대상으로 한다.

`app.main`은 파일 lock으로 중복 서버를 차단한다. Uvicorn workers=1을 유지한다. `stop.ps1`은 현재 프로젝트 service record의 PID/생성시각/instance token을 검사해 정상 종료를 요청한다.

오프라인 `replay`는 별도 경로로 모든 프레임을 처리한다. real-time FPS와 offline 처리량은 같은 지표가 아니다. browser 표시 지연과 카메라 물리 종단 지연은 측정하지 않은 경우 null이다.

기본 PyAV 경로의 received timestamp는 Python에 디코딩된 프레임이 준비된 시점이다. 실제 네트워크 패킷 도착시각과 FFmpeg 내부 지연은 포함하지 않는다. 따라서 대시보드 p50/p95/p99 이름은 `decode_ready_to_encoded_ms`이며 **host 패킷 수신→브라우저 지연으로 보고하지 않는다**. 카메라 센서/인코더/NVR/소켓/디코더/브라우저 지연의 전체 분리는 실제 장비 계측이 남아 있다.

측정 후 microbatch=2/4를 선택할 수 있다. 현재 준비된 카메라에서 최대 1프레임씩만 모으고 별도 대기 없이 즉시 추론한다. cursor를 이동해 특정 카메라가 기아 상태가 되지 않도록 한다. batch 결과 개수와 원본 크기를 검증하고 원래 frame 객체와 같은 순서로 연결한다. 기본 설정은 기준선인 batch=1을 유지한다.
## 최종 워밍업 보완

CUDA 시작 시 최대 microbatch까지 모든 batch 크기(기본1–4)와1:1/16:9/4:3 입력을 미리 실행한다. 한 카메라 중단으로 batch가 바뀌는 경우의 최초 연산 지연을 줄이기 위한 조치다. 그 밖의 입력 비율이나 다른 GPU에서는 첫 shape 비용을 별도로 확인한다. 모델 ready는 이 워밍업 이후 설정된다.
