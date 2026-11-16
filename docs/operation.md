# 운영

사람의 전체 박스는 `PERSON`으로 유지한다. 현재 프레임에서 관측한 얼굴·어깨·골반 자세 좌표로 머리·몸통 영역을 계산하고, 미착용이 확인되면 각각 빨간 `NO - HARD HAT`, `NO-VEST` 박스를 표시한다. 두 부위가 모두 관측되고 미착용이 확인되면 PERSON을 포함해 세 개의 박스가 표시된다. 이 윤곽은 자세 좌표 기반 추정이며 직접 학습한 부위 bbox나 픽셀 segmentation이 아니다. 전신 박스를 고정 비율로 나누던 방식은 제거했다. 필요한 관절 신뢰도가 .5 미만이거나 사람 간 연결이 모호하면 해당 PPE 상태는 `unknown`이며 빨간 위치 박스를 만들지 않는다. 부위 위치와 보호구 착용 판단은 별도 단계다.

부위 모델 입력은 현재 프레임의 사람 검출 주변 crop이다. `max_body_people_per_frame` 기본8명, `body_min_person_height` 기본90픽셀로 작업량을 제한한다. 이 한도를 넘는 사람 또는 너무 작은 사람은 부위 위치가 없을 수 있다. 상태 API의 `body_part_classes`, `body_part_input`, `max_body_people_per_frame`에서 구성을 확인한다. crop 좌표는 원본 표시 픽셀로 한 번만 복원하며, 이전 프레임의 부위 박스로 빈 관측을 채우지 않는다.

기본은 127.0.0.1:8765, 단일 서버다. `start.ps1`은 서버와 브라우저를 열고 `-NoBrowser`는 서버만 시작한다. `stop.ps1`은 이 프로젝트 instance token으로 정상 종료를 요청한다. 종료 실패 시 로그를 확인하며 모든 python 프로세스를 일괄 종료하지 않는다.

기본 runtime은 PyTorch CUDA FP16, 모델 입력640, 최대 microbatch4다. 이미 준비된 최신 프레임만 모으므로 느린 카메라가 batch를 채울 때까지 기다리지 않는다. batch1 기준선과 비교한 개발 PC 결과에 따른 선택이며 노트북에서 다시 측정해야 한다. 미리보기 목표12FPS와 실제 FPS는 구분해 표시한다.

검출 개선 구성은 사람 전용 v5, 보호구 v2, 자세 보조 모델 `body-pose-yolo26n-v2`를 함께 사용한다. `models/model_registry.json`의 `active_classes`가 각 모델의 출력 역할을 제한하고 `inference`가 head/신뢰도를 고정한다. 보호구·자세 모델의 사람 박스는 별도로 표시하거나 추적하지 않는다. 추론 CPU 스레드는 1개이며 GPU 작업은 기존 CUDA 배치를 사용한다. .10~.25 사람 관측은 기존 ID와 연결된 경우에만 표시하므로 낮은 신뢰도의 새 대상이 경보를 만들지 않는다. 설정 revision에는 모델 registry hash도 들어간다. 새 설정 적용에는 프로젝트 서버를 재시작한다.

`configs/cameras.local.yaml`에 4개 이내 고유 ID를 지정한다. RTSP는 `url_env`만 기록하고 `.env`에 자격증명을 둔다. 파일 입력은 `source_type: file`, `path`, `loop`를 사용한다. 자동 재연결, loop, seek는 새 session이며 이전 추적 상태를 이어받지 않는다. DEMO 표시는 생성한 정지 이미지 시퀀스용이며 실제 현장 영상이 아니다.

휴대전화처럼 회전 메타데이터가 있는 영상은 표시 방향으로 보정해 분석한다. ROI는 보정된 표시 영상의 정규화 좌표다. 이전 버전에서 옆으로 누운 영상에 ROI를 저장했다면 보정된 화면에서 다시 설정해야 한다. 원본 영상은 변경하지 않는다. 한 개 파일의 독립 전수 시험은 `python -m scripts.replay <영상> --output <새 결과 폴더> --no-zones`로 실행해 다른 카메라의 기존 구역을 적용하지 않을 수 있다.

대시보드에서 ROI 편집 → 영상 위 꼭짓점 클릭 → 다각형 추가 → 구역 저장 순서로 여러 구역을 설정한다. 클릭 좌표는 영상 비율을 유지한 canvas의 CSS 크기로 나누고 canvas backing은 devicePixelRatio로 설정한다. 저장은 정규화 좌표이므로 확대/축소와 별개다. 고DPI 실제 브라우저 동작은 아직 미검증이다. 잘못된 자기교차·범위 밖·중복 ID 다각형은 API가 거부한다.

경보음은 사용자 클릭으로만 켠다. 이벤트의 ‘확인 처리’는 acknowledged이며 위험의 resolved와 다르다. 관측 불가 이벤트는 연결이 끊겨 안전해졌다는 뜻이 아니다. 영상 없이 마지막 박스를 LIVE로 남기지 않고 1초 TTL 후 현재 관측 없음 화면을 보낸다.

이벤트 SQLite와 증거는 `outputs/runtime`에 있다. evidence는 JPEG 및 H.264 MP4, 기본 pre/post 3초이며 저장이 늦거나 입력이 끊기면 확보한 구간만 남긴다. DB에는 사용한 모델 hash·설정 revision·camera/session/track·판단 근거·시간이 남는다. 오류/큐 드롭/디스크 부족은 `/api/system`과 대시보드 경고에서 확인한다. 증거는 기본 14일/10GB, 최소 여유 2GB 정책을 적용하며 소유 폴더 밖 파일을 정리하지 않는다.

LAN bind는 `SAFETY_HOST`와 길이 24자 이상의 `SAFETY_API_TOKEN`을 요구한다. 로그인은 HttpOnly SameSite cookie를 발급하고 API·MJPEG·WebSocket을 보호한다. 기본 HTTP는 신뢰할 수 있는 로컬 LAN용이며 원격망은 HTTPS reverse proxy 또는 VPN이 필요하다. Origin이 다른 변경 요청은 거부한다. 인터넷 노출은 수행하지 않았다.

`/health/live`는 서버 생존, `/health/ready`는 4채널/모델 준비를 뜻한다. 개발 모델·미지원 지게차·미연결 카메라 상태에서는 ready=503이 정상이다. ready 응답이 현장 인수 인증을 뜻하지 않으며 별도 acceptance 기록이 필요하다.

주요 API: `/api/system`, `/api/models`, `/api/metrics`, `/api/cameras`, `/api/cameras/{id}/start|stop|seek`, `/api/cameras/{id}/stream.mjpg`, `/api/cameras/{id}/zones`, `/api/events`, `/api/events/{id}/ack`, `/api/evidence/{id}`, `/ws/events`. Swagger는 `/docs`다.

현재 제한: 사람/PPE bbox 기반 가시성은 부분 가림을 완전히 알 수 없다. 프레임 이동·역광·야간·작은 PPE·손에 든 안전모·걸어 둔 조끼·일반 모자·유사색 작업복에 대한 현장 정답 평가를 수행해야 한다. 지게차 앞 보행자나 탑승자를 임의로 제외하지 않는다. 자동 비상정지/PLC 제어에는 연결하지 않는다.
