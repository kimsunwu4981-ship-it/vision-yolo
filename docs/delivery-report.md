# 납품 및 남은 인수 조건

2026-11-16, 독립 프로젝트 `industrial-safety-yolo`. 신규 코드·전용 가상환경·독립 Git 저장소·신규 PPE 학습 모델을 제공한다. **개발용 구현과 검증 결과이며, 필수 지게차 모델과 실제 현장 인수는 미완료다.**

## 제공한 기능

4개 독립 입력, 최신 프레임 슬롯, 공정한 CUDA 추론 스케줄러, 원본 좌표 검출, Ultralytics 내장 ByteTrack의 camera/session/class 분리, PPE equipped/missing/unknown, 다중 다각형 ROI와 진입·이탈·체류, 지게차 상대 근접 규칙 및 선택 homography 경로를 구현했다. 지게차 모델이 없어 실제 지게차 검출·근접 위험의 현장 성능은 제공하지 않는다. 일반 차량 모델의 이름을 지게차로 바꾸지 않았다.

SQLite WAL 이벤트, acknowledge 분리, 제한된 JPEG/H.264 전후 증거, 저장 실패 격리, 보존 정책, 실제 MJPEG를 표시하는 한국어 HTML, WebSocket 상태·이벤트, FPS·메모리·현재 관측 상태, CPU 강등, 단일 프로젝트 프로세스 소유, 설치·시작·중지·진단·학습·평가·재생·벤치마크 도구를 포함한다. bbox 가시성·PPE 연결은 휴리스틱이므로 현장 보정과 정답 평가가 필요하다.

## 실행과 산출물

현재 개발 폴더에서는 PowerShell로 아래 명령을 실행한다. DEMO는 보유 test 사진을 H.264로 만든 전송 시험 자료다.

```powershell
cd 'C:\Users\shgus\OneDrive\바탕 화면\vision\industrial-safety-yolo'
.\scripts\start.ps1 -Demo
# http://127.0.0.1:8765
.\scripts\stop.ps1
```

실제 입력은 `.env`와 `configs/cameras.local.yaml`을 설정한 뒤 `start.ps1`만 실행한다. 신규 장비에서는 먼저 `setup.ps1`을 실행한다. UI 파일은 `frontend/dashboard.html`이며 외부 CDN이 없다. 파일만 열면 백엔드 연결 안내가 표시된다. 정확한 단계는 README와 운영 문서에 있다.

`outputs/delivery/industrial-safety-yolo-source.zip`, `industrial-safety-yolo-models.zip`은 소스와 이번 신규 가중치를 분리한 로컬 전달 패키지다. 모델 ZIP을 소스 폴더에 함께 풀어 사용한다. 데이터·비밀번호·DB·가상환경은 포함하지 않는다. `release-manifest.json`이 최종 source commit 및 패키지 hash의 기준이다.

비교용 전수 처리 결과는 `outputs/replay-demo-final/`에 있다. DEMO4개 각200프레임을 실제 모델로 처리해0드롭과 camera/session/frame 일치를 확인했다. 요청된6종 결과 파일과 입력/모델 hash를 제공한다. 원본 현장 영상이 생기면 같은 replay 도구로 별도 run을 만들어야 하며 이 DEMO 사건을 정답으로 간주하지 않는다. 최종 start/stop 스크립트도 실행 검증했으며 전달 시 검증 서버는 종료 상태다.

## 데이터와 학습

PPE 원천 8,845장을 디코딩·라벨 검증했다. Human/Helmet/Safety Vest의 의미가 함께 일치하는 표본만 채택하고, 다른 ontology의 부분 라벨·일반 모자·명시적 no-* 이름을 임의 병합하지 않았다. 같은 옥상 현장, 원본 stem, exact hash 및 dHash 유사성 그룹으로 나눠 4,789장(train3,174/validation930/test685)을 사용했다. 완전한 현장·시간 메타데이터가 없어 완전히 독립된 현장 test로 보장하지 않는다.

물류 폴더의 지게차 ZIP은 0바이트이며 별도 임시 다운로드도 완성 자료가 아니다. 현장 사진 중 기존 예측 표시·모니터 촬영·무라벨 자료는 정답 학습/정량 평가에 사용하지 않았다. 원본은 수정하지 않았다. 출처·라이선스와 채택/제외 이유는 `docs/dataset-audit.md`, 파일별 hash와 split은 `data/manifests/`에 있다.

공식 YOLO26s 초기 가중치 SHA-256:
`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.

신규 본 학습 run `ppe-yolo26s-v2`, seed42, 640px, batch16, AdamW, 30 epochs. validation fitness로 선택한 best SHA-256:
`03c3efe5908fd7e25c21bbc0b92200787bdff7c961995c2a03d36c29afcf81dd`.

Test685장 mAP50 **0.810**, mAP50–95 **0.486**. 클래스별 P/R/AP와 작은 안전모 recall0.510, 실패 이미지, validation-only confidence 탐색은 `docs/model-evaluation.md`에 있다. 신규 모델은 development_evaluated 상태다. 지게차가 빠지고 현장 승인이 없어 `/health/ready`는 503을 반환한다. 서버 생존과 영상 표시가 가능한 상태와 현장 운용 준비 완료를 구분한다.

## 검증과 한계

31개 자동 테스트, 실제 GPU 모델·4개 생성 영상·HTTP/MJPEG/ROI/이벤트 확인·증거 MP4·WebSocket 통합 검사, 별도 새 가상환경 설치·데이터 재구성·CPU smoke 학습·평가를 수행했다. 세부 근거와 미실시 항목은 `docs/verification.md`에 기록했다. 브라우저 자동화 접근이 실패하여 실제 화면 클릭·고DPI/확대 동작은 API 통과와 구분해 미검증으로 남겼다.

데스크톱 RTX5070Ti16GB의 1/2/4채널 및 30분 DEMO 결과는 `docs/benchmark-4ch.md`와 원시 `outputs/benchmarks/`에 있다. 사용자 노트북 RTX5070Ti Laptop12GB 결과가 아니다. 디코딩 준비→JPEG 지연은 host 수신→브라우저 표시·센서 종단 지연이 아니다. 사건 정확도·오경보·야간/가림·ID switch·2시간 운용은 미검증이다. 4채널 완전 구성에 추가될 지게차 모델 비용도 아직 포함할 수 없다.

실제로 검증한 입력 경로는 **로컬 생성 H.264 파일→PyAV→신규 CUDA 모델→HTTP MJPEG**다. C340I 직접 RTSP나 NVR 중계 연결은 하지 못했다. 알려지지 않은 카메라 주소·계정을 임의로 설정하지 않았으며 기본 실제 카메라 4개는 미설정이다.

## 저장소와 필요한 입력

Git root는 위 신규 프로젝트 폴더, branch는 `main`. 단계별 로컬 커밋을 만들었으며 remote와 push는 없다. 기존 저장소의 history/remote를 변경하지 않았다. 실행 코드에 legacy import나 PYTHONPATH 연결이 없고, 새 registry는 프로젝트 내부 가중치와 provenance hash만 사용한다. 설치·코드·DB·로그·모델의 기존 환경 의존성은 추가하지 않았다.

남은 작업에 필요한 입력은 완성된 지게차 이미지/영상 및 검수 라벨·라이선스, 원본 현장 연속 영상과 사건/추적 정답, C340I/NVR 실제 하드웨어·펌웨어·네트워크 경로와 로컬 `.env` 인증정보, 테스트 노트북 접근이다. 이를 받은 뒤 신규 지게차 학습, 전체 모델 비용을 포함한 4채널 30분/2시간, 야간·가림·운전자 승하차·근접 정답 평가, 실제 브라우저·물리 지연 검증을 완료해야 한다. 원격 저장은 저장소 URL과 공개 범위가 필요하다.
