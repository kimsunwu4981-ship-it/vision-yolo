# C340I / NVR1104H-4P 연결

현재 실제 카메라 IP·자격증명·장치 하드웨어/펌웨어를 받지 못해 **직접/NVR RTSP 연결 모두 미검증**이다. 자동 LAN 스캔이나 다른 기존 프로그램의 비밀 설정 탐색은 하지 않았다.

권장 초기 시험은 유선 LAN + H.264 + RTSP TCP + 메인 영상이다. 원본 해상도, 모델 입력 640, 미리보기 최대폭 960은 서로 다르다. 작은 PPE를 위해 메인부터 평가하고 서브만으로 충분하다고 판단하지 않는다. 메인/서브의 crop·화각이 다르면 박스를 복사하지 않는다. IR 야간을 별도 표본으로 평가한다.

카메라 직접 주소:

```text
rtsp://USER:PASSWORD@CAMERA_IP:554/stream1
rtsp://USER:PASSWORD@CAMERA_IP:554/stream2
```

stream1=메인, stream2=서브이며 카메라 번호가 아니다. 사용자/비밀번호의 특수문자는 URL percent-encoding한다. 실제 URL은 `.env`의 `CAMERA_1_RTSP` 등에만 넣는다.

NVR PoE 구성: C340I 4대 → NVR PoE 포트 → NVR LAN → 현장 스위치 → PC. Network Isolation이 켜져 있으면 LAN PC가 카메라 서브넷에 직접 접근하지 못할 수 있다. 승인 없는 격리 해제는 하지 말고 지원되는 NVR 스트림을 우선 검증한다.

```text
rtsp://USER:PASSWORD@NVR_IP:554/live/1/1/avm
rtsp://USER:PASSWORD@NVR_IP:554/live/2/1/avm
rtsp://USER:PASSWORD@NVR_IP:554/live/3/1/avm
rtsp://USER:PASSWORD@NVR_IP:554/live/4/1/avm
```

장치별 지원 여부를 반드시 확인한다. 실제 HW/FW, 2.8/4/6mm 렌즈, codec/profile/fps/bitrate, 네트워크 경로, TCP timeout, 권한, NVR 재중계 여부를 점검표에 기록한다. NVR 녹화와 공유되는 stream profile 변경은 녹화 해상도·대역폭에도 영향을 준다.

구형 C340I 문서에는 메인 최대 2560×1440, 서브 640×480이 표시된다. 일부 V1.30 페이지에는 서브 736×416도 있어 제품명만으로 profile을 고정하지 않는다.

공식 근거: [카메라 RTSP](https://www.tp-link.com/baltic/support/faq/3718/), [C340I V1 사양](https://www.tp-link.com/tr/business-networking/vigi-network-camera/vigi-c340i/v1%20%284mm%20lens%29/), [V1.30 사양](https://www.tp-link.com/es/business-networking/vigi-network-camera/vigi-c340i/v1.30%20%284mm%20lens%29/), [NVR RTSP](https://www.tp-link.com/fr/support/faq/5223/), [Network Isolation](https://www.tp-link.com/us/support/faq/4672/).

호스트 수신→JPEG 지연과 센서→브라우저 물리 종단 지연을 구분한다. 실제 장비에서 LED/시계 동시 촬영으로 종단 지연을 측정하고, 브라우저 표시 시각과 호스트 시간을 동기화해 별도로 기록한다.
