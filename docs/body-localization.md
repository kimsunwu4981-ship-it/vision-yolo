# 머리·몸통 위치 보정 — 2026-11-16

이전 화면은 사람 전신 박스를 일정 비율로 잘라 머리와 조끼 영역을 표시했다. 몸을 숙이거나 측면을 보이면 이 사각형이 실제 머리·몸통과 어긋났다. 현재는 전신 `PERSON` 검출을 유지하고 현재 사람 crop에서 얼굴·어깨·골반 관절을 추정한다. `app/pose_regions.py`가 이를 원본 표시 픽셀의 머리·몸통 영역으로 변환한다. PPE 부재가 확인되면 해당 위치에 빨간 `NO - HARD HAT`, `NO-VEST`를 표시한다.

이는 관절 기반 외곽 추정이다. 직접 학습한 head/torso bbox 또는 픽셀 segmentation, 독립 정답으로 입증된 정확한 윤곽으로 표현하지 않는다. 머리는 코·눈·귀, 몸통은 양 어깨·양 골반의 신뢰도 .5 이상을 요구한다. 크기 제한, 일대일 소속, 겹친 사람의 모호성 검사를 거친다. 필수 좌표가 없으면 unknown으로 두며 과거 박스나 전신 비율로 채우지 않는다. 확인 대기 중에도 unknown이므로 항상 세 박스를 강제로 표시하지 않는다.

## 새 모델과 학습 범위

`body-pose-yolo26n-v2` SHA256: `f6db8b1b1eb46c639a1136a8406fc4f0a3b425cad837e5ff8a1385bb701bfe6c`.

공식 초기 `yolo26n-pose.pt` SHA256: `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`. 기존 프로젝트의 가중치는 사용하지 않았다. 원래 사람 학습 분할에서 균형을 맞춘 403장으로 사람 confidence 출력층만 새로 학습했다. 8 epochs, 640, batch16, AdamW lr .0001, seed53, CUDA FP32, bbox detection loss만 사용했다. validation 930장으로 best를 선택하고 test 685장으로 사람 bbox를 평가했다. 관절 정답을 임의로 만들지 않았다.

얼굴·관절 위치 가중치, backbone/neck, box regression, 관련 통계는 공식 초기값 그대로 동결했다. 저장된 가중치 비교에서 동결된 파라미터·buffer의 최대 차이는 0이다. confidence 출력층의 실제 변경량과 manifest SHA256은 `models/body-pose-yolo26n-v2.provenance.json`에 기록했다. 따라서 **관절 좌표를 신규 관절 정답으로 재학습한 모델은 아니다.** 새로 학습·평가한 사람 confidence 적응과 동결된 공식 pose 사전학습을 조합한다.

| 보조 pose 모델의 사람 bbox 평가 | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| validation | .9368 | .9220 | .9115 | .3006 |
| development test | .9179 | .8036 | .8580 | .4490 |

위 수치는 관절/머리/몸통 정확도가 아니다. 운영 PERSON 박스는 기존 v5가 담당한다. pose는 `nms=false`, person confidence .25, keypoint confidence .5, 640 정사각 crop, CUDA FP16으로 실행한다. 이 임계값은 보수적인 개발 설정이며 관절 신뢰도 확률이 보정되었다고 주장하지 않는다.

재현 명령은 `python -m scripts.download_weights --model yolo26n-pose.pt`, 사람 v4 manifest에 대응하는 파생 데이터 준비 후 `python -m scripts.train_pose_adapter --run-id body-pose-new`다. 이 도구는 표준 PoseModel checkpoint를 저장하고 detection validator로 bbox만 평가한다. v1은 동결 대상 검증 assertion에서 학습 전에 중단했고 v2에서 pose head의 모든 하위 모듈을 명시적으로 동결했다.

## 직접 부위 검출 후보를 채택하지 않은 이유

보유 데이터의 `no hat` 13번과 `no vest` 14번 박스가 실제 머리·몸통을 가리키는 것을 32개 음성 포함 표본에서 확인했다. 8/9/13을 head, 14/15를 torso로 묶어 위치만 학습했다. 일반 hat을 안전모 착용으로 바꾸지 않았다. source family/stem/hash/dHash를 묶고 기존 모델 split과 충돌하는 그룹은 제외했다. train878/validation254/test770장이며, 음성 라벨과 현장 종류가 split마다 크게 불균형했다.

- `body-parts-yolo26n-v1`: 공식 n에서 새 학습. test head AP50 .1907, torso .4138, 평균 AP50–95 .1117. 실제 크레인 영상의 몸통 위치가 불안정해 미등록.
- `body-parts-yolo26n-v2`: v1을 이어 원본 정답에서 파생한 instance context crop으로 추가 학습. train1350/validation256/test1890 crop. crop test head AP50 .3338, torso .5298, 평균 AP50–95 .1752. validation에서 고른 confidence .4745를 적용하면 크레인 영상의 몸통 누락이 많아 미등록. crop test는 v1의 전체 이미지 test와 같은 난도가 아니며 crop들도 독립 표본이 아니다.

학습·평가 기록은 로컬 `outputs/training/`에 보존하고 source/manifest를 Git으로 관리한다. 부위 모델의 validation 성능만 근거로 개선되었다고 보고하지 않았다.

## 크레인 영상 전수 재시험

원본 `크레인 점검.mp4` 546프레임/22.75초를 새 구성으로 모두 재추론했다. 영상은 학습 데이터에 포함하지 않았다. 다만 개발 중 반복해서 시각 확인했으므로 blind test는 아니다.

| 항목 | 결과 |
|---|---:|
| PERSON 관측 프레임 | 546/546 |
| 머리 위치 확보 프레임 | 512/546 |
| 몸통 위치 확보 프레임 | 538/546 |
| PERSON + 두 미착용 빨간 박스 동시 표시 | 403/546 |
| 사람 추적 ID | 1 |
| 처리 평균 / p95 | 25.57 / 28.61 ms |

위 횟수는 정답 recall이 아니다. 처리 시간은 offline 추론·규칙 처리 구간이며 decode/encode/실시간 4채널/브라우저 지연을 뜻하지 않는다. 12개 표본을 시각 확인했고 숙인 자세에서도 머리와 몸통을 따로 따라가는 것을 확인했다. 일부 관절이 불확실하거나 확인 대기 중인 구간에서는 빨간 박스가 사라진다.

결과 폴더는 `outputs/crane-inspection-localized-20261116/`. 새 영상 `crane-inspection-improved.mp4`, 이전 실제 비율 박스 영상과 좌우 비교한 `crane-inspection-before-after.mp4`, 표본 `localization-contact.jpg`, 원시 tracks/detections 및 `localization-report.json`, `video-qa.json`을 보존한다. 두 MP4 모두 546프레임/22.75초/음성1개, 단조 PTS, rotation0을 전수 재디코딩 확인했다. 원본 SHA256은 `5b20c7c4175b7db1fd6482cd48ce68bdfacf4d7b615098c90c5885964cee5cd1`로 불변이다.

소프트웨어 전체 tests 47개와 실제 CUDA 3모델 HTTP 통합 9개 검사를 통과했다. crop 오프셋 한 번 복원, 다른 카메라/프레임 분리, 누락 관절/NaN 배제, 위치 이동, 모호한 소속 거부를 포함한다. 실제 현장 부위 bbox 정답, 관절 정답, PPE 사건 정확도 및 목표 노트북 인수는 별도 검증이 필요하다.

추가로 동일 크레인 파일 네 복제 입력을 desktop RTX5070Ti에서 35초 실행했다. 첫5초를 제외한 채널별 평균 추론은18.88–20.05FPS, 미리보기10.57–10.62FPS, 최종 decode-ready→JPEG p95는156–157ms였다. 큐 최대1, 저장 오류/저장 드롭0, 파이프라인 오류 없음. 캡처 최신 슬롯의 누적 프레임 드롭은932 전후(채널별912–954)이며 모델 로딩 중 캡처된 프레임까지 포함한다. 실물 카메라 네 대, 목표 노트북, 브라우저 지연 또는 장시간 인수 결과가 아니다. 로그: `outputs/benchmarks/run-2026-11-16T141821-153525+0000/`.
