# 모델 평가

모델 `ppe-yolo26s-v2`, SHA256 `03c3efe5908fd7e25c21bbc0b92200787bdff7c961995c2a03d36c29afcf81dd`. 평가일 2026-11-16, desktop RTX5070Ti, PyTorch CUDA. 세 클래스 개발 모델이며 지게차는 미학습이다.

| split / class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| validation / person | .978 | .909 | .960 | .815 |
| validation / helmet | .940 | .844 | .854 | .533 |
| validation / safety_vest | .963 | .916 | .980 | .444 |
| test / person | .900 | .776 | .845 | .600 |
| test / helmet | .671 | .704 | .713 | .354 |
| test / safety_vest | .824 | .832 | .872 | .504 |

validation 930장 mAP50 .931 / mAP50–95 .597, test 685장 mAP50 .810 / mAP50–95 .486. Ultralytics 검출 metric이며 사건 단위 PPE 미착용 recall과 다르다. test P/R은 평가기의 confidence curve 요약이며 운영 conf=.25에서의 고정 임계값 성능과 동일하다고 주장하지 않는다.

본 학습 best 선택은 validation fitness에만 의존했다. validation 930장만 사용한 별도 confidence 탐색에서 macro bbox F1 최대값 .9251의 임계값은 **.5365**였다(CPU FP32). 클래스 F1은 person .9425 / helmet .8930 / safety_vest .9397이다. `models/ppe-yolo26s-v2.threshold-selection.json`에 보존했다. 이 bbox F1 최적값을 PPE 미착용 사건 임계값으로 간주하지 않으며, 측정 기준선의 운영 confidence .25와 시간 규칙은 현장 validation 전 초기값으로 유지했다. 동일 옥상 현장을 한 split으로 묶는 등 누수를 줄였으나 원본 site/time metadata가 불완전하므로 현장 독립 성능으로 사용할 수 없다. bbox 검출 성능만으로 안전 이벤트 목표 달성을 주장하지 않는다.

추가 평가 도구 `scripts.evaluate_slices`는 test 전체를 conf=.25, IoU=.5에서 매칭해 원본 픽셀 면적 32²/96² 기준 크기별 recall을 기록한다. 결과는 `outputs/training/ppe-yolo26s-v2/evaluation/slices.json`, 실패 사례 24장까지는 `evaluation/failures/`에 생성한다. 이 자료는 운영 acceptance가 아니다.

실행 결과: 작은 안전모 157개 중80개 매칭(recall .510), 중간 안전모487개 중369개(.758), 큰 안전모285개 중231개(.811). 중간 사람22개 중4개(.182), 큰 사람1,475개 중1,168개(.792). 중간 조끼145개 중90개(.621), 큰 조끼896개 중823개(.919). 작은 사람/조끼 표본은 없어 N/A다. 685장 중525장에서 하나 이상의 미검출 또는 미매칭 예측이 있었다. 면적 기준 원본 자체가 640×640인 데이터의 결과이며 실제 4MP 원거리 성능으로 환산하지 않는다.

미평가: 지게차 bbox, 원본 연속 영상의 ID switch/fragmentation, 사건 단위 precision/recall, 오경보/카메라·시간, IR 야간·실제 가림·운전자 승하차·지게차 앞 보행자. 입력과 정답이 없는 지표는 0 또는 임의 수치로 채우지 않았다.

목표인 중요 침입·근접 사건 recall95%, PPE 미착용 recall90%, 사건 precision95%, 오경보1건/카메라·시간 이하는 **미확인**이다. 현재 안전모 test bbox recall .704, precision .671은 추가 데이터·라벨 점검·현장 촬영 조건 개선이 필요함을 보여 준다.
