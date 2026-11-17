# 모델 평가

2026-11-17 최신 구성은 사람 scale-v7 / PPE v2 / pose v2다. 새 보호구 후보는 영상 회귀로 제외했다. [기존 라벨 기반 개선](recognition-improvement-20261117.md)에 같은 운영점의 TP/FP/FN, 후보 제외 이유와 원거리 CCTV 미해결 결과를 기록했다. 아래는 이전 평가 이력이다.

머리·몸통 표시 보정은 [부위 위치 보정](body-localization.md)에 별도로 기록했다. 현재 사람 v5 + PPE v2에 `body-pose-yolo26n-v2`를 추가했다. 아래 사람/PPE 수치를 pose/부위 정확도로 해석하지 않는다.

## 2026-11-16 검출 누락 개선 후 구성

현재 사람은 새로 학습한 `person-yolo26s-v5`, 안전모·조끼는 기존 `ppe-yolo26s-v2`의 해당 클래스만 사용한다. 원본 공식 가중치를 그대로 운영하지 않는다. 사람 신뢰도 .25, 기존 추적 유지용 관측 하한 .10, 보호구 신뢰도 .5365365를 적용했다. 보호구 값은 아래의 기존 validation-only bbox F1 선택값이며 사건 정확도 보증값이 아니다. 모델별 `nms=true`를 명시했다. 설치된 Ultralytics 8.4.215에서는 `nms=false`만 one-to-one head이고 기본값 None도 one-to-many head를 사용하므로, 준비 단계와 실제 추론이 같은 설정을 쓰도록 고정했다.

검출 누락의 주요 증거는 기존 train 3,174장 중 2,951장(93.0%)이 같은 옥상 현장 계열이라는 점이다. 입력 해상도·정사각형 변환·FP32 변경은 크레인 영상의 누락을 해결하지 못했다. 사람 전용 모델은 원래 train 중 현장 그룹 최대 180장으로 균형을 맞춘 403장으로 새로 학습했으며, 원래 validation 930장과 test 685장은 학습에 넣지 않았다. 크레인 영상도 학습에 사용하지 않았다. 여러 후보의 test 결과를 확인했으므로 이 기존 개발 test를 새로운 blind acceptance로 간주하지 않는다.

| 고정 운영점, 기존 test 685장, FP16, IoU≥.5 | 이전 Precision | 개선 Precision | 이전 Recall | 개선 Recall |
|---|---:|---:|---:|---:|
| 사람 (.25 유지) | .8989 | .9405 | .7836 | .9178 |
| 안전모 (.25 → .5365) | .6359 | .7093 | .7330 | .6803 |
| 조끼 (.25 → .5365) | .7825 | .8714 | .8780 | .7877 |

사람 TP/FN은 1,173/324에서 1,374/123, FP는 132에서 87로 개선됐다. 보호구는 오검출을 줄이는 대신 일부 검출을 놓치는 절충이 있다. 보호구 객체 recall 감소는 미착용 오경보를 늘릴 수 있으므로 현장 사건 정답으로 추가 평가해야 한다. 시간 확인과 unknown 처리는 유지했다. 재현 명령: `python -m scripts.evaluate_operating_points --output outputs/operating-points.json`.

confidence curve 기준 새 사람 모델 test P/R/AP50/AP50–95는 .9261/.8771/.9346/.5536이다. 이전 .9003/.7762/.8445/.6005 대비 검출 재현율은 개선됐지만 엄격한 박스 위치 AP50–95는 낮아졌다. ROI 경계 정확도가 모든 현장에서 개선됐다고 주장하지 않는다. 체크포인트는 validation fitness로, 추론 head는 validation AP50–95로 선택했다.

크레인 영상 전체와 표본 박스 결과·영상은 로컬 `outputs/crane-improvement/` 및 최종 replay 폴더에 보존한다. 프레임에서 사람이 검출된 비율은 정답 recall과 구분하며, 12개 표본 전신 박스는 assistant가 대략 검토한 기준으로 독립적인 사람 정답이 아니다. 지게차, 야간, 실물 4대 카메라 및 목표 노트북 검증은 여전히 미완료다.

## 개선 전 기준선

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
