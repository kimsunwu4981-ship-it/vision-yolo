# 기존 라벨 기반 사람 개선과 보호구 후보 검증

사람 모델은 기존 개발 이미지 성능이 개선돼 적용했다. 새 보호구 후보는 이미지 수치가 일부 좋아져도 실제 영상에서 맨머리를 안전모로 오인해 제외했다. **보호구 검출기 교체와 ‘크레인 재해.mp4’의 원거리 사람 누락 해결은 달성하지 못했다.** 기존 PPE v2를 유지한다. 아래 개발 이미지 성능을 해당 CCTV 영상의 정확도로 해석하면 안 된다.

## 적용 구성

| 역할 | 모델 | 활성 클래스 | confidence / head |
|---|---|---|---|
| 사람 | person-yolo26s-scale-v7 | person | .25 / one-to-many + NMS |
| 안전모 | ppe-yolo26s-v2 | helmet | .5365365 / one-to-many + NMS |
| 조끼 | ppe-yolo26s-v2 | safety_vest만 | .5365365 / one-to-many + NMS |
| 부위 위치 | body-pose-yolo26n-v2 | head, torso | 기존 .25 / keypoint .5 |

새 사람 SHA256: `3bd15c733eed1a9aa033719d2da2b3c889501f14c15ac1e668a6f2be5bcbcec9`.

**미채택** 안전모 후보 SHA256: `dae8ee7512f1da818fe97bca89735fa7061081c85fd568786cc17be4e03b0744`. 운영 registry에는 등록하지 않는다. 기존 PPE checkpoint 한 개에서 안전모와 조끼를 함께 처리해, 현재 총 모델 수는 사람/PPE/pose 3개다.

체크포인트·데이터 해시·평가·정책은 `models/*.provenance.json`, `models/*.evaluation.json`, `models/ppe-yolo26s-scale-v5.operating-policy.json`, `data/manifests/*scale*/`에 보존했다. 이전 구성은 `models/model_registry.before-scale-20261117.json`에 있다. 가중치 원본은 Git에 넣지 않는다.

## 고정 운영점 비교

동일한 기존 test 685장, 원본 좌표, FP16, IoU≥.5, 클래스별 Hungarian 일대일 매칭. 추적용 약한 person 관측은 제외했다. 변경 전은 사람 v5와 PPE v2의 실제 이전 설정이다. 아래 값은 Ultralytics confidence curve의 요약 P/R과 별도로 직접 계산했다.

| 클래스 | 이전 정밀도 | 이후 정밀도 | 이전 재현율 | 이후 재현율 | TP / FP / FN 전 → 후 |
|---|---:|---:|---:|---:|---|
| 사람 | 94.05% | 95.01% | 91.78% | 92.85% | 1374/87/123 → 1390/73/107 |
| 안전모 | 70.93% | 70.93% | 68.03% | 68.03% | 632/259/297 → 동일 |
| 조끼 | 87.14% | 87.14% | 78.77% | 78.77% | 820/121/221 → 동일 |

별도 Ultralytics native test AP50 / AP50–95는 사람 .93457/.55353 → .96538/.55463이다. 사람 위치 AP50–95 개선은 매우 작으므로 박스 경계가 전반적으로 크게 정확해졌다고 주장하지 않는다. 소형 객체 합성 test의 사람 AP50 / AP50–95는 .81361/.43022 → .89501/.50514이며, 같은 원본을 합성한 상관된 표본이므로 독립 현장 이미지 성능이 아니다.

미채택 안전모 v5의 test AP50/AP50–95는 .78165/.37951로 기존 .71314/.35335보다 높았다. 그러나 validation F1 임계값 .66066에서는 test 정밀도 90.78%, 재현율 62.54%로 재현율이 후퇴했다. validation F2 .62062도 검토했다. 기본 detector 임계값 .25에서는 test 정밀도 79.57%, 재현율 76.32%로 모두 올랐지만, **근거리 전수 영상에서 맨머리를 안전모로 오인해 5프레임을 equipped로 확정했다.** 세 박스 동시 표시도 202프레임으로 악화돼 해당 후보를 제외했다. 높은 임계값의 사람+새안전모 조합은 442프레임을 표시했지만, 안전모 객체 재현율 후퇴를 숨긴 채 채택하지 않았다.

이러한 개발 정책 비교에는 기존 test와 진단 영상 확인이 영향을 미쳤으므로 최종 독립 test나 validation-only 선택으로 주장하지 않는다. 사람도 validation AP가 조금 더 높은 one-to-one head에서 근거리 영상 두 프레임에 거의 같은 사람 박스가 중복돼, NMS가 있는 one-to-many head를 선택하고 다시 평가했다.

## 학습 데이터와 실패한 후보

사용자 답변대로 기존 라벨 데이터만 사용했다. 두 크레인 영상은 학습·증강·배경 추출에 전혀 사용하지 않았다. 원본 데이터는 읽기 전용이며 새 외부 학습 데이터도 추가하지 않았다. 기존 프로젝트에서 이번에 학습한 가중치를 이어 학습했고, 초기 출처는 공식 YOLO26s다. 다른 프로젝트의 모델·코드·환경을 가져오지 않았다.

- 사람 v6: 기존 403장을 native/축소/격자 장면으로 확장해 20 epoch 전체 층 학습. 합성 성능은 높았지만 기존 test AP50–95 .41492로 하락했고, 재해 영상의 철재·설비를 사람으로 오인해 제외했다.
- 사람 v7: 같은 403개 원본, 29개 그룹. 원본 장면 3배, 원래 작은 사람이 있는 장면 9배 샘플링, 축소 장면 1배, 기존 사람 라벨과 16px 이상 떨어진 배경 crop 121개. 총 학습 입력 행 1,841개이며 독립 원본 수가 증가한 것은 아니다. 배경 crop은 기존 라벨의 완전성에 의존한다. backbone 10개 층의 가중치와 BN 통계를 동결하고 10 epoch, AdamW lr .00002, 회전 10도, 약한 scale/translate로 학습했다. 동결 상태 최대 변화는 0이다. checkpoint는 변경하지 않은 native validation으로 선택했다.
- PPE v4: 기존 upper 403개와 기존 lower PPE 605개를 사용했다. lower 자료는 사람 라벨이 불완전하므로 사람 학습에 쓰지 않았다. 모호한 `hat` 클래스 8이 있는 이미지는 제외했다. 기존 9 helmet→0, 15 vest→1로 매핑했고 upper는 1→0, 2→1이다. 총 1,008개 원본으로 native/축소/격자를 만들어 20 epoch 학습했다.
- PPE v5: v4에서 upper native 5배, lower native 1배, upper 축소 1배로 총 3,023개 입력 행을 구성해 10 epoch 보정했다. backbone 가중치·BN 통계 변화 0. 안전모는 위 영상 회귀 때문에, 조끼는 AP50 .85131 / AP50–95 .48513이 기존 .87189/.50497보다 낮아 모두 채택하지 않았다.

모든 원본의 기존 split을 유지했다. 각 후보에 대해 train/validation/test 사이 이미지 SHA256 및 출처 그룹 교집합은 0이며 파생 이미지·라벨과 참조 원본 해시도 확인했다. 이는 원본 정답의 완전성이나 촬영 현장의 독립성을 인증하는 절차는 아니다. 학습 후보를 반복해서 확인한 기존 개발 test라는 한계가 있다.

공식 YOLO11s도 변경 없는 진단용 초기 모델로만 비교했으나 원거리 문제를 해결하지 못했다. 학습·등록·배포하지 않았다.

## 영상 결과와 제한

두 원본의 SHA256을 실행 전후 확인했다. 모든 프레임을 처리했으며 입력 영상의 누락, 덮어쓰기, 학습 사용은 없다. 최종 프레임·상태 집계와 영상 QA는 아래 폴더의 `test-report.json`, `video-qa.json` 및 이 문서의 동명 JSON에 보존한다.

최종 근거리 결과는 사람 박스 546/546프레임 유지, head 512→514, torso 538→546, 두 미착용 부위와 PERSON의 세 박스 동시 표시 403→418프레임이다. 안전모를 착용했다고 확정한 오판은 0프레임이다. 최종 원거리 사람 박스는 88→28프레임으로 줄었고, 이는 재현율 개선이나 오검출 감소의 정답 지표가 아니다. 표본에서도 실제 사람을 놓치므로 원거리 개선으로 주장하지 않는다. 두 결과 영상은 전체 프레임 수, 증가하는 PTS, 원본 방향과 음성 스트림 보존을 재디코딩 확인했다.

- 최종 채택 근거리: `outputs/crane-inspection-accepted-20261117/`. 720×1280, 546프레임, 22.75초. 이전 실제 분석 영상과 동기화한 비교본이다.
- 최종 채택 원거리: `outputs/crane-accident-accepted-20261117/`. 1280×720, 1,092프레임, 영상 45.5초/음성 포함 컨테이너 45.97초. 표본 장면에서도 사람이 누락되며 개선 성공 영상으로 소개하지 않는다.
- `*-improved-20261117/`는 높은 임계값의 미채택 PPE 후보, `*-final-20261117/`는 낮은 임계값에서 영상 회귀가 발견된 미채택 후보 기록이다. 이 폴더 이름을 채택 여부로 해석하지 않는다.

사람 박스 프레임 수, 부위 위치 확보 수, 세 박스 동시 표시 수는 독립 정답 재현율이 아니다. 작은·가려진 사람, 확인되지 않은 얼굴·어깨·골반은 여전히 unknown으로 처리한다. 빨간 미착용 박스를 만들기 위해 고정 비율의 머리·몸통 박스를 복원하지 않았다. 실제 body ROI의 일대일 연결·시간 확인 조건도 유지했다.

원거리 영상은 기존 학습 원본과 사람 크기, 상부 시점, 가림, 영상의 대비·질감이 크게 다르다. 합성 축소만으로 이 차이를 해결하지 못했다. 이는 이번 실험의 한계이며, 어떤 전처리나 추가 학습으로도 불가능하다고 단정하는 것은 아니다. 실물 카메라·목표 노트북·야간·미착용 사건 정답 인수는 여전히 미완료다.

## 재현과 검증

새 도구는 기존 manifest와 프로젝트 체크포인트를 입력으로 사용한다. 이미 존재하는 run을 덮어쓰지 않으며 새 `--run-id`를 지정한다. `train_existing_scales` → `train_person_conservative` / `train_ppe_conservative`의 기존 파생 데이터 의존성은 각 provenance에 기록돼 있다.

```powershell
python -m scripts.audit_scale_manifests person-yolo26s-scale-v7 ppe-yolo26s-scale-v5
python -m scripts.evaluate_operating_points --before-registry models/model_registry.before-scale-20261117.json --output outputs/fixed-operating-recheck.json
python -m scripts.replay video.mp4 --output outputs/new-replay --registry outputs/candidate-registry.json
```

`--registry`로 후보를 격리 재생할 수 있다. 실제 사용한 registry 해시를 run manifest/config revision에 기록하며 운영 파일을 바꾸지 않는다. 빈 후보 registry가 운영 모델로 몰래 대체되지 않는 테스트를 추가했다. 전체 pytest 48개 통과. 실제 CUDA 모델로 HTTP/MJPEG/WebSocket·재시작·이벤트·증거 통합 검사를 수행했다. 해당 전송 fixture는 정지 사진 기반 DEMO이며 CCTV 정확도 평가는 아니다.
