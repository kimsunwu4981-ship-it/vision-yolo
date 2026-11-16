# 데이터 감사

후속 부위 위치 시험에서는 기존에 제외했던 13 `no hat`/14 `no vest` 계열을 32개 표본으로 의미 검토한 후 머리/몸통 위치 학습에만 사용했다. 일반 hat을 안전모 착용으로 바꾸지 않았다. 두 직접 부위 모델은 실제 영상 성능 부족으로 미등록이다. 현재 등록 pose 모델은 기존 사람403장과 검증930/test685장만 사용하며 관절 정답을 생성하지 않았다. [분할·변환·제한 상세](body-localization.md). 아래 통합 모델의 원래 선택 기록은 그대로 보존한다.

조사일 2026-11-16. 기존 프로젝트에서는 지정된 데이터·라벨·data.yaml·README만 읽었다. 원본을 수정하거나 이전 학습 가중치·코드·환경을 가져오지 않았다. 신규 외부 학습 데이터는 다운로드하지 않았다.

| 원본 | 결과 / 처리 |
|---|---|
| `D:/agent/vigent/data/datasets/construction_ppe_v3i` | 이미지·라벨 8,845쌍. 디코딩, 5열 YOLO bbox, class index, 유한값, 정규화 경계 검사 통과. 원본 metadata의 16개 class를 감사 |
| `D:/121.물류창고 내 작업 안전 데이터` | `VS_07_지게차.zip` 0바이트, `.zip.irx795` 다운로드 임시파일 860,258,304바이트. 완성 라벨·데이터 없음. 둘 다 제외 |
| `C:/Users/shgus/OneDrive/바탕 화면/VISION_현장실측사진` | 사진 18장 + 개요 이미지 1장. 일부 기존 모델 박스/모니터 화면/데모. 나머지 원본 사진도 정답 라벨 없음. 학습·정량 정확도 평가에서 모두 제외 |

전체 inventory 8,866개 레코드. 원본 이미지 SHA256 중복 78그룹. 이미지 hash, 라벨 hash, 크기, 밝기 요약, dHash, 작은 bbox 수, 원본 split, source family, 채택/제외 이유를 CSV에 기록했다. 임시파일은 완성 데이터로 처리하지 않고 대용량 hash 대상으로도 삼지 않았다.

샘플을 직접 검토한 `Human`→person, `Helmet`→helmet, `Safety Vest`→safety_vest만 통합 모델에 사용했다. `vest`는 안전조끼 샘플이었으나 해당 소스는 사람 라벨이 없는 부분 라벨 계열이다. `helmet/vest` 계열 전체를 합치면 미라벨 사람을 배경으로 학습하게 되므로 제외했다. `hat`은 안전모로 일괄 매핑하지 않았으며 `no hat/no vest`도 의미·음성 박스 검증 전이라 제외했다. 지게차 class를 이름만 만들어 실제 검출 가능하다고 표시하지 않았다.

사람 라벨이 존재하고 upper-case ontology의 class 0–4만 포함하는 이미지 4,802장을 후보로 뒀다. 동일 SHA256 중복 13장을 제거하여 **4,789장**을 선택했다. 원본 전체 이미지 중 4,043장은 부분/혼합 라벨 문제로 제외했다. 이 조건과 샘플 확인은 전체 객체 라벨의 완전성을 보증하지 않는다. 원본 장면의 미라벨 객체·상업 이미지·가림·음성 장면을 사람이 추가 검수해야 한다.

분할은 원본 random train/valid/test를 그대로 쓰지 않았다. 파일명 source family, Roboflow hash 제거한 원본 stem, exact SHA256, dHash Hamming≤4를 union한 뒤 group별로 배정했다. 유사 이미지 쌍 17,985개를 같은 group에 묶었다. 직접 본 샘플에서 `Video`와 `KakaoTalk`가 동일 옥상 촬영현장임을 확인해 추가로 union했다. 최종 35개 group, train 3,174 / validation 930 / test 685장이다. 인접 프레임이 확실한 Video 계열 전체는 한 split에만 있다.

**원본 영상 ID·현장·시간대 metadata가 완전하지 않으므로 이 test를 완전한 현장 독립 test라고 주장하지 않는다.** group isolation은 개발 누수 완화다. 외형이 다른 프레임·회전·증강까지 모든 교차현장 중복을 보증하지 않는다. 일부 원본은 이미 640×640 stretch/노이즈/회전 형태이며 정보 손실을 복구할 수 없다. README의 augmentation 설명만으로 무증강을 가정하지 않았다.

객체 수:

| split | 사람 | 안전모 | 조끼 |
|---|---:|---:|---:|
| train | 3,300 | 3,161 | 3,190 |
| validation | 965 | 602 | 905 |
| test | 1,497 | 929 | 1,041 |

조도는 image luminance로만 기록한다. 이 값을 주야간 정답으로 바꾸지 않는다. 가림·IR 야간·현장 이벤트 정답이 없어 해당 정확도는 미평가다. 객체 크기별 test recall 및 실패 사례 이미지는 신규 모델 평가 도구가 따로 생성한다.

원본 metadata의 라이선스는 CC BY 4.0이고 출처는 [Construction PPE v3](https://universe.roboflow.com/skcet-g4h72/construction-ppe-rdhzo/dataset/3)다. 원 기여자의 모든 이미지 권리는 별도 확인되지 않았다. 데이터는 Git에 재배포하지 않으며 출처·변환 내역은 manifest와 THIRD_PARTY_NOTICES에 유지한다.

지게차 신규 학습에는 완전히 다운로드된 원천 영상/이미지, 검수 bbox, 클래스 설명, train/val/test 원본 그룹, 이용 조건이 필요하다. 지게차만 라벨된 데이터라면 사람·PPE 모델과 별도 모델로 학습할 수 있고, 이때 전체 추론 비용을 다시 측정해야 한다. 현재 런타임은 클래스가 겹치지 않는 복수 모델 registry를 지원한다.
