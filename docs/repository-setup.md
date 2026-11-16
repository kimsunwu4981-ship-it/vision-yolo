# 독립 저장소

프로젝트 루트: `C:/Users/shgus/OneDrive/바탕 화면/vision/industrial-safety-yolo`.
상위 vision 폴더는 Git 저장소가 아니었고 이 하위 폴더에서 새 `git init -b main`을 수행했다. 기존 저장소 clone/fork/worktree/.git 복사는 사용하지 않았다. 새 가상환경은 이 폴더의 `.venv`이다.

소스·HTML·테스트·실행/학습/평가 스크립트·고정 requirements·example 설정·데이터와 모델 provenance manifest·문서를 commit한다. 대용량 이미지/영상, 공식 초기 가중치와 새 학습 가중치, DB·증거·로그·비밀 설정·가상환경은 제외한다. 초기 납품 시에는 원격 저장소를 연결하지 않았다. 이후 사용자의 업로드 요청으로 [nohhyunseong1-hash/vision-yolo](https://github.com/nohhyunseong1-hash/vision-yolo)를 비공개로 생성해 `origin`에 연결했다. 업로드 브랜치는 `main`이다.

커밋 작성자는 자동화 주체 `Codex <codex@localhost>`이다. 사용자 이름을 임의로 추정하지 않았다. Git 전역 설정은 바꾸지 않았다. 시스템 Git의 접근 불가 전역 excludes 경로 때문에 커밋 명령에서 프로젝트의 빈 excludes 파일을 일회성으로 지정했다. 마지막 커밋 ID와 검증 결과는 최종 보고를 참조한다.

재현 검증용으로 commit `ea38ce9`를 `git archive`하여 `outputs/clean-checkout`에 풀고, 새로운 별도 `.venv`에서 고정 의존성 설치, 데이터 3,174/930/685 재구성, 공식 초기 가중치의 CPU smoke 학습·validation을 완료했다. 이후 소스 archive `0d4639f`를 반영하여 setup, 31개 테스트, 실제 HTTP 실행을 확인했다. 기존 `.venv`나 legacy 프로젝트를 복사하지 않았다. checkout의 운영 모델은 Git에 없으므로 모델 미준비로 표시되며 실제 HTTP 생존·대시보드는 200이었다. CPU smoke `clean-smoke-v1`의 checkpoint SHA-256은 `87ba3aeeb1ab223cb784b07743135504c799fb1e380f22dfe8ed4f82b83e58e2`다. 본 학습 30 epoch 전체를 이 checkout에서 다시 수행한 것은 아니다. 이후 최종 소유 프로세스 확인·보고서 변경은 원래 프로젝트에서 검증했다.

패키징 전 `git ls-files`로 `.env`, `.venv`, 원본/파생 이미지, `.pt`, `.db`, 증거가 추적되지 않는지 검사한다. `app/`에는 legacy 경로나 PYTHONPATH 조작이 없다. 데이터 접근 경로는 audit와 manifest에만 있고 실행 모델은 이 프로젝트 ROOT 안 registry 파일만 허용한다.

`python -m scripts.package_release`는 현재 HEAD의 소스 ZIP과 hash 검증된 이번 학습 모델 ZIP을 `outputs/delivery`에 별도로 만든다. 같은 폴더에 두 ZIP을 풀고 setup을 실행한다. 데이터·인증정보·가상환경은 포함하지 않는다. 정확한 납품 commit과 ZIP hash는 `outputs/delivery/release-manifest.json`에 기록한다. 기존 로컬 ZIP은 생성 당시 commit의 스냅샷이며 이후 GitHub 변경과 자동 동기화되지 않는다. 가중치 ZIP은 GitHub 소스 업로드에 포함하지 않는다.
