# 독립 저장소

프로젝트 루트: `C:/Users/shgus/OneDrive/바탕 화면/vision/industrial-safety-yolo`.
상위 vision 폴더는 Git 저장소가 아니었고 이 하위 폴더에서 새 `git init -b main`을 수행했다. 기존 저장소 clone/fork/worktree/.git 복사는 사용하지 않았다. 새 가상환경은 이 폴더의 `.venv`이다.

소스·HTML·테스트·실행/학습/평가 스크립트·고정 requirements·example 설정·데이터와 모델 provenance manifest·문서를 commit한다. 대용량 이미지/영상, 공식 초기 가중치와 새 학습 가중치, DB·증거·로그·비밀 설정·가상환경은 제외한다. 원격 저장소는 지정되지 않아 remote 설정과 push를 수행하지 않는다.

커밋 작성자는 자동화 주체 `Codex <codex@localhost>`이다. 사용자 이름을 임의로 추정하지 않았다. Git 전역 설정은 바꾸지 않았다. 시스템 Git의 접근 불가 전역 excludes 경로 때문에 커밋 명령에서 프로젝트의 빈 excludes 파일을 일회성으로 지정했다. 마지막 커밋 ID와 검증 결과는 최종 보고를 참조한다.

재현 검증용으로 commit `ea38ce9`를 `git archive`하여 `outputs/clean-checkout`에 풀고, 새로운 별도 `.venv`에서 setup과 pytest 및 데이터 재구성을 수행했다. 기존 `.venv`나 legacy 프로젝트를 복사하지 않았다. checkout의 모델 파일은 Git에 없으므로 정상적으로 모델 미준비 상태를 표시한다. 이 프로젝트의 공식 초기 가중치 hash를 확인해 smoke 학습을 재현하며, 본 학습 전체 재수행은 이미 수행한 30 epoch run과 구분해 기록한다.

패키징 전 `git ls-files`로 `.env`, `.venv`, 원본/파생 이미지, `.pt`, `.db`, 증거가 추적되지 않는지 검사한다. `app/`에는 legacy 경로나 PYTHONPATH 조작이 없다. 데이터 접근 경로는 audit와 manifest에만 있고 실행 모델은 이 프로젝트 ROOT 안 registry 파일만 허용한다.
