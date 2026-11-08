# GuardBench Classifier 에이전트 운영 규칙

이 저장소는 GuardBench 판정기(classifier)의 실험·평가 코드를 둔다. 저장소의 상시 규칙은 이 문서에, 개별 요구사항은 GitHub Issue에 둔다. 개별 요구사항을 이 문서에 복제하지 않는다.

## 판단 기준

1. 현재 Issue의 승인된 요구사항과 사용자의 명시적 지시
2. `APPROVED` 상태의 저장소 문서와 결정 기록
3. 테스트와 현재 코드 계약
4. `DRAFT` 상태의 문서

- 요청을 충족하는 최소 변경만 수행한다.
- `DRAFT`나 미결정 사항을 확정 요구사항으로 간주하지 않는다. 의존해야 진행할 수 있으면 미결정을 Issue에 기록하고 중단한다.
- `Decision` Issue에서 에이전트가 대안을 임의로 선택하지 않는다. 근거와 선택별 영향을 보고하고 결정을 기다린다.
- 새 의존성, 모델 다운로드, 외부 API 호출, 데이터셋 추가는 사전 확인한다.
- 기존 미커밋 변경을 보존한다. 출처가 불명확한 변경을 되돌리지 않는다.
- 현재 Issue/PR와 무관한 이슈, 다른 worktree, 다른 커밋을 대화와 리뷰에 끌어오지 않는다.

## Issue와 브랜치

- 모든 작업은 Issue에서 시작한다. Issue가 없으면 먼저 Issue 작성을 요청하거나 초안을 제시한다.
- Issue는 `.github/ISSUE_TEMPLATE/`의 Experiment, Engineering Task, Bug, Decision 중 맞는 템플릿으로 작성한다. 필수 항목과 에이전트 작업 범위를 채운다.
- Issue 하나당 별도 worktree와 `agent/{issue-number}-{slug}` 브랜치를 사용한다. 기준은 최신 `origin/dev`다.

```bash
git fetch origin
git worktree add ../guardbench-classifier-issue-{N} -b agent/{N}-{slug} origin/dev
```

- Issue에 명시된 에이전트 작업 범위를 따른다.

| 작업 범위 | 허용 |
| --- | --- |
| 조사와 진단만 허용 | 읽기·검색·보고. 파일 변경 금지 |
| 파일 수정과 검증까지 허용 | 수정과 검증. 커밋 금지 |
| 검증된 로컬 커밋까지 허용 | 수정, 검증, 관련 로컬 커밋 |
| 검증된 로컬 커밋 및 push·PR 생성까지 허용 | 위 범위와 사전 보고 후 Issue 브랜치 push, PR 생성. 병합 금지 |

- 작업 범위가 없거나 모호하면 원격 작업을 추정하지 않는다.
- 어느 범위도 병합, force push, 태그 변경, secret 취급을 허용하지 않는다.

## Git 규칙

- 기본 브랜치는 `dev`다. `main`과 `dev`에는 직접 push하지 않는다. `dev`는 Issue 브랜치의 PR 병합으로만, `main`은 `dev`에서 온 PR 병합으로만 갱신한다.
- PR은 `.github/pull_request_template.md` 구조를 유지해 실제 작업 내용으로 채운다. 관련 Issue는 `Closes #N`(병합 시 GitHub가 자동 종료) 또는 `Refs #N`(연결만)으로 적는다.
- 독립적으로 검토 가능한 변경은 별도 커밋으로 나눈다. 단위 예: 데이터 로딩·분할, 모델 어댑터, 출력 파싱, 지표, 실행 관리(runner), 실험 설정, 보고서, 문서.
- 커밋 전에 `git status`와 staged diff로 Issue 범위와 변경 파일이 일치하는지 확인한다.
- push 전에 `git log --oneline origin/dev..HEAD`, `git diff --stat origin/dev...HEAD`, 전체 diff 범위를 보고한다.
- 검증하지 않은 결과를 완료했다고 표현하지 않는다.

## 브랜치 생애주기

1. Issue 작성 → worktree와 `agent/{N}-{slug}` 브랜치 생성(위 명령)
2. 작업, 리뷰 단위 커밋, push 전 점검 후 Issue 브랜치 push
3. `dev` 대상 PR 생성, 본문에 `Closes #N`
4. 사람이 병합한다. 기본 방식은 merge commit이다. 병합되면 GitHub가 Issue를 닫고 원격 작업 브랜치를 자동 삭제한다(`delete_branch_on_merge`). squash·rebase로 병합하면 아래 head 포함 확인이 실패하므로, 정리하지 말고 보고한다.
5. 병합을 확인한 뒤 로컬을 정리한다. 아래 조건을 모두 확인한 경우에만 삭제하고, 하나라도 아니면 삭제하지 않고 보고한다.
   - PR 상태가 `MERGED`다.
   - 작업 브랜치 head가 `origin/dev`에 포함된다.
   - worktree에 커밋하지 않은 변경이 없다(`git status --short`가 비어 있음).

```bash
git fetch --prune origin
git merge-base --is-ancestor agent/{N}-{slug} origin/dev   # 포함 확인 (exit 0)
git -C ../guardbench-classifier-issue-{N} status --short    # 비어 있어야 함
git worktree remove ../guardbench-classifier-issue-{N}      # .venv 등 무시 파일만 남았으면 --force
git branch -d agent/{N}-{slug}
```

6. `main`은 `dev`에서 온 PR 병합으로만 갱신한다.

## 코드 구조 원칙

책임을 섞지 않는다. 실험 하나가 한 파일에서 데이터 검증부터 보고서까지 모두 처리하지 않게 한다.

- `data`: 데이터셋 로딩, 해시 검증, 분할
- `models`: 생성 모델 호출 어댑터, encoder 학습·추론
- `parsing`: 출력 계약 파싱. 테스트를 반드시 둔다.
- `metrics`: 지표, paired 비교, 슬라이스
- `runner`: 실행 관리(lock, 추가 전용 기록, 빠진 행만 재개, 상태). 실험 내용은 모른다.
- `experiments`: 실험별 설정과 사전 동결한 선택 규칙
- `report`: 결과에서 보고서 생성

공통 기능은 한 곳에만 구현하고 실험별로 복사하지 않는다.

## 실험 무결성

- 라벨이 사람 검증 gold가 아니면 결과를 "정확도"가 아니라 "라벨 일치도"로 보고한다.
- 분할은 같은 질문(group)이 여러 분할에 걸치지 않게 한다. 학습 행을 평가에 섞지 않는다.
- 모델·epoch·threshold 선택 규칙은 test를 보기 전에 기록하고 동결한다. test는 선택 후 한 번만 평가한다.
- 파싱 실패, cap 도달, 최종 답 없음은 invalid로 기록한다. 다른 라벨로 대체하지 않는다.
- 결과를 보고 프롬프트, cap, 라벨을 고치지 않는다. 바꿔야 하면 새 실험으로 분리한다.
- 정답 라벨, 원 annotation, ID, 조건 이름을 모델 입력에 넣지 않는다.
- 지연·비용 수치는 측정 조건(GPU 공유 여부, 배치 크기, 엔진)과 함께 보고한다. 조건이 다른 수치를 직접 비교하지 않는다.
- 모델 reasoning 원문은 로컬에만 두고 보고서나 대화에 그대로 옮기지 않는다.

## 데이터와 산출물

- 외부 데이터셋 원본, 모델 출력, 체크포인트, 대용량 산출물은 커밋하지 않는다. 출처, 버전(revision), 해시, 라이선스와 내려받기 방법만 기록한다.
- 평가 전용으로 지정된 데이터는 학습이나 튜닝에 쓰지 않는다.
- 비밀 정보, 자격 증명, 개인정보를 커밋하지 않는다.

## 로컬 실행 환경

- GPU는 다른 작업과 공유한다. 다른 사람의 GPU 작업을 중단하지 않는다.
- OOM은 배치 축소 같은 제한된 기술적 재시도로만 처리한다.
- 장시간 작업은 분리된 프로세스로 실행하고, 진행 상태를 LLM 반복 조회가 아니라 상태 파일과 비LLM 감시 스크립트로 확인한다.

## 중단 조건

- Issue 내용이나 작업 범위를 확인할 수 없음
- 완료 조건이 충돌하거나 측정할 수 없음
- 미결정 사항에 따라 결과 해석이 달라짐
- 새 의존성, 모델, 데이터셋, 외부 API가 필요함
- secret, 운영 데이터 또는 파괴적인 Git 작업이 필요함
- 범위 밖 원인으로 검증이 실패함

추측하지 말고 현재 상태와 필요한 결정을 보고한다.

## 보고

진행·완료 보고는 한국어로 쓴다. 코드, 식별자, 경로는 원문 그대로 둔다.
