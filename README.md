# GuardBench Classifier

GuardBench response behavior classifier(`prompt`, `response` → `COMPLY | REFUSE`)의 실험·평가 코드입니다. 작업 규칙은 [AGENTS.md](AGENTS.md)를 따릅니다.

## 환경 설정

필요: [uv](https://docs.astral.sh/uv/), NVIDIA GPU와 CUDA 12.8을 지원하는 드라이버.

```bash
uv sync                      # Python 환경 (torch 2.9.1 cu128, transformers 4.57)
cp .env.example .env         # 환경변수 설정 후 값 채우기
uv run --env-file .env guardbench-env   # 버전, GPU, vLLM 서버, 설정 확인
uv run pytest
```

### vLLM 서버 (Docker)

vLLM 0.16.0은 Docker 공식 이미지로 실행합니다. 호스트(Rocky Linux 8, glibc 2.28)에는 vLLM 휠이 설치되지 않기 때문입니다. 필요: Docker와 NVIDIA container runtime.

```bash
docker compose up -d vllm                    # .env의 VLLM_* 값으로 시작
uv run --env-file .env guardbench-vllm wait  # 준비될 때까지 대기
uv run --env-file .env guardbench-vllm status
uv run --env-file .env pytest -m integration # 실제 서버로 greedy 반복 일치 확인
docker compose down                          # 중지, GPU 반환
```

- GPU는 다른 작업과 공유합니다. `VLLM_GPU_MEMORY_UTILIZATION`은 남은 GPU 메모리보다 작게 둡니다.
- 모델은 호스트의 Hugging Face 캐시를 마운트해 재사용합니다.

- 환경변수는 `.env`에만 두고 커밋하지 않습니다. 변수 목록과 설명은 `.env.example`에 있습니다.
- 코드에서는 `guardbench_classifier.config.Settings`로만 환경변수를 읽습니다.
- 데이터셋, 모델, 실행 결과, 체크포인트는 커밋하지 않습니다.

## 실험 실행 (runner)

`guardbench_classifier.runner.run()`은 `(id, payload)` 목록과 처리 함수를 받아 `runs/<run_id>/`에 기록합니다. 실험 내용(데이터, 프롬프트, 파싱, 지표)은 호출하는 쪽의 몫입니다.

```python
from guardbench_classifier.runner import run

summary = run("exp-001", items, process, config={"prompt": "v1", "gpu_shared": True}, concurrency=1)
```

| 파일 (`runs/<run_id>/`) | 내용 |
| --- | --- |
| `worker.lock` | 실행 중 `flock`. 이미 잡혀 있으면 즉시 실패 |
| `manifest.json` | 첫 실행 조건: git commit·dirty, 설정, 버전, 동시성, 입력 해시 |
| `records.jsonl` | 완료한 행(추가 전용). 재개하면 여기에 없는 id만 처리 |
| `errors.jsonl` | 처리 함수가 예외를 낸 행. 완료로 보지 않고 재개 때 다시 시도 |
| `status.json` | state(running/complete/failed), total, done, error_rows, pid, 시각 |
| `git.diff` | `allow_dirty=True`로 dirty 트리에서 시작했을 때의 `git diff HEAD` |
| `git.status` | 같은 경우의 `git status --porcelain`. untracked 파일 목록(내용은 저장하지 않음) |

- 같은 `run_id`로 다시 호출하면 빠진 행만 처리합니다. config, 동시성, 입력, 코드 상태(commit, dirty diff)가 달라지면 실패하므로 새 `run_id`를 씁니다.
- 커밋하지 않은 변경이 있으면 거부합니다(`.gitignore` 대상 제외). 의도한 경우만 `allow_dirty=True`를 씁니다.
- 처리 함수가 돌려준 invalid(파싱 실패 등)는 그대로 기록하고, 예외만 `errors.jsonl`로 갑니다.
- 장시간 작업은 분리된 프로세스로 실행하고, 진행은 LLM 없이 상태 파일로 확인합니다.

```bash
setsid nohup uv run --env-file .env python -m my_experiment > runs/exp-001.log 2>&1 &
uv run guardbench-run status runs/exp-001   # status.json과 pid 생존 여부
```
