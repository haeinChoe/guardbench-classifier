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
uv run --env-file .env guardbench-vllm up           # Qwen 하나만: GPU 여유 메모리 확인 → 기동 → health 대기 → status
uv run --env-file .env guardbench-vllm status
uv run --env-file .env pytest -m integration        # 실제 서버로 greedy 반복 일치 확인
uv run --env-file .env guardbench-vllm down         # 전부 중지, GPU 반환
uv run --env-file .env guardbench-vllm up encoder   # encoder 하나만 (아래 참고)
```

- 서버는 **하나씩** 띄웁니다. `up`과 `wait`는 `qwen`(기본) 또는 `encoder`를 인자로 받고, `status`는 인자가 없으면 설정된 서버를 모두 보여줍니다. 다른 작업과 GPU를 공유하는 상태(약 3.5 GB 사용 중, 16 GB GPU)에서는 Qwen(비율 0.65)과 encoder(약 1.0~1.4 GB)를 함께 띄울 메모리가 부족하고, 이전 실험도 둘을 동시에 띄우지 않았습니다. 단계마다 필요한 서버만 띄우고 `down`으로 내린 뒤 다음 서버를 띄웁니다.
- `up <서버>`는 `서버 메모리 비율 × GPU 전체 메모리 + VLLM_GPU_MEMORY_MARGIN_MIB`(CUDA 컨텍스트 몫)가 `nvidia-smi`의 여유 메모리보다 크면 아무것도 띄우지 않고 실패합니다. 다른 서버가 이미 떠 있으면 그만큼 줄어든 여유 메모리로 검사합니다. 이미 healthy하면 다시 띄우지 않습니다. 실패하면 종료 코드가 0이 아닙니다. 대기 시간은 `--timeout`(기본 600초)입니다.
- 기동 중 컨테이너가 종료되면 timeout을 기다리지 않고 바로 실패합니다. 컨테이너를 지우기 전에 로그를 `runs/vllm-logs/<서버>-<UTC 시각>.log`에 저장하고 오류 메시지에 경로를 적습니다. 이번 `up`이 띄운 컨테이너만 지웁니다.
- `up`이 확인한 포트·모델·revision·max-model-len·메모리 비율·encoder dtype은 `docker compose`에 프로세스 환경변수로 넘어가 `.env`보다 우선합니다. 확인한 값과 실제 기동 값은 항상 같습니다. `--env-file .env` 없이 실행하면 `.env`는 Python에 읽히지 않으므로 기본값(Qwen 0.65, 포트 8000)이 쓰입니다.
- `status`는 서버가 응답하면 실행 중인 컨테이너의 `--dtype`도 보여줍니다. `down`은 두 서버를 모두 내립니다.
- GPU는 다른 작업과 공유합니다. 메모리 확인은 첫 번째 GPU 기준입니다.
- 모델은 호스트의 Hugging Face 캐시를 마운트해 재사용합니다.

- 환경변수는 `.env`에만 두고 커밋하지 않습니다. 변수 목록과 설명은 `.env.example`에 있습니다.
- 코드에서는 `guardbench_classifier.config.Settings`로만 환경변수를 읽습니다.
- 데이터셋, 모델, 실행 결과, 체크포인트는 커밋하지 않습니다.

#### encoder 서버 (선택)

`.env`의 `VLLM_ENCODER_CHECKPOINT_DIR`에 HF 형식(`config.json` + safetensors) 체크포인트 디렉터리를 지정하고 `guardbench-vllm up encoder`로 띄웁니다(compose profile `encoder`, 서비스 `vllm-encoder`). 디렉터리는 컨테이너에 읽기 전용으로 마운트되고 다운로드는 하지 않습니다. 비워 두면 encoder는 `not configured`이고 `up encoder`는 아무것도 띄우지 않고 실패합니다. 경로가 설정됐는데 디렉터리가 없을 때도 같습니다. Qwen만 쓰는 실험에는 영향이 없습니다. 분류 요청은 `/classify`로 보냅니다.

- 정밀도는 `VLLM_ENCODER_DTYPE`(`float32`(기본), `bfloat16`, `float16`)로 고정합니다. 지정하지 않으면 vLLM이 fp16을 고르는데, 정밀도가 다르면 경계에 있는 행의 예측이 바뀔 수 있습니다. 로컬 확인(705건)에서는 vLLM fp32와 transformers fp32가 모든 예측에서 같았고, 같은 체크포인트를 bf16 autocast로 돌린 파일럿과는 4건이 달랐습니다. encoder 결과는 사용한 dtype과 함께 보고하세요. 값은 `Settings.public_snapshot()`으로 실행 manifest에 기록됩니다.
- 포트와 메모리 비율은 `VLLM_ENCODER_PORT`, `VLLM_ENCODER_GPU_MEMORY_UTILIZATION`입니다. 서비스 정의는 `docker compose --profile encoder config`로 확인합니다.

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
