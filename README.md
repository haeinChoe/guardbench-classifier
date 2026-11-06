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
