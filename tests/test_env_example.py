from pathlib import Path

# Variables whose empty value differs from "unset" (empty CUDA_VISIBLE_DEVICES hides all GPUs,
# empty HF_HOME relocates the cache). They must stay commented out in .env.example.
EMPTY_IS_HARMFUL = {"CUDA_VISIBLE_DEVICES", "HF_HOME"}


def test_env_example_does_not_set_harmful_empty_values() -> None:
    example = Path(__file__).resolve().parents[1] / ".env.example"
    assigned = {line.split("=", 1)[0].strip() for line in example.read_text().splitlines()
                if "=" in line and not line.lstrip().startswith("#")}
    assert not (assigned & EMPTY_IS_HARMFUL)
