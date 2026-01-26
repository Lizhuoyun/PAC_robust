import json

from src.logging.logger import ExperimentLogger


def test_experiment_logger_jsonl(tmp_path):
    cfg = {"logging": {"backend": "jsonl"}}
    path = tmp_path / "metrics.jsonl"
    logger = ExperimentLogger(cfg, str(path))
    logger.log({"step": 1, "loss": 0.5}, step=1)
    logger.finish()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["step"] == 1


def test_experiment_logger_wandb_fallback(tmp_path):
    cfg = {"logging": {"backend": "wandb", "wandb": {"mode": "offline"}}}
    path = tmp_path / "metrics.jsonl"
    logger = ExperimentLogger(cfg, str(path))
    logger.log({"step": 2, "loss": 0.2}, step=2)
    logger.finish()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
