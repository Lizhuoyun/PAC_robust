import json
import os
from typing import Any, Dict, Optional


class JsonlBackend:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def finish(self) -> None:
        return None


class WandbBackend:
    def __init__(self, cfg: Dict[str, Any]):
        try:
            import wandb
        except Exception:
            self.wandb = None
            self.enabled = False
            return
        self.wandb = wandb
        self.enabled = True
        wandb_cfg = cfg.get("logging", {}).get("wandb", {})
        mode = wandb_cfg.get("mode", "offline")
        if mode == "disabled":
            self.enabled = False
            return
        os.environ.setdefault("WANDB_MODE", mode)
        self.wandb.init(
            project=wandb_cfg.get("project", "icml_wcr_spectral"),
            entity=wandb_cfg.get("entity"),
            group=wandb_cfg.get("group"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
            dir=wandb_cfg.get("dir", "./wandb"),
            config=cfg,
            mode=mode,
            resume=wandb_cfg.get("resume", "allow"),
            save_code=wandb_cfg.get("save_code", True),
        )
        if wandb_cfg.get("watch_model", False):
            self.wandb.watch(None, log="all")

    def log(self, record: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled:
            self.wandb.log(record, step=step)

    def finish(self) -> None:
        if self.enabled:
            self.wandb.finish()


class ExperimentLogger:
    def __init__(self, cfg: Dict[str, Any], jsonl_path: str):
        backend = cfg.get("logging", {}).get("backend", "jsonl")
        self.jsonl = JsonlBackend(jsonl_path)
        self.wandb = None
        if backend in ("wandb", "both"):
            self.wandb = WandbBackend(cfg)

    def log(self, record: Dict[str, Any], step: Optional[int] = None) -> None:
        self.jsonl.log(record)
        if self.wandb is not None:
            self.wandb.log(record, step=step)

    def log_artifact(self, path: str, name: str) -> None:
        if self.wandb is None or not self.wandb.enabled:
            return
        artifact = self.wandb.wandb.Artifact(name, type="artifact")
        artifact.add_file(path)
        self.wandb.wandb.log_artifact(artifact)

    def finish(self) -> None:
        self.jsonl.finish()
        if self.wandb is not None:
            self.wandb.finish()
