from pathlib import Path
from datetime import datetime

import yaml


def create_output_dir(
    experiment_path: str,
    base_dir: str = "outputs",
    cfg: dict | None = None,
    tag: str | None = None,
) -> Path:
    tag = "run" if tag is None else tag

    experiment_name = Path(experiment_path).stem

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = Path(base_dir) / experiment_name / (f"{tag}-" + timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    # save config
    if cfg is not None:
        cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

        with open(output_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

    return output_dir, experiment_name
