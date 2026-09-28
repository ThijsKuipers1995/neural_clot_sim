from omegaconf import OmegaConf
from ml_collections import ConfigDict


def load_config(experiment_path: str, argv=None):
    cfg = OmegaConf.load(experiment_path)
    cfg = ConfigDict(OmegaConf.to_container(cfg, resolve=True))

    if argv is None:
        return cfg

    overrides = [arg for arg in argv if "=" in arg]
    for override in overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], ConfigDict):
                raise ValueError(f"field '{k}' not in config YAML file: {key}.")
            d = d[k]

        if keys[-1] not in d:
            raise ValueError(f"field '{keys[-1]}' not in config YAML file: {key}.")

        # Try to parse numeric/bool values automatically
        if value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        elif value.lower() in {"null", "none"}:
            value = None
        else:
            try:
                value = float(value) if ("." in value) or ("e" in value) else int(value)
            except ValueError:
                pass

        d[keys[-1]] = value

    return cfg
