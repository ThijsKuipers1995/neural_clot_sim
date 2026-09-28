import importlib


def load_module(cfg):
    if "_target_" not in cfg:
        raise ValueError("Config must have a '_target_' key")

    module_path, attr_name = cfg["_target_"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, attr_name)

    return cls
