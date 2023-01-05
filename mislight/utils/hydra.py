from typing import Any, Callable, Dict, List

import hydra
from omegaconf import DictConfig

def instantiate_list(cfg: DictConfig) -> List:
    """Instantiates multiple targets from config."""
    targets= []

    for _, c in cfg.items():
        if isinstance(c, DictConfig) and "_target_" in c:
            targets.append(hydra.utils.instantiate(c))

    return targets