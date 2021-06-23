import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import json
import os
import joblib
from .data.generate_ps_v2 import generate_ps

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config")
def main(cfg) -> None:
    dbinfo = cfg.db
    print(dbinfo)
    processing_dir = cfg.processing_dir
    print(cfg.output_path)
    os.makedirs(processing_dir, exist_ok=True)
    
    candidates = generate_ps(dbinfo, processing_dir)
    joblib.dump(candidates, cfg.output_path)


if __name__ == "__main__":
    try:
        main(None)
    except Exception as e:
        raise e


