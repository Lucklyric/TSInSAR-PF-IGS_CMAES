import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import json
import os

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config")
def main(cfg) -> None:
    pl.seed_everything(cfg.seed)
    print(OmegaConf.to_yaml(cfg))
    data = hydra.utils.instantiate(cfg.db)
    data_loader = DataLoader(data, 32, collate_fn =lambda x:x, num_workers=32, shuffle=False)
    solver = hydra.utils.instantiate(cfg.solver)

    processing_dir = cfg.processing_dir
    os.makedirs(processing_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=processing_dir, name='', version='final')
    trainer = pl.Trainer(checkpoint_callback=False, logger=logger)
    # trainer = pl.Trainer(checkpoint_callback=False, limit_test_batches=2)
    result = trainer.test(solver, test_dataloaders=data_loader, ckpt_path=None)

    log.info('result len {}'.format(len(result)))
    with open(cfg.output_path, 'w') as f:
        json.dump(result, f)
    log.info('result saved to {}'.format(cfg.output_path))

if __name__ == "__main__":
    try:
        main(None)
    except Exception as e:
        raise e
