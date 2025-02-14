import argparse
from glob import glob
import os
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
import torch
from torch.utils.data import DataLoader

import networks as models
from datasets.fives import FIVES
from datasets.chase import CHASEDBDataset
from datasets.drive import DRIVEDataset
from datasets.utils import fives_loader
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch



def main(CFG):
    seed_torch()

    logger.info(f'RUNNING with the following configurations!!! \n \n {CFG} \n\n')

    if CFG['dataset']['type'] == 'FIVES':
        # Load FIVES dataset in training mode. fives_loader will split.
        dataset = FIVES(CFG=CFG, mode='train')
        train_loader, val_loader = fives_loader(Dataset=dataset, CFG=CFG)

    elif CFG['dataset']['type'] == 'CHASEDB':
        # Use relative paths for CHASEDB
        train_path = os.path.join("..", "datasets", "CHASE_DB1", "train")
        valid_path = os.path.join("..", "datasets", "CHASE_DB1", "test")
        train_dataset = CHASEDBDataset(CFG, os.path.join(train_path, "images", "*"), os.path.join(train_path, "labels", "*"))
        val_dataset = CHASEDBDataset(CFG, os.path.join(valid_path, "images", "*"), os.path.join(valid_path, "labels", "*"))

        train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], pin_memory=True, drop_last=True, num_workers=CFG['num_workers'])
        val_loader  = DataLoader(val_dataset, batch_size=CFG['batch_size'], pin_memory=True, drop_last=True, num_workers=CFG['num_workers'])


    elif CFG['dataset']['type'] == 'DRIVE':
         # Use relative paths for DRIVE
        train_path = os.path.join("..", "datasets", "DRIVE", "train")
        valid_path = os.path.join("..", "datasets", "DRIVE", "test")
        train_dataset = DRIVEDataset(train_path)
        val_dataset = DRIVEDataset(valid_path)

        train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'],
                                pin_memory=False, drop_last=True, num_workers=CFG['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'],
                                pin_memory=False, drop_last=True, num_workers=CFG['num_workers'])
    else:
        raise NotImplementedError("Dataset type should be either DRIVE | FIVES | CHASEDB ")



    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader,
        device = device
    )

    trainer.train()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", help="Configuration file to load", required=True)
    parser.add_argument("-m", "--model", help="Model to train", type=str, required=True) # Added model argument
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as file:
        CFG = safe_load(file) # Load as a standard dictionary

    CFG['model']['type'] = args.model  # Override model type

    main(CFG)
