import argparse
from glob import glob
import os
from bunch import Bunch  # Keep this, it might be used elsewhere
from loguru import logger
from ruamel.yaml import safe_load
import torch

import networks as models
from datasets.fives import FIVES
from datasets.utils import fives_loader
from datasets.utils import load_subgroup_images
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch



def main(CFG, disease):
    seed_torch()

    logger.info(f'RUNNING with the following configurations!!! \n \n {CFG} \n\n')

    # Use relative path for FIVES
    train_x, train_y, _, _  = load_subgroup_images(disease, root="../dataset/FIVES")
    dataset = FIVES(CFG=CFG, images_path=train_x, mask_paths=train_y)
    train_loader, val_loader = fives_loader(Dataset=dataset, CFG=CFG)

    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG)

    logger.info(f'\n{model} with {loss}\n')
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
    # testing on a disease means, train, val on 3 other categories
    parser.add_argument("-d", "--disease", help="Disease to test on")
    parser.add_argument("-cf", "--config", help="Configuration file to load", required=True)
    parser.add_argument("-m", "--model", help="Model to train", type=str, required=True) # Added model argument
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as file:
        CFG = safe_load(file) # Load as a standard dictionary

    # adjust the save directory to store checkpoints for each disease.
    CFG['save_dir'] = f"{CFG['save_dir']}{args.disease}/"  #Keep this, it's for organization
    CFG['model']['type'] = args.model  # Override model type - CORRECTED
    main(CFG, args.disease)

    #python -u src/train_ood.py --config configs/ood.yaml --disease N
