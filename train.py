import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import timm
import torch
import torch.nn as nn
from einops import rearrange
torch.set_float32_matmul_precision("high")

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from rhp import RHP
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mvit import mvit_small_patch16, mvit_base_patch16
from utils.dataset import ImageFolderLMDB

# seed
# seed_everything(42, workers=True)

# arguments
parser = argparse.ArgumentParser(description='rhp-lightning-test')

parser.add_argument('--db_path', type=str, required = True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gpus', type=int, default=2)
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--model', type=str, default='small')
parser.add_argument('--save_name', type=str, default='test')

args = parser.parse_args()

# constants
TEMP       = 0.2
IMAGE_SIZE = 224
NUM_WORKERS = 12
EVERY_N_EPOCHS = 1

LR         = args.lr
BATCH_SIZE = args.bs
DEVICES    = args.gpus
MASK_RATIO = args.mask_ratio

# model
base_model = mvit_small_patch16(args.ckpt) if args.model == 'small' else  mvit_base_patch16(args.ckpt)

# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = RHP(net, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images):
        return self.learner(
            images,
        )

    def ctr(self, q, k):
        N = q.shape[0]
        logits = torch.mm(q, k.t()) / TEMP
        labels = (torch.arange(N, dtype=torch.long) + N * self.trainer.global_rank).to(q.device)
        loss = self.loss(logits, labels)
        return loss * 2 * TEMP

    def training_step(self, images, _):
        online_pred, target_proj = self.forward(images)
        online_pred_one, online_pred_two = online_pred
        target_proj_one, target_proj_two = target_proj

        if self.trainer.world_size != 1:
            target_proj_one = self.trainer.strategy.all_gather(target_proj_one)
            target_proj_two = self.trainer.strategy.all_gather(target_proj_two)
            target_proj_one = rearrange(target_proj_one, 'N B D -> (N B) D')
            target_proj_two = rearrange(target_proj_two, 'N B D -> (N B) D')

        loss = self.ctr(online_pred_one, target_proj_two) + self.ctr(online_pred_two, target_proj_one)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# main

if __name__ == '__main__':
    ds = ImageFolderLMDB(args.db_path, IMAGE_SIZE)
    train_loader = DataLoader(
        ds, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, 
        pin_memory = True,
        shuffle = True,
        drop_last = True
    )

    model = SelfSupervisedLearner(
        base_model,
        mask_ratio = MASK_RATIO,
        image_size = IMAGE_SIZE,
        hidden_layer = -1, # or hidden_layer = 'norm',
        projection_size = 256,
        projection_hidden_size = 2048,
        moving_average_decay = 0.99
    )

    logger = TensorBoardLogger(
        save_dir = 'lightning_logs',
        name = 'logs_{}'.format(save_name)
    )
    
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs = EVERY_N_EPOCHS,
        save_top_k = -1,
    )

    trainer = pl.Trainer(
        devices = DEVICES,
        max_epochs = args.epoch,
        accumulate_grad_batches = 1,
        sync_batchnorm = True,
        logger = logger,
        callbacks = [checkpoint_callback],
        # deterministic = True
    )

    trainer.fit(model, train_loader)
