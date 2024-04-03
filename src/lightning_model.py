import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoTokenizer


class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.img_encoder = CLIPVisionModelWithProjection.from_pretrained(config.data.model_name)
        self.location_identifier = nn.Sequential(
            nn.LayerNorm(self.img_encoder.config.projection_dim),
            nn.Linear(self.img_encoder.config.projection_dim, self.img_encoder.config.projection_dim*2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.img_encoder.config.projection_dim*2, self.img_encoder.config.projection_dim//2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.img_encoder.config.projection_dim//2, 2)
        )

        self.optimizer = config.train.optimizer
        self.vit_lr = config.train.vit_lr
        self.head_lr = config.train.head_lr
        self.weight_decay = config.train.weight_decay
    
    def forward(self, x):
        img_embeds = self.img_encoder(x).image_embeds
        coords = self.location_identifier(img_embeds)
        return coords

    def configure_optimizers(self):
        parameters = [
            {'params': self.img_encoder.parameters(), 'lr': self.vit_lr},
            {'params': self.location_identifier.parameters(), 'lr': self.head_lr}
        ]

        if self.optimizer == 'Adam':
            opt = torch.optim.Adam(parameters, weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            opt = torch.optim.AdamW(parameters, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            patience=1.0,
            factor=0.8,
        )
        return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'val/loss'}
    
    
    def training_step(self, batch, batch_idx):
        img, coord, _, _ = batch
        pred_coord = self(img)

        loss_coord = F.mse_loss(pred_coord, coord)
        data_dict = {"loss": loss_coord}
        log_dict = {"train/loss": data_dict["loss"]}
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict
    
    def validation_step(self, batch, batch_idx):
        img, coord, _, _ = batch
        pred_coord = self(img)

        loss_coord = F.mse_loss(pred_coord, coord)
        data_dict = {"loss": loss_coord}
        log_dict = {"val/loss": data_dict["loss"]}
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def test_step(self, batch, batch_idx):
        img, coord, _, _ = batch
        pred_coord = self(img)

        loss_coord = F.mse_loss(pred_coord, coord)
        data_dict = {"loss": loss_coord}
        log_dict = {"test/loss": data_dict["loss"]}
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict


class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)
