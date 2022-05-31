import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from lightning import FMix
import pytorch_lightning as pl
from CONFIG import CONFIG
from loss_function import CrossEntropyLossOneHot


class BuildModel(nn.Module):
    def __init__(self, model_name=CONFIG.model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=219, global_pool='avg')

    def forward(self, x):
        x = self.model(x)
        return x


class Lit(pl.LightningModule):
    def __init__(self, model):
        super(Lit, self).__init__()
        self.model = model
        self.model = self.model.to(CONFIG.device)
        self.criterion = CrossEntropyLossOneHot()
        self.fmix = FMix(alpha=CONFIG.alpha, size=(CONFIG.img_size, CONFIG.img_size))
        self.lr = CONFIG.lr

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             epochs=CONFIG.num_epochs,
                                                             steps_per_epoch=CONFIG.steps_per_epoch,
                                                             max_lr=CONFIG.max_lr, pct_start=CONFIG.pct_start,
                                                             div_factor=CONFIG.div_factor,
                                                             final_div_factor=CONFIG.final_div_factor,
                                                             )

        scheduler = {'scheduler': self.scheduler, 'interval': 'step', }

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target']

        image = self.fmix(image)
        output = self.model(image)

        loss = self.fmix.loss(output, target)

        logs = {'train_loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target']
        output = self.model(image)
        loss = self.criterion(output, target)

        return {"val_loss": loss,
                "output": output,
                "target": target
                }

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()

        output_all = torch.cat([output["output"] for output in outputs]).argmax(1).cpu()
        target_all = torch.cat([output["target"] for output in outputs]).argmax(1).cpu()
        f1 = f1_score(output_all, target_all, average='macro')
        val_acc = accuracy_score(output_all, target_all)

        logs = {'val_loss': val_loss_mean, 'val_f1_score': f1, 'val_acc': val_acc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": val_loss_mean, "val_f1_score": f1, 'val_acc': val_acc}