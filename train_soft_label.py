import gc
import os

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import torch
from CONFIG import CONFIG, init_seeds
from dataset import OrchidDataset, get_transform
from models import BuildModel, Lit


if __name__ == '__main__':
    init_seeds(CONFIG.SEED)
    torch.cuda.empty_cache()  # 清除顯存

    all_labels = pd.read_csv("./labels.csv")
    print("Data number：" + str(len(all_labels)))
    soft_labels_filename = "./" + CONFIG.soft_labels_filename
    soft_labels = pd.read_csv(soft_labels_filename)

    unique_category = list(all_labels["category"].unique())
    print("Num classes：" + str(len(unique_category)))
    CONFIG.num_classes = unique_category

    logger = CSVLogger(save_dir='logs_soft_label/', name=CONFIG.model_name)
    logger.log_hyperparams(CONFIG.__dict__)

    all_labels["file_path"] = all_labels["filename"].apply(lambda image: CONFIG.TRAIN_DIR + image)

    skf = StratifiedKFold(n_splits=CONFIG.n_fold, shuffle=True, random_state=CONFIG.SEED)
    for fold_i, (train_idx, valid_idx) in enumerate(skf.split(all_labels['filename'], all_labels["category"])):
        df_train = all_labels.iloc[train_idx, :].reset_index(drop=True)
        df_valid = all_labels.iloc[valid_idx, :].reset_index(drop=True)
        df_train_soft_label = soft_labels.iloc[train_idx, :].reset_index(drop=True)
        df_valid_soft_label = soft_labels.iloc[valid_idx, :].reset_index(drop=True)

        train_dataset = OrchidDataset(df_train, get_transform('train'), soft_labels=df_train_soft_label)
        valid_dataset = OrchidDataset(df_valid, get_transform('valid'), soft_labels=df_valid_soft_label)

        train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.batch_size, shuffle=False, pin_memory=True,
                                  num_workers=0)

        CONFIG.steps_per_epoch = len(train_loader)

        model = BuildModel(model_name=CONFIG.model_name, pretrained=CONFIG.pretrained)

        lit_model = Lit(model.model)

        checkpoint_callback = ModelCheckpoint(monitor='val_f1_score',
                                              save_top_k=3,
                                              save_last=True,
                                              # save_weights_only=True,
                                              filename=f"fold={fold_i}" + '-{epoch:02d}-{val_loss:.4f}-{val_f1_score:.4f}-{val_acc:.4f}',
                                              verbose=False,
                                              mode='max')

        trainer = Trainer(
            max_epochs=CONFIG.num_epochs,
            gpus=[0],
            accumulate_grad_batches=CONFIG.accum,
            # precision=CONFIG.precision,
            callbacks=[checkpoint_callback],
            logger=logger,
            weights_summary='top',
        )

        trainer.fit(model=lit_model, train_dataloader=train_loader, val_dataloaders=valid_loader)

        del model
        del lit_model
        gc.collect()
        torch.cuda.empty_cache()
