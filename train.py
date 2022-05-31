import gc
import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch

from dataset import OrchidDataset, get_transform
from models import BuildModel, Lit
from CONFIG import CONFIG, init_seeds


if __name__ == '__main__':
    init_seeds(CONFIG.SEED)
    torch.cuda.empty_cache()  # 清除顯存

    all_labels = pd.read_csv("./labels.csv")
    print("Data number：" + str(len(all_labels)))

    unique_category = list(all_labels["category"].unique())
    print("Num classes：" + str(len(unique_category)))
    CONFIG.num_classes = len(unique_category)

    logger = CSVLogger(save_dir='logs/', name=CONFIG.model_name)
    logger.log_hyperparams(CONFIG.__dict__)

    all_labels["file_path"] = all_labels["filename"].apply(lambda image: CONFIG.TRAIN_DIR + image)


    skf = StratifiedKFold(n_splits=CONFIG.n_fold, shuffle=True, random_state=CONFIG.SEED)
    for fold_i, (train_idx, valid_idx) in enumerate(skf.split(all_labels['filename'], all_labels["category"])):
        df_train = all_labels.iloc[train_idx, :].reset_index(drop=True)
        df_valid = all_labels.iloc[valid_idx, :].reset_index(drop=True)

        train_dataset = OrchidDataset(df_train, get_transform('train'))
        valid_dataset = OrchidDataset(df_valid, get_transform('valid'))

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

        history = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')

        valid_acc = history['val_f1_score'].dropna().reset_index(drop=True)

        plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(valid_acc, color="b", marker="x", label='valid/val_f1_score')
        plt.ylabel('Accuracy', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='lower right', fontsize=18)
        plt.savefig(f'{trainer.logger.log_dir}/fold={fold_i}-acc.png')

        train_loss = history['train_loss'].dropna().reset_index(drop=True)
        valid_loss = history['val_loss'].dropna().reset_index(drop=True)

        plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_loss, color="r", marker="o", label='train/loss')
        plt.plot(valid_loss, color="b", marker="x", label='valid/loss')
        plt.ylabel('Loss', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='upper right', fontsize=18)
        plt.savefig(f'{trainer.logger.log_dir}/fold={fold_i}-loss.png')

        lr = history['lr'].dropna().reset_index(drop=True)

        plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(lr, color="g", marker="o", label='learning rate')
        plt.ylabel('LR', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='upper right', fontsize=18)
        plt.savefig(f'{trainer.logger.log_dir}/fold={fold_i}-lr.png')

        os.remove(os.path.join(f'./logs/{CONFIG.model_name}/version_{logger.version}/metrics.csv'))
        del model
        del lit_model
        gc.collect()
        torch.cuda.empty_cache()
