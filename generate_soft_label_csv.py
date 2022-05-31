import sklearn.preprocessing
import pandas as pd

import torch

from scipy.special import softmax
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from CONFIG import CONFIG, init_seeds
from dataset import OrchidDataset, get_transform
from models import BuildModel


if __name__ == '__main__':
    init_seeds(CONFIG.seed)
    MODEL_PATH = [
            "./logs/tf_efficientnet_b1_ns/version_8/checkpoints/fold=0-epoch=38-val_loss=0.6127-val_f1_score=0.8301.ckpt",
            "./logs/tf_efficientnet_b1_ns/version_8/checkpoints/fold=1-epoch=38-val_loss=0.6716-val_f1_score=0.8166.ckpt",
            "./logs/tf_efficientnet_b1_ns/version_8/checkpoints/fold=2-epoch=31-val_loss=0.7188-val_f1_score=0.8016.ckpt",
            "./logs/tf_efficientnet_b1_ns/version_8/checkpoints/fold=3-epoch=28-val_loss=0.6972-val_f1_score=0.8325.ckpt",
            "./logs/tf_efficientnet_b1_ns/version_8/checkpoints/fold=4-epoch=25-val_loss=0.6903-val_f1_score=0.8136.ckpt"
        ]
    df = pd.read_csv("./labels.csv")
    df["file_path"] = df["filename"].apply(lambda image: CONFIG.TRAIN_DIR + image)

    model = BuildModel(model_name=CONFIG.model_name, pretrained=CONFIG.pretrained)

    skf = StratifiedKFold(n_splits=CONFIG.n_fold, shuffle=True, random_state=CONFIG.seed)
    train_data_cp = []
    for fold_i, (train_index, val_index) in enumerate(skf.split(df['filename'], df["category"])):
        train_data = df.iloc[train_index, :].reset_index(drop=True)
        val_data = df.iloc[val_index, :].reset_index(drop=True)
        val_data_cp = val_data.copy()

        encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        encoder_val_data = pd.DataFrame(encoder.fit_transform(df[['category']]).toarray())
        val_data_cp = val_data_cp.join(encoder_val_data).drop(["category", "file_path"], axis=1)

        val_dataset = OrchidDataset(
            val_data, get_transform('valid')
        )

        val_dataloader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=False, pin_memory=True,
                                  num_workers=0)
        submission = []
        model.load_state_dict(torch.load(MODEL_PATH[fold_i])["state_dict"])
        model.to("cuda")
        model.eval()

        for i in range(1):
            val_preds = []
            labels = []
            with torch.no_grad():
                for batch in val_dataloader:
                    image = batch['image'].cuda()
                    label = batch['target']

                    output = model(image)
                    val_preds.append(output)
                    labels.append(label)

                labels = torch.cat(labels)
                val_preds = torch.cat(val_preds)
                submission.append(val_preds.cpu().numpy())

        submission_ensembled = 0
        for sub in submission:
            submission_ensembled += softmax(sub, axis=1) / len(submission)
        val_data_cp.iloc[:, 1:] = submission_ensembled
        train_data_cp.append(val_data_cp)
    df = df.drop(["file_path"], axis=1)
    soft_labels = df[["filename"]].merge(pd.concat(train_data_cp), how="left", on="filename")
    soft_labels.to_csv("./submission_test/soft_label.csv", index=False)
