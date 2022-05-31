import pandas as pd
import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from CONFIG import CONFIG, init_seeds
from dataset import OrchidDataset, get_transform
from models import BuildModel


init_seeds(CONFIG.seed)

MODEL_PATH = [
        "./logs_soft_label/tf_efficientnet_b1_ns/version_17/checkpoints/"
        "fold=0-epoch=36-val_loss=0.6998-val_f1_score=0.8713.ckpt",
        "./logs_soft_label/tf_efficientnet_b1_ns/version_17/checkpoints/"
        "fold=1-epoch=29-val_loss=0.7227-val_f1_score=0.8340.ckpt",
        "./logs_soft_label/tf_efficientnet_b1_ns/version_19/checkpoints/"
        "fold=2-epoch=36-val_loss=0.7590-val_f1_score=0.8509.ckpt",
        "./logs_soft_label/tf_efficientnet_b1_ns/version_19/checkpoints/"
        "fold=3-epoch=32-val_loss=0.7646-val_f1_score=0.8399.ckpt",
        "./logs_soft_label/tf_efficientnet_b1_ns/version_19/checkpoints/"
        "fold=4-epoch=38-val_loss=0.7816-val_f1_score=0.8602.ckpt"
    ]

# 預測資料的csv檔
test_data = pd.read_csv("./test_labels(2190).csv")
test_data["file_path"] = test_data["filename"].apply(lambda image: CONFIG.TRAIN_DIR + image)

test_dataset = OrchidDataset(test_data, get_transform('train'))
test_loader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=0)

model = BuildModel(model_name=CONFIG.model_name, pretrained=CONFIG.pretrained)

submission = []

for path in CONFIG.MODEL_PATH:
    model.load_state_dict(torch.load(path)["state_dict"])
    model.to("cuda")
    model.eval()
    for i in range(8):
        test_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                image = batch['image'].cuda()
                output = model(image)
                test_preds.append(output)

            test_preds = torch.cat(test_preds)
            submission.append(test_preds.cpu().numpy())

submission_ensembled = 0
for sub in submission:
    submission_ensembled += softmax(sub, axis=1) / len(submission)
for i in range(len(test_data)):
    test_data.iloc[i, 1] = np.argmax(submission_ensembled[i])
test_data = test_data.drop("file_path", axis=1)
test_data.to_csv("./submission/submission.csv", index=False)