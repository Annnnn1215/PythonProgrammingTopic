import torch
from pytorch_lightning import seed_everything
import numpy as np


class CONFIG:
    TRAIN_DIR = "./training/"
    SEED = 42
    model_name = 'tf_efficientnet_b0_ns'  # tf_efficientnet_b1_ns
    soft_labels_filename = ""
    pretrained = True
    img_size = 224
    num_classes = int(219)
    lr = 1e-4
    max_lr = 1e-3
    pct_start = 0.2
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    num_epochs = 40
    batch_size = 16
    accum = 1
    precision = 16
    n_fold = 5
    alpha = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_seeds(seed=1215):
    """
        設定隨機種子
    """
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False