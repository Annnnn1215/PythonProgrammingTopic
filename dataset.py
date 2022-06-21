from torch.utils.data import Dataset
import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from loss_function import one_hot
from CONFIG import CONFIG
import cv2
import torch
import numpy as np


class OrchidDataset(Dataset):
    def __init__(self, df, transform=None, soft_labels=None):
        self.image_path = df['file_path'].values
        self.labels = df["category"].values
        self.transform = transform
        if soft_labels is None:
            print("soft_labels is None")
            self.soft_labels = None
        else:
            self.soft_labels = soft_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape != (480, 640, 3):
            image = image.transpose(1, 0, 2)

        augmented = self.transform(image=image)
        image = augmented['image']

        if self.soft_labels is not None:
            label = torch.add(one_hot(self.labels[idx], 219) * 0.7,
                              torch.tensor((self.soft_labels.iloc[idx, 1:].values * 0.3).astype(np.float32),
                                           dtype=torch.float32))
        else:
            label = one_hot(self.labels[idx], 219)

        return {'image': image, 'target': label}


def get_transform(phase: str):
    if phase == 'train':
        return Compose([
            A.Resize(height=CONFIG.img_size, width=CONFIG.img_size),
            OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1)]),
            OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.5),
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])
    else:
        return Compose([
            A.Resize(height=CONFIG.img_size, width=CONFIG.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def TTA(phase: int, img_size: int):
    if phase == 0:
        return Compose([
            A.Resize(height=img_size, width=img_size),
            OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1)]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])
    if phase == 1:
        return Compose([
            A.Resize(height=img_size, width=img_size),
            OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1)]),
            A.VerticalFlip(p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])
    if phase == 2:
        return Compose([
            A.Resize(height=img_size, width=img_size),
            OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1)]),
            A.HorizontalFlip(p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])
    if phase == 3:
        return Compose([
            A.Resize(height=img_size, width=img_size),
            OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1)]),
            A.VerticalFlip(p=1),
            A.HorizontalFlip(p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])
    if phase == 4:
        return Compose([
            A.Resize(height=img_size, width=img_size),
            OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1)]),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])