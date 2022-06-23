import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from config import *

plt.style.use('ggplot')



class Averager():
    " Tính toán loss trung bình cho mỗi epoch"
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    """
    Hàm định nghĩa cách load batch data
    """
    return tuple(zip(*batch))



def get_train_transform():
    " Hàm định nghĩa các phương pháp Data Augmentation cho bộ dữ liệu huấn luyện"
    return A.Compose([
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, p=0.5
        ),
        A.ColorJitter(p=0.5),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })



def get_valid_transform():
    " Hàm định nghĩa các phương pháp Data Augmentation cho bộ dữ liệu kiểm định"
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

