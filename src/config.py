import torch


CLASSES = [
    'person', 'sofa', 'bottle', 'tvmonitor', 'cat', 'pottedplant',
    'horse', 'car', 'dog', 'train', 'bicycle', 'aeroplane',
    'diningtable', 'motorbike', 'chair', 'cow', 'bus', 'bird', 'boat',
    'sheep'
]

OUT_DIR = '/kaggle/working/'

EPOCHS = 200
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 32
RESIZE_TO = 512


TRAIN_IMAGES = '/kaggle/input/pascal-voc-2012/VOC2012/JPEGImages/'
TRAIN_LABELS = '/kaggle/input/pascal-voc-2012/VOC2012/Annotations/'
