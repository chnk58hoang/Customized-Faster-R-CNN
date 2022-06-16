import torch.cuda

CLASSES = [
    '__background__',
    'person', 'sofa', 'bottle', 'tvmonitor', 'cat', 'pottedplant',
    'horse', 'car', 'dog', 'train', 'bicycle', 'aeroplane',
    'diningtable', 'motorbike', 'chair', 'cow', 'bus', 'bird', 'boat',
    'sheep'
]


OUT_DIR = '/kaggle/working/'

EPOCHS = 200
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 32  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 200  # number of epochs to train for
NUM_WORKERS = 1

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Images and labels direcotry should be relative to train.py
TRAIN_IMAGES = '../input/pascal-voc-2012/VOC2012/JPEGImages'
TRAIN_LABELS = '../input/pascal-voc-2012/VOC2012/JPEGImages'

