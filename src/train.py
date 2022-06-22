import argparse
import pandas as pd
import torch
from modifydata import modify
from xml_to_csv import xml_to_csv
from config import *
from sklearn.model_selection import train_test_split
from dataset import CustomDataset
from torch.utils.data import DataLoader, random_split
from model import create_model
from engine import train_one_epoch, evaluate
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, help='number of clusters')

parser.add_argument('--min_dim', type=int, help='min_dimension')
parser.add_argument('--max_dim', type=int, help='max_dimension')
parser.add_argument('--mode', type=str, help='mode')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--finetune', default=False, type=bool)
parser.add_argument('--batchsize', default=32, type=int)

args = parser.parse_args()

"""Annotation from xml to csv"""

annotation_path = '/kaggle/input/pascal-voc-2012/VOC2012/Annotations'
xml_df = xml_to_csv(annotation_path)
xml_df.to_csv(('/kaggle/working/annotation.csv'), index=None)

"""Modify csv data"""
data = pd.read_csv('/kaggle/working/annotation.csv')
modified_data = modify(data=data, min_dimension=args.min_dim, max_dimension=args.max_dim)

"create datasets and dataloaders"

train_val_dataset = CustomDataset(images_path=TRAIN_IMAGES, labels_path=TRAIN_LABELS, width=RESIZE_TO, height=RESIZE_TO,
                                  classes=CLASSES)
train_len = int(len(train_val_dataset) * 0.85)
valid_len = len(train_val_dataset) - train_len

train_dataset, val_dataset = random_split(train_val_dataset, lengths=[train_len, valid_len])
train_dataset.transforms = get_train_transform()
val_dataset.transforms = get_valid_transform()

train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, collate_fn=collate_fn)

"""create Faster R-CNN model"""
model = create_model(len(CLASSES), k=args.k, data=modified_data, mode=args.mode, fine_tune=args.finetune)
model.to(DEVICE)

"""Define optimizer, scheduler"""

total_params = sum(p.numel() for p in model.parameters())
params = [p for p in model.parameters() if p.requires_grad == True]
# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS + 25, T_mult=1, verbose=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
train_loss_list = []

"""Training progress"""

for epoch in range(EPOCHS):
    train_loss_hist = Averager()
    train_loss_hist.reset()

    _, batch_loss_list = train_one_epoch(
        model,
        optimizer,
        train_dataloader,
        DEVICE,
        epoch,
        train_loss_hist,
        print_freq=100,
        scheduler=scheduler
    )

    evaluate(model, valid_dataloader, device=DEVICE)

    # Add the current epoch's batch-wise lossed to the `train_loss_list`.
    train_loss_list.extend(batch_loss_list)

    # Save the current epoch model.
    save_model(OUT_DIR, epoch, model, optimizer)

    # Save loss plot.
    save_train_loss_plot(OUT_DIR, train_loss_list)
