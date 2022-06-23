import argparse
import pandas as pd
from kmeans.modifydata import modify
from kmeans.xml_to_csv import xml_to_csv
from config import *
from dataset.dataset import CustomDataset
from torch.utils.data import DataLoader, random_split
from engine.model import create_model
from engine.engine import train_one_epoch, evaluate
from engine.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, help='số lượng nhóm muốn phân cụm')
    parser.add_argument('--min_dim', type=int, help='min_dimension')
    parser.add_argument('--max_dim', type=int, help='max_dimension')
    parser.add_argument('--mode', type=str,
                        help="chế độ khởi tạo Anchox.Original: khởi tạo giống như trong bài báo.'Kmean: khởi tạo ứng dụng kmean")
    parser.add_argument('--lr', default=0.001, type=float, help="Hệ số học")
    parser.add_argument('--batchsize', default=32, type=int, help="kích thước batch")
    args = parser.parse_args()

    """Chuyển từ các file .xml về file .csv để tiện cho quá trình sử dụng kmean"""
    annotation_path = '/kaggle/input/pascal-voc-2012/VOC2012/Annotations'
    xml_df = xml_to_csv(annotation_path)
    xml_df.to_csv(('/kaggle/working/annotation.csv'), index=None)

    """Modify csv data"""
    data = pd.read_csv('/kaggle/working/annotation.csv')
    modified_data = modify(data=data, min_dimension=args.min_dim, max_dimension=args.max_dim)

    "Taọ dataset và dataloader"
    train_val_dataset = CustomDataset(image_paths=TRAIN_IMAGES, label_paths=TRAIN_LABELS, width=RESIZE_TO,
                                      height=RESIZE_TO, classes=CLASSES)
    train_len = int(len(train_val_dataset) * 0.85)
    valid_len = len(train_val_dataset) - train_len
    train_dataset, valid_dataset = random_split(train_val_dataset, lengths=[train_len, valid_len])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, collate_fn=collate_fn)

    "Tạo mô hình"
    model = create_model(len(CLASSES), k=args.k, data=modified_data, mode=args.mode)
    model.to(DEVICE)

    """Tạo hàm tối ưu, hàm điều chỉnh hệ số học"""
    total_params = sum(p.numel() for p in model.parameters())
    params = [p for p in model.parameters() if p.requires_grad == True]
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
    train_loss_list = []

    """Vòng lặp huấn luyện"""
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
        train_loss_list.extend(batch_loss_list)
