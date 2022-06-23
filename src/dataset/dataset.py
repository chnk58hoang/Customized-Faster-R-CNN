import os
import glob as glob
from xml.etree import ElementTree as et
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths, width, height, classes, transforms=None):
        """
            image_paths: đường dẫn đến folder chứa các file ảnh
            label_paths: đường dẫn đến folder chứa các file annotation .xml
            width, height: kích thước bức ảnh muốn đưa về
            classes: danh sách tên các lớp đối tượng
            transform: hàm phục vụ quá trình data augmentation
        """

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.width = width
        self.height = height
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        self.all_image_paths = []
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(f"{self.image_paths}/{file_type}"))
        self.all_annot_paths = glob.glob(f"{self.label_paths}/*.xml")

        self.read_and_clean()
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        self.transforms = transforms

    def read_and_clean(self):
        """
        Loại bỏ những ảnh có file annotation không chứa object nào.
        """
        for annot_path in self.all_annot_paths:
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = False
            for _ in root.findall('object'):
                object_present = True
            if object_present == False:
                print(f"Removing {annot_path} and corresponding image")
                self.all_annot_paths.remove(annot_path)
                self.all_image_paths.remove(annot_path.split('.xml')[0] + '.jpg')

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.image_paths, image_name)
        image = cv2.imread(image_path)

        "Đưa ảnh từ định dạng BGR về RGB"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        "Đưa ảnh về kích thước định nghĩa sẵn"
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized = torch.tensor(image_resized)
        "Chuyển về dạng tensor"
        image_resized = image_resized.transpose(0, 2)
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.label_paths, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        "Lấy chiều rộng và chiều cao của ảnh"
        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            "Lấy các tọa độ xmin,ymin,xmax,ymax ứng với từng object"
            labels.append(self.classes.index(member.find('name').text))
            xmin = float(member.find('bndbox').find('xmin').text)
            xmax = float(member.find('bndbox').find('xmax').text)
            ymin = float(member.find('bndbox').find('ymin').text)
            ymax = float(member.find('bndbox').find('ymax').text)

            "Scale các giá trị trên cho tương ứng với bức ảnh đã resize"
            xmin_final = int((xmin / image_width) * self.width)
            xmax_final = int((xmax / image_width) * self.width)
            ymin_final = int((ymin / image_height) * self.height)
            ymax_final = int((ymax / image_height) * self.height)

            "Lấy tất cả các thông tin của các boundbox"
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        "Chuyển về tensor"
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        "Đưa các nhãn vào dictionary target"
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([self.all_images.index(image_name)])

        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

            return image_resized, target

        return image_resized, target

    def __len__(self):
        return len(self.all_images)
