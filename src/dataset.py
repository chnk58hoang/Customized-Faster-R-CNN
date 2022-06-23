import os
import glob as glob
from xml.etree import ElementTree as et
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class FileLoader():
    def __init__(self, image_paths, label_paths, width, height, classes):
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

        # Remove all annotations and images when no object is present.
        self.read_and_clean()
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def read_and_clean(self):
        """
        This function will discard any images and labels when the XML
        file does not contain any object.
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

    def __call__(self):
        data = []
        for image_name in self.all_images:
            image_path = os.path.join(self.image_paths, image_name)
            image = cv2.imread(image_path)
            # convert BGR to RGB color format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_resized = cv2.resize(image, (self.width, self.height))
            image_resized = torch.tensor(image_resized)
            image_resized = image_resized.transpose(0, 2)
            annot_filename = image_name[:-4] + '.xml'
            annot_file_path = os.path.join(self.label_paths, annot_filename)

            boxes = []
            labels = []
            tree = et.parse(annot_file_path)
            root = tree.getroot()

            # get the height and width of the image
            image_width = image.shape[1]
            image_height = image.shape[0]

            # box coordinates for xml files are extracted and corrected for image size given
            for member in root.findall('object'):
                # map the current object name to `classes` list to get...
                # ... the label index and append to `labels` list
                labels.append(self.classes.index(member.find('name').text))

                # xmin = left corner x-coordinates
                xmin = float(member.find('bndbox').find('xmin').text)
                # xmax = right corner x-coordinates
                xmax = float(member.find('bndbox').find('xmax').text)
                # ymin = left corner y-coordinates
                ymin = float(member.find('bndbox').find('ymin').text)
                # ymax = right corner y-coordinates
                ymax = float(member.find('bndbox').find('ymax').text)

                # resize the bounding boxes according to the...
                # ... desired `width`, `height`
                xmin_final = int((xmin / image_width) * self.width)
                xmax_final = int((xmax / image_width) * self.width)
                ymin_final = int((ymin / image_height) * self.height)
                ymax_final = int((ymax / image_height) * self.height)

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

            # bounding box to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # area of the bounding boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # no crowd instances
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            # labels to tensor
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # prepare the final `target` dictionary
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = torch.tensor([self.all_images.index(image_name)])

            data.append({'image': image_resized, 'target': target})

        return data


class CustomDataset(Dataset):
    def __init__(self, datalist, transforms):
        self.datalist = datalist
        self.transforms = transforms

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image = self.datalist[idx]['image']
        target = self.datalist[idx]['target']

        if self.transforms:
            sample = self.transforms(image=image,
                                     bboxes=target['boxes'],
                                     labels=target['labels'])
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image, target
