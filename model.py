import numpy as np
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from cluster import iou_base_cluster, euclid_base_cluster


def create_model(num_classes, data, k,mode,fine_tune = False):
    conv1 = torchvision.models.resnet18(pretrained=True).conv1
    bn1 = torchvision.models.resnet18(pretrained=True).bn1
    resnet18_relu = torchvision.models.resnet18(pretrained=True).relu
    resnet18_max_pool = torchvision.models.resnet18(pretrained=True).maxpool
    layer1 = torchvision.models.resnet18(pretrained=True).layer1
    layer2 = torchvision.models.resnet18(pretrained=True).layer2
    layer3 = torchvision.models.resnet18(pretrained=True).layer3
    layer4 = torchvision.models.resnet18(pretrained=True).layer4
    backbone = nn.Sequential(
        conv1, bn1, resnet18_relu, resnet18_max_pool,
        layer1, layer2, layer3, layer4
    )

    for param in backbone.parameters():
        param.requires_grad = fine_tune

    backbone.out_channels = 512

    if mode == 'kmean':
        aspect_ratios, scales = iou_base_cluster(data, k)
        anchor_generator = AnchorGenerator(
            sizes=(scales,),
            aspect_ratios=((0.5,1.0,2.0),)
        )
    if mode == 'original':
        anchor_generator = AnchorGenerator(
            sizes=((128,256,512),),
            aspect_ratios=((0.5,1.0,2.0),)
        )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    print(model)
    return model
