import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator



def create_model(num_classes, data, k, mode,cluster):
    """
    num_classes: số lớp đối tượng trong bộ dữ liệu
    data: dữ liệu về kích thước, tỉ lệ cạnh của các bounding box
    k: số lượng nhóm muốn phân cụm
    mode: chế độ khởi tạo Anchor Box
    """

    "Định nghĩa backbone CNN cho phần trích chọn đặc trưng"
    backbone = torchvision.models.vgg16(pretrained=True).features
    backbone.out_channels = 512

    "Khởi tạo Anchor"
    if mode == 'kmean':
        aspect_ratios, scales = cluster(data, k)
        anchor_generator = AnchorGenerator(
            sizes=(scales,),
            aspect_ratios=(aspect_ratios,)
        )
    elif mode == 'original':
        anchor_generator = AnchorGenerator(
            sizes=((128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    "Định nghĩa mô hình Faster R-CNN"
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model
