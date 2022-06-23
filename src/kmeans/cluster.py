from sklearn.cluster import KMeans
import numpy as np


def euclidean_base_cluster(data, k):
    "Phân cụm kmeans dựa trên khoảng cách Euclidean"
    X = data[['b_w', "b_h"]].to_numpy()
    K = KMeans(k, random_state=0)
    labels = K.fit(X)
    clusters = labels.cluster_centers_
    ar = clusters[:, 0] / clusters[:, 1]
    scale = clusters[:, 1] * np.sqrt(ar)

    return tuple(ar), tuple(scale)


def iou(box, clusters):
    """ Tính IoU giữa bounding box và một cluster"""
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """ Tính IoU trung bình giữa nhiều box và một cluster """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """ Tịnh tiến các box về gốc tọa độ """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Phân cụm kmeans dựa trên IoU
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def iou_base_cluster(data, k):
    """Tiến hành phân cụm dựa trên IoU"""
    X = data[['b_w', 'b_h']].to_numpy()
    cluster = kmeans(X, k)
    ar_iou = cluster[:, 0] / cluster[:, 1]
    scale_iou = cluster[:, 1] * np.sqrt(ar_iou)

    return tuple(ar_iou), tuple(scale_iou)
