import numpy as np


def change_to_wh(data):
    "Hàm tính toán chiều rộng và chiều cao các bounding box dựa trên tọa độ"
    data['w'] = data['xmax'] - data['xmin'] + 1
    data['h'] = data['ymax'] - data['ymin'] + 1
    return data


def _compute_new_static_size(width, height, min_dimension, max_dimension):
    orig_height = height
    orig_width = width
    orig_min_dim = min(orig_height, orig_width)

    large_scale_factor = min_dimension / float(orig_min_dim)

    large_height = int(round(orig_height * large_scale_factor))
    large_width = int(round(orig_width * large_scale_factor))
    large_size = [large_height, large_width]
    if max_dimension:

        orig_max_dim = max(orig_height, orig_width)
        small_scale_factor = max_dimension / float(orig_max_dim)

        small_height = int(round(orig_height * small_scale_factor))
        small_width = int(round(orig_width * small_scale_factor))
        small_size = [small_height, small_width]
        new_size = large_size
    if max(large_size) > max_dimension:
        new_size = small_size
    else:
        new_size = large_size

    return new_size[1], new_size[0]


def modify(data, min_dimension, max_dimension):
    data = change_to_wh(data)
    data['new_w'], data['new_h'] = np.vectorize(_compute_new_static_size)(data['width'],
                                                                          data['height'], min_dimension, max_dimension)
    data['b_w'] = data['new_w'] * data['w'] / data['width']
    data['b_h'] = data['new_h'] * data['h'] / data['height']
    data['b_ar'] = data['b_w'] / data['b_h']

    base_box = 512 * 512
    data['b_area_scale'] = (data['w'] * data['h'] / (base_box)).apply(np.sqrt)

    base_anchor = 512
    data['tf_scale'] = data['b_h'] * (data['b_ar']).apply(np.sqrt) / base_anchor
    data['tf_scale_2'] = data['b_w'] / ((data['b_ar']).apply(np.sqrt) * base_anchor)

    return data
