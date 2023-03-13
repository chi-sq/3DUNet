# coding: utf-8
# @Author: Salt
from scipy import ndimage


def resize_volume(img, desired_depth=128, desired_width=128, desired_height=128):
    """Resize across z-axis"""
    # Set the desired depth

    # Get current depth
    current_depth = img.shape[0]
    current_width = img.shape[-1]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    if img.max() == 4:
        img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=0)
    else:
        img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=3)  # 三次样条

    return img
