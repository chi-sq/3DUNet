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
    
    depth_factor = desired_depth / current_depth
    width_factor = desired_width / current_width
    height_factor = desired_height / current_height

    # Bilinear interpolation would be order=1,nearest is order=0, and cubic is the default (order=3).
    if img.max() == 2 or img.max() == 1:  # for seg
        img = ndimage.zoom(img, (depth_factor, height_factor, width_factor), order=0)  # we use order=0
    else:  # for image
        img = ndimage.zoom(img, (depth_factor, height_factor, width_factor), order=3)

    return img
