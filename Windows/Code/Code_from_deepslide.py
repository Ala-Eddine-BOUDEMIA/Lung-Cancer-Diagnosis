from skimage.measure import block_reduce
from PIL import Image
import pandas as pd
import numpy as np
import torchvision
import random

##################################
######___utils_processing.py___###
##################################
def is_purple(crop, purple_threshold = 100 * 2, purple_scale_size = 15) -> bool:

    """
    Determines if a given portion of an image is purple.

    Args:
        crop: Portion of the image to check for being purple.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.

    Returns:
        A boolean representing whether the image is purple or not.
    """
    crop = np.array(crop)
    block_size = (crop.shape[0] // purple_scale_size, crop.shape[1] // purple_scale_size, 1)
    pooled = block_reduce(image=crop, block_size=block_size, func=np.average)

    # Calculate boolean arrays for determining if portion is purple.
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    cond1 = r > g - 10
    cond2 = b > g - 10
    cond3 = ((r + b) / 2) > g + 20

    # Find the indexes of pooled satisfying all 3 conditions.
    pooled = pooled[cond1 & cond2 & cond3]
    num_purple = pooled.shape[0]

    return num_purple > purple_threshold