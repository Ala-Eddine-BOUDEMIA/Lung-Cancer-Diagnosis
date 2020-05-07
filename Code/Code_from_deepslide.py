import Config
#############
import random
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import block_reduce
from typing import (Dict, IO, List, Tuple)
##########################################

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

#############################
######___utils_model.py___###
#############################
class Random90Rotation:
    def __init__(self, degrees: Tuple[int] = None) -> None:
        """
        Randomly rotate the image for training. Credits to Naofumi Tomita.

        Args:
            degrees: Degrees available for rotation.
        """
        self.degrees = (0, 90, 180, 270) if (degrees is None) else degrees

    def __call__(self, im: Image) -> Image:
        """
        Produces a randomly rotated image every time the instance is called.

        Args:
            im: The image to rotate.

        Returns:    
            Randomly rotated image.
        """
        return im.rotate(angle=random.sample(population=self.degrees, k=1)[0])


def calculate_confusion_matrix(all_labels, all_predicts, classes = Config.args.Classes) -> None:
    """
    Prints the confusion matrix from the given data.

    Args:
        all_labels: The ground truth labels.
        all_predicts: The predicted labels.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
    """
    remap_classes = {x: classes[x] for x in range(len(classes))}

    # Set print options.
    # Sources:
    #   1. https://stackoverflow.com/questions/42735541/customized-float-formatting-in-a-pandas-dataframe
    #   2. https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
    #   3. https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 0

    actual = pd.Series(pd.Categorical(
        pd.Series(all_labels).replace(remap_classes), categories=classes),
                       name="Actual")

    predicted = pd.Series(pd.Categorical(
        pd.Series(all_predicts).replace(remap_classes), categories=classes),
                          name="Predicted")

    cm = pd.crosstab(index=actual, columns=predicted, normalize="index")

    cm.style.hide_index()
    print(cm)

    return cm