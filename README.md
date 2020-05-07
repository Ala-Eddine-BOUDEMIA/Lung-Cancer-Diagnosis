# PFE

Deep learning classification of lung cancer subtypes using High Resolution Microscopy Images (Whole-Slide Images)

## Requirements

- [NumPy 1.16](https://numpy.org/)
- [OpenSlide Python](https://openslide.org/api/python/)
- [pandas](https://pandas.pydata.org/)
- [PIL](https://pillow.readthedocs.io/en/5.3.x/)
- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [PyTorch](https://pytorch.org/)
- [scikit-image](https://scikit-image.org/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [NVIDIA GPU](https://www.nvidia.com/en-us/)
- [Ubuntu](https://ubuntu.com/)

# Usage

Take a look at `Config.py` before you begin to get a feel for what parameters can be changed.

## 1. 1-Split:

Splits the data into a validation and test set. Default validation whole-slide images (WSI) per class is 20 and test images per class is 30. You can change these numbers by changing the `Validation_WSI_Size` and `Test_WSI_Size` flags at runtime. You can skip this step if you did a custom split (for example, you need to split by patients).

```
python3 1-split.py
```

Note that the data will not be duplicated but moved.

**Inputs**: `All_wsi`, `Validation_WSI_Size`, `Test_WSI_Size` 

**Outputs**: `Train_WSI`, `Validation_WSI`, `Test_WSI`

### Example
```
python3 1-split.py --Validation_WSI_Size 10 --Test_WSI_Size 20
```

## 2. 2-Processing

- Generate patches for the training set.
- Generate patches for the validation set.
- Generate patches for the testing set

```
python3 2-Processing.py
```
If your histopathology images are H&E-stained, whitespace will automatically be filtered. .

**Inputs**: `Train_WSI`, `Validation_WSI`, `Test_WSI`

**Outputs**: `Train_patches` (fed into model for training), `Validation_patches` (for validation), `Test_patches` (for testing)

## 3. 3-Train_Val

```
python3 3-Train_Val.py
```

We are using ResNet-18 You can change the model from `Model_Utils.py`. There is an option to retrain from a previous checkpoint. Model checkpoints are saved by default every epoch in `Checkpoints`. loss history and metrics history of each epoch are saved in a csv file at `Diagnostics`. The best model parameters are saved at `Best_model_weights`
You can put `Sanity_Check` to True, to run only one epoch when you are testing the code.

**Inputs**: `Validation_patches`, `Train_patches`

**Outputs**: `Checkpoints`, `Diagnostics`, `Best_model_weights`

### Example
```
python3 3-Train_Val.py --batch_size 32 --num_epochs 100 --save_interval 5
```

## 4. 4-Test

Run the model on all the patches in the Test_patches folder generated previously.
Patches should be organized by subtypes, for example : Test_patches/classe_a/image1.jpeg

```
python3 4-Test.py
```

We automatically choose the model with the best validation accuracy. You can also specify your own. 

**Inputs**: `Test_patches`

**Outputs**: `Predictions`

## 5. 5-Evaluation

Use the outputed csv file from `4-Test.py` and predict on WSI level, the results will be outputed in a csv file with the name of the WSI image and will contain classes that were predicted, the ratio of the predicted class ans its confidence.

```
python3 5-Evaluation.py
```

We automatically choose the model with the best validation accuracy. You can also specify your own. 

**Inputs**: `Predictions`

**Outputs**: `WSI_Predictions`