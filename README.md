# Lung Cancer Diagnosis

__Diagnosis of histologic growth patterns of lung cancer in digital slides using deep learning.__

# Table of contents

* [General Information](#General-Information)
* [Requirements](#Requirements)
* [Installation](#Installation)
* [Usage](#Usage)
* [Known Issues and Limitations](#Known-Issues-and-Limitations)
* [Future Work](#Future-Work)
* [Sources](#Sources)

# General Information

- _This is a final year graduation project._
- We are using 26 whole-slide images obtained from The Cancer Genome Atlas (LUAD).
* Distribution of data among histologic patterns is as follows:

Histologic pattern | ACINAR | CRIBRIFORM | MACROPAPILLARY | NON CANCEROUS | SOLID 
-------------------|--------|------------|----------------|---------------|------
Crops | 22 | 4 | 23 | 53 |85
Patches (overlap = 1) | 1328 | 85 | 4053 | 8418 | 5821
 
# Requirements

- [OpenSlide Python](https://openslide.org/api/python/)
- [PIL](https://pillow.readthedocs.io/en/5.3.x/)
- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [PyTorch](https://pytorch.org/)
- [pytorch-gradcam](https://pypi.org/project/pytorch-gradcam/)
- [scikit-image](https://scikit-image.org/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [tensorboard](https://www.tensorflow.org/tensorboard)
- [torchvision](https://pytorch.org/docs/stable/torchvision/index.html#module-torchvision)
- [torchsummary](https://pypi.org/project/torchsummary/)

# Installation 

## Download rpository:


`git clone https://github.com/Ala-Eddine-BOUDEMIA/Lung-Cancer-Diagnosis.git`


`cd Lung-Cancer-Diagnosis`


## Install packages:


`sudo pip3 install -r requirements.txt`


## Run project:


`cd Code`


`python3 1-Preprocessing.py`


`python3 2-Processing.py`


`python3 3-Split.py`


`python3 4-Train_Val.py`


`python3 5-Test.py`


`python3 6-Evaluation.py`


`python3 7-Visualization.py`


# Usage

Take a look at `Code/Config.py` before you begin to get a feel for what parameters can be changed.

## 1. Preprocessing:

- Reads from `Annotations` folder that contains XML annotation files.
- Reads from `All_WSI` folder that contains Whole Slide Images.
- Generates patches and saves information about the patches in a csv file.


```
python3 Code/1_Preprocessing.py
```

**Inputs**: `All_WSI`, `Annotations`

**Outputs**: `Patches/SUBTYPE`, `CSV_files/Annotations`

- Note that: `SUBTYPE == ACINAR, CRIB, MICROPAP, NC, SOLID`.

If your histopathology images are H&E-stained, whitespace will automatically be filtered. 
You can change overlapping area using the `--Overlap` option.

## 2. Processing:

The goal of this code is to balance data using data augmentation techniques.

- Reads patches randomly from each subtype directory at a time.
- Applies diffrent transformations to the image.
- The nature and number of transformations applied to an image are chosen randomly.
- Modified patches are saved in the same directory as the original image.


```
python3 Code/2_Preprocessing.py
```

Note that this may take some time and eventually a significant amount of space. Change `--Maximum` to be smaller if you wish not to generate as many windows. 

**Inputs**: `Patches/SUBTYPE`

**Outputs**: `Patches/SUBTYPE`

## 3. Split:

Splits the data into a train, validation and test set. Default validation and test patches per class is 1000.
You can change these numbers by changing the `--Validation_Set_Size` and `--Test_Set_Size`. 
You can skip this step if you did a custom split (for example, you need to split by patients).

Note that the modified images will be ditributed to the same set as the original, so the model won't be memorizing patterns.


```
python3 Code/3_split.py
```

**Inputs**: `Patches` 

**Outputs**: `Train_folder/Train_patches`, `Train_folder/Validation_patches`, `Train_folder/Test_patches`

## 4. Train_val:

We recommend using ResNet-18 if you are training on a relatively small histopathology dataset. You can change hyperparameters using the `argparse` flags. There is an option to retrain from a previous checkpoint. Model checkpoints are saved by default every epoch in `Train_folder/Model/Checkpoints`.

**Inputs**: `Train_folder/Train_patches`, `Train_folder/Validation_patches`

**Outputs**: `Train_folder/Model/Checkpoints`, `Train_folder/Model/Best_model_weights`, `CSV_files/Diagnostics`, 
`Tensorboard`

## 5. Test:

Run the model on all the patches for each WSI in the test set.


```
python3 Code/5_test.py
```

We automatically choose the model with the best validation accuracy while training. You can also specify your own. 

**Inputs**: `Train_folder/Test_patches`

**Outputs**: `CSV_files/Diagnostics`, `CSV_files/Predictions`, `Tensorboard`

## Tensorboard

We are using tensorboard to evaluate the model. 
- Uploads a grid of train and validation images to make sure that the patches are good. 
- Uploads the model's graph.
- Uploads confusion matrix and classification report of each epoch.
- Plots the loss function.
- Plots precision recall curve for each class. 

## 6. Evaluation:

Aggregates the patches predictions from the Test code to predict a label at the whole-slide level.
There are various methods to do so, we decided to perform patch averages. Therefore we average the probabilities of all patch predictions, and take the class with the highest probability.


```
python3 Code/6_Evaluation.py
```

**Inputs**: `CSV_files/Predictions`

**Outputs**: `CSV_files/Predictions_cleaned`, `CSV_files/WSI_Name_Prediction.csv`

Note that `WSI_Name_Prediction` refers to actaul name of the WSI in question.

## 7. Visualization

This code allows to see what the network is looking at is to visualize the predictions for each class.

Note that The visualization is a patch level visualization using GradCAM.


```
python3 Code/7_visualization.py
```

**Inputs**: `CSV_files`, `Train_folder`

**Outputs**: `Visualization/Patchs`

# Known Issues and Limitations

- No multiprocessing is supported.
- This code work only when the labels are at the tissue level. In case where no XML annotation file is persent, `1_Preprocessing` will not be able to function properly, therefore neither `2_Processing` nor `3_Split` will.
- `3_Split` Takes a lot of time since it works randomly.
- The overall project structure might be confusing.

# Future Work

- [ ] Try diffrent architectures.
- [ ] Optimize the code to :
	* To reduce computation time.
	* To support multiprocessing.
	* To handle diffrent situations.
- [ ] Visualize on WSI level.
- [ ] Create a web interface. 

# Sources

1. Jason Wei, Laura Tafe, Yevgeniy Linnik, Louis Vaickus, Naofumi Tomita, Saeed Hassanpour, "Pathologist-level Classification of Histologic Patterns on Resected Lung Adenocarcinoma Slides with Deep Neural Networks", Scientific Reports;9:3358 (2019).

2. Gertych, A., Swiderska-Chadaj, Z., Ma, Z. et al. Convolutional neural networks can accurately distinguish four histologic growth patterns of lung adenocarcinoma in digital slides. Sci Rep 9, 1483 (2019). (https://doi.org/10.1038/s41598-018-37638-9) 

# License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)