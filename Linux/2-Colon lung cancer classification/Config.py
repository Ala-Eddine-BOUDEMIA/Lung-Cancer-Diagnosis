import argparse
from typing import Dict
from pathlib import Path

import torch

# source: https://docs.python.org/3/library/argparse.html

parser = argparse.ArgumentParser(
	description = "Tools and parameters", 
	formatter_class = argparse.ArgumentDefaultsHelpFormatter)

##########################################
######__Preprocessing__&__Processing######
##########################################
parser.add_argument(
	"--All_Dataset", 
	type = Path, 
	default = Path("lung_colon_image_set"), 
	help = "Location of all images")

parser.add_argument(
	"--Patches",
	type = Path, 
	default = Path("Patches"), 
	help = "Location to store the generated patches")

parser.add_argument(
	"--CSV_files",
	type = Path, 
	default = Path("CSV_files"), 
	help = "Location to store the generated patches")

parser.add_argument(
	"--Classes",
	type = list,
	default = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"],
	help = "Subtypes of cancer")

parser.add_argument(
	"--Maximum",
	type = int,
	default = 10000,
	help = "Number of patches per class")

#####################
######__Split__######
#####################
parser.add_argument(
	"--Train_folder",
	metavar = 'train',
	type = Path,
	default = Path("Train_folder"),
	help = "Location to be created to contain train-val-test patches directories")

parser.add_argument(
	"--Train_Patches", 
	metavar = 't_patches', 
	type = Path, 
	default = Path("Train_folder/Train_patches"), 
	help = "Location to be created to store jpeg patches training set")

parser.add_argument(
	"--Validation_Patches", 
	metavar = 'v_patches', 
	type = Path, 
	default = Path("Train_folder/Validation_patches"), 
	help = "Location to be created to store jpeg patches validation set")

parser.add_argument(
	"--Test_Patches", 
	metavar = 'Tst_patches', 
	type = Path, 
	default = Path("Train_folder/Test_patches"), 
	help = "Location to be created to store jpeg patches testing set")

parser.add_argument(
	"--Train_Set_Size", 
	type = int, 
	default = 8000, 
	help = "Number of patches per class in validation set")

parser.add_argument(
	"--Validation_Set_Size", 
	type = int, 
	default = 1000, 
	help = "Number of patches per class in validation set")

parser.add_argument(
	"--Test_Set_Size", 
	type = int, 
	default = 1000, 
	help = "Number of patches per class in Test set")

#####################
######__Model__######
#####################
parser.add_argument(
	"--num_epochs",
    type = int,
    default = 100,
    help = "Number of epochs for training")

parser.add_argument(
	"--learning_rate",
    type = float,
    default = 0.00001,
    help = "Learning rate to use for gradient descent")

parser.add_argument(
	"--batch_size",
    type = int,
    default = 16,
    help = "Mini-batch size to use for training")

parser.add_argument(
	"--Sanity_Check",
	type = bool,
	default = False,
	help = "Weither to stop training after one batch or not")

parser.add_argument(
	"--BestWeights",
	type = Path,
	default = Path("Train_folder/Model/Best_model_weights/"),
	help = "Location to store best model")

parser.add_argument(
	"--Path2Weights",
	type = Path,
	default = Path("Train_folder/Model/Best_model_weights/weights.pt"),
	help = "File to store best model")

parser.add_argument(
	"--Diagnostics_Directory",
	type = Path,
	default = Path("CSV_files/Diagnostics/"),
	help = "Location to write out the diagnostics.")

parser.add_argument(
	"--Diagnostics",
	type = Path,
	default = Path("CSV_files/Diagnostics/Model_Diagnostics.csv"),
	help = "Location to write out the diagnostics.")

parser.add_argument(
	"--Resume_checkpoint",
    type = bool,
    default = False,
    help = "Resume model from checkpoint file")

parser.add_argument(
	"--Save_interval",
    type = int,
    default = 1,
    help = "Number of epochs between saving checkpoints")

parser.add_argument(
	"--Checkpoints_folder",
    type = Path,
    default = Path("Train_folder/Model/Checkpoints"),
    help = "Directory to save model checkpoints to")

parser.add_argument(
	"--Checkpoint_file",
	type = Path,
	default = Path("Train_folder/Model/Checkpoints/'replace with checkpoint file name'"),
	help = "Checkpoint file to load if resume_checkpoint_path is True")

parser.add_argument(
	"--Predictions_Directory",
	type = Path,
	default = Path("CSV_files/Predictions/"),
	help = "Location to write out the predictions.")

parser.add_argument(
	"--Predictions",
	type = Path,
	default = Path("CSV_files/Predictions/predictions.csv"),
	help = "Location to write out the predictions.")

#############################
######__Visualization__######
#############################
parser.add_argument(
	"--Visualization",
	type = Path,
	default = Path("Visualization"),
	help = "Location to save visualization")

#########################
######__Arguments__######
#########################
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")