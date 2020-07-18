from pathlib import Path
from typing import Dict
import argparse
import torch

# source: https://docs.python.org/3/library/argparse.html

parser = argparse.ArgumentParser(description = "Tools and parameters", 
								formatter_class = argparse.ArgumentDefaultsHelpFormatter)

##########################################
######__Preprocessing__&__Processing######
##########################################
parser.add_argument("--All_WSI", 
					metavar = 'wsi', 
					type = Path, 
					default = Path("All_WSI"), 
					help = "Location of all WSI")

parser.add_argument("--Annotations",
					type = Path, 
					default = Path("Annotations"), 
					help = "Annotations directory")

parser.add_argument("--Patches",
					type = Path, 
					default = Path("Patches"), 
					help = "Location to store the generated patches")

parser.add_argument("--CSV_files",
					type = Path, 
					default = Path("CSV_files"), 
					help = "Location to store the generated patches")

parser.add_argument("--Classes",
					type = list,
					default = ["ACINAR", "CRIB", "MICROPAP", "NC", "SOLID"],
					help = "Subtypes of lung cancer")

parser.add_argument("--Window_size",  
					type = int, 
					default = 224 * 2, 
					help = "Size of the sliding window")

parser.add_argument("--Compression_factor",  
					type = int, 
					default = 2, 
					help = "The compression factor")

parser.add_argument("--Overlap",
					type = Dict,
					default = {"ACINAR": 0.45, "CRIB": 0.3, "MICROPAP": 0.75, "NC": 1, "SOLID": 1},
					help = "Overlap factor while generating patches")

parser.add_argument("--Maximum",
					type = int,
					default = 10000,
					help = "Number of patches per class")

#####################
######__Split__######
#####################
parser.add_argument("--Train_folder",
					metavar = 'train',
					type = Path,
					default = Path("Train_folder"),
					help = "Location to be created to contain train-val-test patches directories")

parser.add_argument("--Train_Patches", 
					metavar = 't_patches', 
					type = Path, 
					default = Path("Train_folder/Train_patches"), 
					help = "Location to be created to store jpeg patches training set")

parser.add_argument("--Validation_Patches", 
					metavar = 'v_patches', 
					type = Path, 
					default = Path("Train_folder/Validation_patches"), 
					help = "Location to be created to store jpeg patches validation set")

parser.add_argument("--Test_Patches", 
					metavar = 'Tst_patches', 
					type = Path, 
					default = Path("Train_folder/Test_patches"), 
					help = "Location to be created to store jpeg patches testing set")

parser.add_argument("--Validation_Set_Size", 
					type = int, 
					default = 1000, 
					help = "Number of patches per class in validation set")

parser.add_argument("--Test_Set_Size", 
					type = int, 
					default = 1000, 
					help = "Number of patches per class in Test set")

#####################
######__Model__######
#####################
parser.add_argument("--num_epochs",
                    type = int,
                    default = 50,
                    help = "Number of epochs for training")

parser.add_argument("--learning_rate",
                    type = float,
                    default = 0.001,
                    help = "Learning rate to use for gradient descent")

parser.add_argument("--batch_size",
                    type = int,
                    default = 16,
                    help = "Mini-batch size to use for training")

parser.add_argument("--learning_rate_decay",
                    type = float,
                    default = 1,
                    help = "Learning rate decay amount per epoch")

parser.add_argument("--weight_decay",
                    type = float,
                    default = 0,
                    help = "Weight decay (L2 penalty) to use in optimizer")

parser.add_argument("--Sanity_Check",
					type = bool,
					default = False,
					help = "Weither to stop training after one batch or not")

parser.add_argument("--BestWeights",
					type = Path,
					default = Path("Train_folder/Model/Best_model_weights/"),
					help = "Location to store best model")

parser.add_argument("--Path2Weights",
					type = Path,
					default = Path("Train_folder/Model/Best_model_weights/weights.pth"),
					help = "File to store best model")

parser.add_argument("--Diagnostics_Directory",
					type = Path,
					default = Path("CSV_files/Diagnostics/"),
					help = "Location to write out the diagnostics.")

parser.add_argument("--Diagnostics",
					type = Path,
					default = Path("CSV_files/Diagnostics/Model_Diagnostics.csv"),
					help = "Location to write out the diagnostics.")

parser.add_argument("--Resume_checkpoint",
                    type = bool,
                    default = False,
                    help = "Resume model from checkpoint file")

parser.add_argument("--Save_interval",
                    type = int,
                    default = 1,
                    help = "Number of epochs between saving checkpoints")

parser.add_argument("--Checkpoints_folder",
                    type = Path,
                    default = Path("Train_folder/Model/Checkpoints"),
                    help = "Directory to save model checkpoints to")

parser.add_argument("--Checkpoint_file",
    				type = Path,
    				default = Path("Train_folder/Model/Checkpoints/'replace with checkpoint file name'"),
    				help = "Checkpoint file to load if resume_checkpoint_path is True")

parser.add_argument("--Predictions_Directory",
					type = Path,
					default = Path("CSV_files/Predictions/"),
					help = "Location to write out the predictions.")

parser.add_argument("--Predictions",
					type = Path,
					default = Path("CSV_files/Predictions/predictions.csv"),
					help = "Location to write out the predictions.")

#############################
######__Visualization__######
#############################
parser.add_argument("--Visualization",
					type = Path,
					default = Path("Visualization"),
					help = "Location to save visualization")

#########################
######__Arguments__######
#########################
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")