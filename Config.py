import torch
############
import argparse
from pathlib import Path
########################
#source: https://docs.python.org/3/library/argparse.html
########################################################
parser = argparse.ArgumentParser(description = "Outils pour le PFE", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
#######################
######__General__######
#######################
parser.add_argument("--All_WSI", 
					metavar = 'wsi', 
					type = Path, 
					default = Path("All_WSI"), 
					help = "Location of all WSI")

parser.add_argument("--All_Data_sets", 
					metavar = 'ds', 
					type = Path, 
					default = Path("All_Data_sets"), 
					help = "Data sets directory")

parser.add_argument("--Classes",
					type = list,
					default = ["Normal", "Acinar", "Solid", "Papillary", "Micropaipllary", "Lepidic"],
					help = "Diffrent subtype of cancer")
#####################
######__Split__######
#####################
parser.add_argument("--Train_WSI", 
					metavar = 't_wsi', 
					type = Path, 
					default = Path("All_Data_sets/WSI_train"), 
					help = "Location to be created to store WSI training set")

parser.add_argument("--Validation_WSI", 
					metavar = 'v_wsi', 
					type = Path, 
					default = Path("All_Data_sets/WSI_val"), 
					help = "Location to be created to store WSI validation set")

parser.add_argument("--Validation_WSI_Size", 
					metavar = 'v_size', 
					type = int, 
					default = 30, 
					help = "Number of validation WSI per class")

parser.add_argument("--Test_WSI", 
					metavar = 'Tst_wsi', 
					type = Path, 
					default = Path("All_Data_sets/WSI_test"), 
					help = "Location to be created to store WSI testing set")

parser.add_argument("--Test_WSI_Size", 
					metavar = 't_size', 
					type = int, 
					default = 20, 
					help = "Number of test WSI per class")
##############################
######__Make_CSV_files__######
##############################
parser.add_argument("--Train_labels", 
					metavar = 't_l', 
					type = Path, 
					default = Path("Data_sets/WSI_train/Train_labels/"), 
					help = "File location to be created to store the train_labels.csv file")

parser.add_argument("--Validation_labels", 
					metavar = 'v_l', 
					type = Path, 
					default = Path("Data_sets/WSI_val/Val_labels/"), 
					help = "Location to be created to store the Val_labels.csv file")

parser.add_argument("--Test_labels", 
					metavar = 'tst_l', 
					type = Path, 
					default = Path("Data_sets/WSI_test/Test_labels/"), 
					help = "Location to be created store the test_labels.csv file")
##########################
######__Processing__######
##########################
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

parser.add_argument("--Window_size",  
					type = int, 
					default = 224 * 5, 
					help = "Size of the sliding window")

parser.add_argument("--Compression_factor",  
					type = int, 
					default = 5, 
					help = "The compression factor")
#####################
######__Model__######
#####################
parser.add_argument("--num_epochs",
                    type = int,
                    default = 1,
                    help = "Number of epochs for training")

parser.add_argument("--learning_rate",
                    type = float,
                    default = 0.001,
                    help = "Learning rate to use for gradient descent")

parser.add_argument("--batch_size",
                    type = int,
                    default = 6,
                    help = "Mini-batch size to use for training")

parser.add_argument("--learning_rate_decay",
                    type = float,
                    default = 0.0001,
                    help = "Learning rate decay amount per epoch")

parser.add_argument("--weight_decay",
                    type = float,
                    default = 0.85,
                    help = "Weight decay (L2 penalty) to use in optimizer")

parser.add_argument("--Path2Weights",
					type = Path,
					default = Path("Train_folder/Model/Best_model_weights/weights.pth"),
					help = "Location to store best model")

parser.add_argument("--Sanity_Check",
					type = bool,
					default = True,
					help = "Weither to stop training after one batch or not")

parser.add_argument("--Predictions",
					type = Path,
					default = Path("Train_folder/Model/Predictions/"),
					help = "Location to write out the predictions.")

parser.add_argument("--Save_interval",
                    type = int,
                    default = 1,
                    help = "Number of epochs between saving checkpoints")

parser.add_argument("--Checkpoints_folder",
                    type = Path,
                    default = Path("Train_folder/Model/Checkpoints"),
                    help = "Directory to save model checkpoints to")

parser.add_argument("--Resume_checkpoint",
                    type = bool,
                    default = False,
                    help = "Resume model from checkpoint file")

parser.add_argument("--Checkpoint_file",
    				type = Path,
    				default = Path("Train_folder/Model/Checkpoints/resnet18_e0_val0.50000.pt"),
    				help = "Checkpoint file to load if resume_checkpoint_path is True")

#########################
######__Arguments__######
#########################
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")