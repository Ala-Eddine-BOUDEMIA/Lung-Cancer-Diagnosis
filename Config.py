import argparse
from pathlib import Path
########################
#source: https://docs.python.org/3/library/argparse.html
########################################################
parser = argparse.ArgumentParser(description = "Outils pour le PFE", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
######################
######__Inputs__######
######################
parser.add_argument("--All_WSI", 
					metavar = 'wsi', 
					type = Path, 
					default = Path("Input_Images_WSI"), 
					help = "Location of all WSI")

parser.add_argument("--All_Data_sets", 
					metavar = 'ds', 
					type = Path, 
					default = Path("Data_sets"), 
					help = "Data sets directory")

parser.add_argument("--Validation_WSI_Size", 
					metavar = 'v_size', 
					type = int, 
					default = 20, 
					help = "Number of validation WSI per class")

parser.add_argument("--Test_WSI_Size", 
					metavar = 't_size', 
					type = int, 
					default = 30, 
					help = "Number of test WSI per class")
#####################################
######__Folders_to_be_created__######
#####################################
parser.add_argument("--Train_WSI", 
					metavar = 't_wsi', 
					type = Path, 
					default = Path("Data_sets/WSI_train"), 
					help = "Location to be created to store WSI training set")

parser.add_argument("--Validation_WSI", 
					metavar = 'v_wsi', 
					type = Path, 
					default = Path("Data_sets/WSI_val"), 
					help = "Location to be created to store WSI validation set")

parser.add_argument("--Test_WSI", 
					metavar = 'Tst_wsi', 
					type = Path, 
					default = Path("Data_sets/WSI_test"), 
					help = "Location to be created to store WSI testing set")

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

parser.add_argument("--Train_Patches", 
					metavar = 't_patches', 
					type = Path, 
					default = Path("Data_sets/Train_patches"), 
					help = "Location to be created to store jpeg patches training set")

parser.add_argument("--Validation_Patches", 
					metavar = 'v_patches', 
					type = Path, 
					default = Path("Data_sets/Validation_patches"), 
					help = "Location to be created to store jpeg patches validation set")

parser.add_argument("--Test_Patches", 
					metavar = 'Tst_patches', 
					type = Path, 
					default = Path("Data_sets/Test_patches"), 
					help = "Location to be created to store jpeg patches testing set")
##########################
######__Processing__######
##########################
#parser.add_argument()
#########################
######__Arguments__######
#########################
args = parser.parse_args()