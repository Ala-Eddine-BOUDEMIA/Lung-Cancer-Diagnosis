import Split
import Patches
import Make_CSV_files
#####################

def Main():
	ans = 0
	while ans != 4:

		print("1-Split the data")
		print("2-Create csv files containing training, validating and testing labels")
		print("3-Create patches from the original WSI (SVS to JPEG)")
		print("4-Quit")
		ans = input("Enter your choice: ")

	if ans == 1:
		Split.split()
	elif ans == 2:
		Make_CSV_files.Make_csv_labels_files()
	elif ans == 3:
		Patches.read_patch()