import Split
import Make_CSV_files
import Generate_224_Patches
###########################

def Main(ans):

	if ans == 1:
		Split.split()
	elif ans == 2:
		Make_CSV_files.Make_csv_labels_files()
	elif ans == 3:
		Generate_224_Patches.generate_patches()

if __name__ == '__main__':
	
	ans = 0
	while ans != 4:

		print("1- Split the data")
		print("2- Create csv files containing training, validating and testing labels")
		print("3- Create patches of size 224*224 from the original WSIs")
		print("4-Quit")
		ans = input("Enter your choice: ")

	Main(ans)