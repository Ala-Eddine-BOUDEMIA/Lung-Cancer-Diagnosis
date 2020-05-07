import Utils
import Config
#############
import random
import shutil
from pathlib import Path
########################

def split(Test_WSI = Config.args.Test_WSI, Train_WSI = Config.args.Train_WSI,
	All_WSI = Config.args.All_WSI, Validation_WSI = Config.args.Validation_WSI, 
	Validation_WSI_Size = Config.args.Validation_WSI_Size, Test_WSI_Size = Config.args.Test_WSI_Size):
	
	path_list, wsi_paths = Utils.parse_dir(All_WSI, "svs")

	FoldersToBeCreated = [Validation_WSI, Test_WSI, Train_WSI]

	for x in FoldersToBeCreated:
		Utils.create_folder(x)

	Val_set_size = Validation_WSI_Size
	Test_set_size = Test_WSI_Size
	Train_set_size = len(path_list) - Val_set_size - Test_set_size

	r = []
	for (dataset, set_size) in ((FoldersToBeCreated[0], Val_set_size), 
								(FoldersToBeCreated[1], Test_set_size), 
								(FoldersToBeCreated[2], Train_set_size)):
		length = 0
		while length < set_size :
			n = random.randint(0,len(path_list)-1)
			if n not in r :
				if path_list[n].exists() == True:
					src = str(path_list[n])
					dst = Path(dataset)
					shutil.move(src, dst)
					r.append(n)
					length += 1

	#Utils.Make_csv_labels_files(FoldersToBeCreated)

if __name__ == '__main__':
	split()