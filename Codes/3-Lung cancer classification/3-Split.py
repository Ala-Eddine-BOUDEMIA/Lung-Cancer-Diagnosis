import time
import shutil
import random

from pathlib import Path
from pathlib import PurePosixPath as posix

import Utils
import Config


def split(Patches, Test_Patches, Test_Set_Size, Train_Patches, 
		Validation_Patches, Validation_Set_Size):
	
	since = time.time()

	paths_per_class = {}
	set_size_per_class = {}
	directory_per_class = sorted([d for d in Patches.iterdir() if d.is_dir()])

	for each_directory in directory_per_class:
		tiff_paths = sorted(
			[f for f in each_directory.iterdir() if f.is_file()])
		
		paths_per_class[str(posix(each_directory)).split("/")[-1]] = tiff_paths
		Train_set_size = len(tiff_paths) - (Validation_Set_Size + Test_Set_Size)
		
		set_size_per_class[str(posix(each_directory)).split("/")[-1]] = [
			Validation_Set_Size, Test_Set_Size, Train_set_size
			]

	FoldersToBeCreated = [Validation_Patches, Test_Patches, Train_Patches]
	folders_per_set = {}
	
	for x in FoldersToBeCreated:
		liste = []
		for y in paths_per_class.keys():
			z = Utils.create_folder(x.joinpath(y))
			liste.append(z)
		folders_per_set[str(posix(x)).split("/")[-1]] = liste 

	for i, folder in enumerate(folders_per_set.keys()):
		print(f'\nMoving files to {folder}')
		for input_paths, set_size in zip(
			paths_per_class.values(), set_size_per_class.values()):
			r = []
			length = 0			
			while length < set_size[i]:
				n = random.randint(0, len(input_paths) - 1)
				if n not in r :
					patch_name = ("_".join(str(posix(
						input_paths[n])).split("/")[-1].split("_")[0:5])
						)
					for index in range(len(input_paths)):
						name = ("_".join(str(posix(
							input_paths[index])).split("/")[-1].split("_")[0:5])
							)
						if name == patch_name:
							if (input_paths[index].exists() == True and 
								length < set_size[i]):
								src = input_paths[index]
								dst = (Path(
									"/".join([str(FoldersToBeCreated[i]), 
									"/".join(str(posix(src)).split('/')[1:])]))
									)
								shutil.move(src, dst)
								r.append(index)
								length += 1

	print(f"\nSplit complete in: {(time.time() - since) // 60:.2f} minutes")
	

if __name__ == '__main__':
	
	split(
		Patches = Config.args.Patches, 
		Test_Patches = Config.args.Test_Patches, 
		Test_Set_Size = Config.args.Test_Set_Size,
		Train_Patches = Config.args.Train_Patches,
		Validation_Patches = Config.args.Validation_Patches,  
		Validation_Set_Size = Config.args.Validation_Set_Size)