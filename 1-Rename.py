import os
from os import path
from pathlib import Path 
import shutil

def parse_dir():

	input_directories = Path('/home/ala/Desktop/PFE/Input_Images_WSI')
	path_list = [d for d in input_directories.iterdir() if d.is_dir()]
	num_dir = len(path_list)
	wsi_paths, annotation_paths = [], []

	for i in range(num_dir):
		input_directorie = path_list[i]
		wsi_path = sorted(list(input_directorie.glob('**/*.svs')))
		annotation_path = sorted(list(input_directorie.glob('**/*.txt')))
		wsi_paths.append(wsi_path[0])
		annotation_paths.append(annotation_path[0])
	
	return(input_directories, path_list, wsi_paths, annotation_paths)

def rename_all():

	(input_directories, path_list, wsi_paths, annotation_paths) = parse_dir()

	wsi_renamed = []
	annotation_renamed = []

	for i in range(len(wsi_paths)):
		wsi_target = Path("/home/ala/Desktop/PFE/Input_Images_WSI/image " + str(i) + ".svs")
		annotation_target = Path("/home/ala/Desktop/PFE/Input_Images_WSI/image " + str(i) + ".txt")

		if Path(wsi_target).exists() and (wsi_paths[i].stat().st_size) != (Path(wsi_target).stat().st_size): 
			wsi_list = [f for f in input_directories.iterdir() if f.is_file()]
			wsi_target = Path("/home/ala/Desktop/PFE/Input_Images_WSI/image " + str(int(len(wsi_list)/2) + i) + ".svs")
			annotation_target = Path("/home/ala/Desktop/PFE/Input_Images_WSI/image " + str(int(len(wsi_list)/2) + i) + ".txt")
			w = wsi_paths[i].rename(wsi_target)
			a = annotation_paths[i].rename(annotation_target)
		else:
			w = wsi_paths[i].rename(wsi_target)
			a = annotation_paths[i].rename(annotation_target)

		#wsi_renamed.append(w)
		#annotation_renamed.append(a)
		shutil.rmtree(path_list[i], ignore_errors=False, onerror=None)

	#return (wsi_renamed, annotation_renamed)

if __name__ == "__main__":

	rename_all()