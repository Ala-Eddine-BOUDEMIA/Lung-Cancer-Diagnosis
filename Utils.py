import os
import csv
from os import path
from pathlib import Path 
from itertools import islice 

def create_folder(output_folder):

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	return output_folder

def parse_dir(input_directories):

	path_list = [d for d in input_directories.iterdir() if d.is_dir()]
	num_dir = len(path_list)

	wsi_paths, annotation_paths = [], []

	for i in range(num_dir):

		input_directorie = path_list[i]
		wsi_path = sorted(list(input_directorie.glob('**/*.svs')))
		annotation_path = sorted(list(input_directorie.glob('**/*.txt')))
		if wsi_path != []:
			wsi_paths.append(wsi_path[0])
		if annotation_path != []:
			annotation_paths.append(annotation_path[0])

	return(path_list, wsi_paths, annotation_paths)