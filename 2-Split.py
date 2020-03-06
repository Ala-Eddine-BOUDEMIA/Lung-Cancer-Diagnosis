import os
from os import path
from pathlib import Path 
import shutil
import random

def create_subfolders():
	
	for i in ['Val_set/', 'Test_set/', 'Train_set']:
		path = '/home/ala/Desktop/PFE/Input_Images_WSI/' + i
		if Path(path).exists() == False:
			os.mkdir(path)		

def parser():

	input_directorie = Path('/home/ala/Desktop/PFE/Input_Images_WSI')
	wsi_paths = sorted(list(input_directorie.glob('**/*.svs')))
	annotation_paths = sorted(list(input_directorie.glob('**/*.txt')))

	return wsi_paths, annotation_paths

def randomize (num_wsi, wsi_paths):

	r = []
	while len(r) < num_wsi :
		n = random.randint(0,len(wsi_paths)-1)
		if n not in r :
			if Path('/home/ala/Desktop/PFE/Input_Images_WSI/image '+str(n)+'.svs').exists() == True:
				r.append(n)

	return r

def create_set(wsi_paths, annotation_paths, set_type):

	num_wsi_val = 2
	num_wsi_test = 2
	num_wsi_train = len(wsi_paths) - num_wsi_val - num_wsi_test

	if set_type == "Val_set":
		r = randomize(num_wsi_val, wsi_paths)
	elif set_type == "Test_set":
		r = randomize(num_wsi_test, wsi_paths)
	else :
		r = randomize(num_wsi_train, wsi_paths)

	for i in range(len(r)):
		src_annotation = str(annotation_paths[r[i]])
		src_wsi = str(wsi_paths[r[i]])
		print(src_wsi, src_annotation)
		dst_annotation = Path('/home/ala/Desktop/PFE/Input_Images_WSI/'+set_type+'/image'+str(r[i])+'.txt')
		dst_wsi = Path('/home/ala/Desktop/PFE/Input_Images_WSI/'+set_type+'/')
		shutil.move(src_annotation,dst_annotation)
		shutil.move(src_wsi,dst_wsi)

if __name__ == '__main__':
	
	create_subfolders()
	wsi_paths, annotation_paths = parser()
	for i in ['Val_set', 'Test_set', 'Train_set']:
		create_set(wsi_paths, annotation_paths, set_type = i)