import Utils
import Config
import Code_from_deepslide
##########################
import os
import csv
import openslide
import numpy as np
from PIL import Image
#####################

def generate_patches(Train_WSI = Config.args.Train_WSI, Validation_WSI = Config.args.Validation_WSI,
	Test_WSI = Config.args.Test_WSI, Train_Patches = Config.args.Train_Patches, Test_Patches = Config.args.Test_Patches,
	Validation_Patches = Config.args.Validation_Patches, Window_size = Config.args.Window_size, 
	compression_factor = Config.args.Compression_factor):

	WSI_sets = [Train_WSI, Validation_WSI, Test_WSI]
	output_folders = [Train_Patches, Validation_Patches, Test_Patches]

	for k in range(len(WSI_sets)):	

		path_list, wsi_paths = Utils.parse_dir(input_directories = WSI_sets[k], forme = "svs")
		Utils.create_folder(output_folders[k])
		
		for i in range(len(wsi_paths)): 

			image_name = str(wsi_paths[i]).split("/")[-1][:-4] 

			img = openslide.OpenSlide(str(wsi_paths[i]))
			l = img.level_count
			print("The number of the levels in the slide : ", l)
			
			w,h = img.level_dimensions[0] 
			print("converting image.svs with width ", w," and height ", h)
			if w % Window_size == 0:
				increment_x = w // Window_size  
			else:
				increment_x = w // Window_size + 1

			if h % Window_size == 0:
				increment_y = h // Window_size  
			else:
				increment_y = h // Window_size + 1

			i, j = 1, 0
			for incre_x in range(increment_x):
				for incre_y in range(increment_y): 

					begin_x = incre_x * Window_size 
					end_x = min(w, begin_x + Window_size) 
					begin_y = incre_y * Window_size
					end_y = min(h, begin_y + Window_size)
					patch_w = end_x - begin_x
					patch_h = end_y - begin_y

					print("N°: ",i)
					print('begin x: ', begin_x, " end x: ", end_x)
					print('begin y: ', begin_y, " end y: ", end_y)
					print("Size of the patch : ",patch_w," * ", patch_h)

					patch = img.read_region((begin_x,begin_y),0,(patch_w,patch_h))
					
					patch_rgb = Image.new("RGB", patch.size,(255,255,255))
					patch_rgb.paste(patch) 
					patch_rgb = patch_rgb.resize((int(patch_w/compression_factor), int(patch_h/compression_factor)), Image.ANTIALIAS) 
					
					patch_name = image_name + "_" + str(incre_x).zfill(4) + "_" + str(incre_y).zfill(4) + '.jpeg'
					path = str(output_folders[k]) + "/" + patch_name
					if k != 2 :
						if Code_from_deepslide.is_purple(patch_rgb) == True :
							patch_rgb.save(path) 
							j += 1 
					elif k == 2:
						patch_rgb.save(path) 
						j += 1

					i += 1

			with open(str(output_folders[k]) + "/" + "images names.csv", "a") as f:
				writer = csv.writer(f, delimiter = "\t")
				writer.writerow([image_name, str(j)])

if __name__ == '__main__':
	generate_patches()