import os
import Utils
import Config
import openslide
from PIL import Image
#####################

def generate_patches():

	Window_size = 224

	WSI_sets = [Config.args.Train_WSI, Config.args.Validation_WSI, Config.args.Test_WSI]
	output_folders = [Config.args.Train_Patches, Config.args.Validation_Patches, Config.args.Test_Patches]

	for k in range(len(WSI_sets)):	

		path_list, wsi_paths, annotation_paths = Utils.parse_dir(input_directories = WSI_sets[k])
		Utils.create_folder(output_folders[k])
		
		for i in range(len(wsi_paths)): 

			image_name = str(wsi_paths[i]).split("/")[-1][:-4]  

			img = openslide.OpenSlide(str(wsi_paths[i]))
			l = img.level_count
			print("The number of the levels in the slide : ", l)
			
			w,h = img.level_dimensions[0] 
			print("converting image.svs with width ", w," and height ", h)
			increment_x = w // Window_size + 1
			increment_y = h // Window_size + 1

			i = 1
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
					patch_rgb = patch_rgb.resize((int(patch_w), int(patch_h)), Image.ANTIALIAS) 
					patch_rgb.save(str(output_folders[k]) + "/" + image_name.split("/")[-1][:-4] + "_" + str(incre_x) + "_" + str(incre_y) + '.jpeg') 
					
					i += 1 