# -*- coding: UTF-8 -*-
#!/usr/bin/python

import openslide
from PIL import Image

def read_patch(input_path):
	i = 1
	compression_factor = 3
	img = openslide.OpenSlide(input_path)
	l = img.level_count
	print("The number of the levels in the slide : ", l)
	w,h = img.level_dimensions[0] # peut être remplacé par la ligne suivante w,h = img.dimensions
	#w,h = img.level_dimensions[1] 
	#w,h = img.level_dimensions[2] = img.level_dimensions[-1]
	#Le changement des niveaux affecte les dimensions de l'image et donc le nombre des patches générés.

	print("converting image.svs with width ", w," and height ", h)
	Window_size = 10000
	increment_x = int(w/Window_size) + 1
	increment_y = int(h/Window_size) + 1 
	#print(increment_x)
	#print(increment_y)

	for incre_x in range(increment_x):
		for incre_y in range(increment_y): 

			begin_x = incre_x * Window_size
			end_x = min(w, begin_x + Window_size) 
			begin_y = incre_y * Window_size
			end_y = min(h, begin_y + Window_size)
			patch_w = end_x - begin_x
			patch_h = end_y - begin_y
			print("N°: ",i)
			#print(incre_x,incre_y)
			print('begin x: ', begin_x, " end x: ", end_x)
			print('begin y: ', begin_y, " end y: ", end_y)
			print("Size of the patch : ",patch_w," * ", patch_h)

			patch = img.read_region((begin_x,begin_y),0,(patch_w,patch_h))
			#patch = img.read_region((begin_x,begin_y),1,(patch_w,patch_h))
			#patch = img.read_region((begin_x,begin_y),2,(patch_w,patch_h))
			
			patch_rgb = Image.new("RGB", patch.size,(255,255,255))
			patch_rgb.paste(patch) 

			patch_rgb = patch_rgb.resize((int(patch_w/compression_factor), int(patch_h/compression_factor)), Image.ANTIALIAS) 
			#patch_rgb = patch_rgb.resize((1000,1000)) 
			patch_rgb.save("/home/ala/Desktop/PFE/Output_Patches/image"+str(i)+".jpeg") 
			i += 1 
	return(i, w, h, increment_x, increment_y)