from xml.dom import minidom
import Code_from_deepslide
from pathlib import Path
from PIL import Image
import numpy as np
import openslide
import Config
import Utils
import csv

def generate_patches(patches, classes, all_wsi, overlap, 
	csv_files, annotations, Window_size, compression_factor):

	csv_files = Utils.create_folder(csv_files.joinpath("Annotations"))
	Utils.create_folder(patches)
	for classe in classes:
		Utils.create_folder(patches.joinpath(classe))

	xml_files = sorted([f for f in annotations.iterdir() if f.is_file()])
	path_list, wsi_paths = Utils.parse_dir(all_wsi, "svs")

	for file, wsi in zip(xml_files, wsi_paths):
		image_name = str(wsi).split("/")[-1][:-4]
		img = openslide.OpenSlide(str(wsi))

		with open(Path("/".join([str(csv_files), image_name + ".csv"])), "w") as w:
			writer = csv.writer(w, delimiter = "\t")
			writer.writerow(["Patch name", "W", "H", "Type", "Region ID",
							"Region begin X", "Region begin Y", "overlap factor"])

			xmldoc = minidom.parse(str(file))
			regions = xmldoc.getElementsByTagName('Region')
			for region in regions:
				X, Y = [], []
				Id =  region.attributes["Id"].value
				Text = region.attributes["Text"].value

				vertecies = region.getElementsByTagName('Vertex')
				for vertex in vertecies:
					X.append(int(vertex.getAttribute("X")))
					Y.append(int(vertex.getAttribute("Y")))

				crop_begin_x, crop_end_x = int(min(X)), int(max(X))
				crop_begin_y, crop_end_y = int(min(Y)), int(max(Y))
				crop_patch_w = crop_end_x - crop_begin_x
				crop_patch_h = crop_end_y - crop_begin_y 

				print(f"\nRegion Id: {Id} \t Type: {Text}")

				crop = img.read_region((crop_begin_x, crop_begin_y), 0, (crop_patch_w, crop_patch_h))
				w, h = crop.size 

				if w % Window_size == 0:
					increment_x = w // Window_size  
				else:
					increment_x = w // Window_size + 1
				if h % Window_size == 0:
					increment_y = h // Window_size  
				else:
					increment_y = h // Window_size + 1

				i = 1
				for incre_x in range(int(increment_x/overlap[Text])):
					for incre_y in range(int(increment_y/overlap[Text])): 

						begin_x = int(crop_begin_x + (incre_x * Window_size) * overlap[Text])
						end_x = min(crop_end_x, begin_x + Window_size) 
						begin_y = int(crop_begin_y + (incre_y * Window_size) * overlap[Text])
						end_y = min(crop_end_y, begin_y + Window_size)
						patch_w = end_x - begin_x
						patch_h = end_y - begin_y
						
						if patch_w != Window_size:
							begin_x = begin_x - (Window_size - patch_w) 
							patch_w = end_x - begin_x
						if patch_h != Window_size:
							begin_y = begin_y - (Window_size - patch_h)
							patch_h = end_y - begin_y
						
						print("\nN°: ",i)
						print('begin x: ', begin_x, " end x: ", end_x)
						print('begin y: ', begin_y, " end y: ", end_y)
						print("Size of the patch : ",patch_w," * ", patch_h)
						
						patch = img.read_region((begin_x, begin_y), 0, (patch_w, patch_h))

						patch_rgb = Image.new("RGB", patch.size, (255,255,255))
						patch_rgb.paste(patch) 
						patch_rgb = patch_rgb.resize((int(patch_w/compression_factor), 
									int(patch_h/compression_factor)), Image.ANTIALIAS) 
						
						patch_name = Path("_".join([image_name, str(incre_x).zfill(4), 
									str(incre_y).zfill(4), str(begin_x), str(begin_y)]) +'.tiff')

						class_path = Path("/".join([Text, str(patch_name)]))
						full_path = patches.joinpath(class_path)
						if Code_from_deepslide.is_purple(patch_rgb):
 							patch_rgb.save(full_path)
						i += 1

						writer.writerow([patch_name, w, h, Text, Id,
										crop_begin_x, crop_begin_y, overlap[Text]])

if __name__ == '__main__':
	generate_patches(
	patches = Config.args.Patches,
	classes = Config.args.Classes,
	all_wsi = Config.args.All_WSI,
	overlap = Config.args.Overlap,
	csv_files = Config.args.CSV_files,
	annotations = Config.args.Annotations,
	Window_size = Config.args.Window_size, 
	compression_factor = Config.args.Compression_factor)