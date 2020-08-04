import random

from pathlib import Path
from pathlib import PurePosixPath as posix

import PIL
from PIL import Image

import Utils
import Config


def resize_patches(patches, classes, all_dataset):

	Utils.create_folder(patches)
	for classe in classes:
		Utils.create_folder(patches.joinpath(classe))

	path_list, jpeg_paths = Utils.parse_dir(all_dataset, "jpeg")

	for p in path_list:
		path_list, jpeg_paths = Utils.parse_dir(p, "jpeg")

		for p in path_list:
			directory_name = str(posix(p)).split("/")[-1]
			files = [f for f in p.iterdir() if f.is_file()]

			for image in files:
				
				image_name = str(posix(image)).split("/")[-1]
				img = Image.open(image)
				patch = img.resize((224,224))
				patch_rgb = Image.new("RGB", patch.size, (255,255,255))
				patch_rgb.paste(patch)

				full_path = patches.joinpath(
					directory_name).joinpath(image_name)
				patch_rgb.save(full_path)
						
						
if __name__ == '__main__':
	
	resize_patches(
		patches = Config.args.Patches,
		classes = Config.args.Classes,
		all_dataset = Config.args.All_Dataset)