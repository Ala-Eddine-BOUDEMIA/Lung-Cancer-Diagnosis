from PIL.ImageOps import flip, mirror
from PIL import ImageFilter
from pathlib import Path
from PIL import Image
import PIL
import random
import Config
import Utils

def transforms(key, image):

	if key == "RandomNoise": 
		transformed_image = image.filter(ImageFilter.GaussianBlur(1))
	elif key == "RandomRotation": 
		transformed_image = image.rotate(random.choice([90, 180, 270]))
	elif key == "flip": 
		transformed_image = flip(image)
	elif key == "mirror": 
		transformed_image = mirror(image)

	return transformed_image 

def augmentor(patches, classes, maximum):

	transformations = ["RandomNoise", "RandomRotation", "flip", "mirror"]

	directories = sorted([d for d in patches.iterdir() if d.is_dir()])

	all_paths = {}
	for directory in directories:
		images_paths = sorted([f for f in directory.iterdir() if f.is_file()])
		all_paths[str(directory).split("\\")[-1]] = images_paths

	for subclass in all_paths.keys():
		num_images_generated = 0
		num_images_desired = maximum - len(all_paths[subclass])
		paths = all_paths[subclass]
		print(f"\nSubtype: {subclass}, \t Num images to be generated: {num_images_desired}")
		print("Generating new images ...")

		while num_images_generated < num_images_desired:
			path = random.choice(paths)
			image_name = str(path).split("\\")[-1][:-5]

			rep = 0
			for transform in transformations:
				if transform in image_name.split("_"):
					rep += 1

			if rep == 0:
				image_to_transform = Image.open(path)
				num_transforms = 0
				num_transforms_to_apply = random.randint(1, len(transformations))
				transformed_image = PIL.Image.new("RGB", image_to_transform.size, (255,255,255))
				transformed_image.paste(image_to_transform)

				keys = []
				while num_transforms < num_transforms_to_apply:
					key = random.choice(transformations)
					while key not in keys:
						keys.append(key)
						transformed_image = transforms(key, transformed_image)
						num_transforms += 1

				new_path = Path('/'.join([str(patches), str(subclass),
					image_name + "_" + "_".join(sorted(keys))]) + ".tiff")

				if new_path.exists() == False:
					transformed_image.save(new_path)
					num_images_generated +=1

if __name__ == '__main__':
	augmentor(
	patches = Config.args.Patches,
	classes = Config.args.Classes,
	maximum = Config.args.Maximum)