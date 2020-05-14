import Utils                
import Config
#############
from PIL import Image
#####################
from itertools import islice
############################

def repiece(Window_size, Predictions, Train_folder, Test_patches, Visualize_folder, compression_factor):

	Utils.create_folder(Visualize_folder)

	a, b, x, y, w_tot, h_tot = 0, 0, 0, 0, 0, 0

	path_list, csv_paths = Utils.parse_dir(Train_folder, "csv")
	
	for csv_path in csv_paths:

		if str(csv_path).split("/")[-2] == 'Test_patches':
			image_names_csv_file = csv_path
		elif str(csv_path).split("/")[-2] == 'Predictions':
			predictions_csv_file = csv_path

	with open(image_names_csv_file, "r") as names:
		with open(predictions_csv_file, "r") as preds:
			for name in islice(names, 1, None):
				wsi_name, W, H, patches_number, increment_x, increment_y = name.split("\t")
				new_im = Image.new('RGB', (int(W)//compression_factor, int(H)//compression_factor))
				for prediction in islice(preds, 1, None):
					patch_name, prediction, confidence = prediction.split("\t")
					patch, incre_x, incre_y, begin_x, begin_y = patch_name.split("_")
					if patch == wsi_name:
						image = Image.open(Test_patches.joinpath(patch_name + ".jpeg"))
						w_tot += image.size[0]
						h_tot += image.size[1] 
						new_im.paste(image, box = (int(begin_x)//compression_factor, int(begin_y)//compression_factor))
				
				img = Image.new('RGB', (w_tot//(int(increment_x)+1), h_tot//(int(increment_y)+1)))
				img.paste(new_im)
				img = img.resize((img.size[0]//compression_factor, img.size[1]//compression_factor), Image.ANTIALIAS)
				img.save(Visualize_folder.joinpath(wsi_name + ".jpeg"))
		
if __name__ == '__main__':
	repiece(
	Window_size = Config.args.Window_size,
	Predictions = Config.args.Predictions, 
	Train_folder = Config.args.Train_folder, 
	Test_patches = Config.args.Test_Patches,
	Visualize_folder = Config.args.Visualize, 
	compression_factor = Config.args.Compression_factor)