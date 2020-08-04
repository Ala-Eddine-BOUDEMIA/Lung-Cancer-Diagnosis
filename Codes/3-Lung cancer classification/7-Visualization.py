import csv
from itertools import islice
from pathlib import Path
from pathlib import PurePosixPath as posix

import torch
import torchvision                          
from torchvision import transforms   
from torchvision.utils import save_image

from PIL import Image
from gradcam import GradCAM
from gradcam.utils import visualize_cam

import Utils                
import Config
import Model_Utils


def viz(device, batch_size, csv_files, 
		path2weights, Train_folder, visualization):

	model = Model_Utils.create_model()
	
	try:
		model.load_state_dict(torch.load(path2weights))
	except:
		ckpt = torch.load(f = path2weights)
		model.load_state_dict(state_dict = ckpt["model_state_dict"])
	
	model.to(device)
	
	config = dict(model_type = 'resnet', arch = model, layer_name = 'layer4')
	gradcam = GradCAM.from_config(**config) 
	
	model.eval()
	
	paths_list, jpeg_list = Utils.parse_dir(Train_folder, "jpeg") 

	for file in jpeg_list:

		dataset_name = str(posix(file)).split("/")[-2]
		directory = Utils.create_folder(
			visualization.joinpath(Path('/'.join(["Patchs", dataset_name]))))

		jpeg_name = str(posix(file)).split("/")[-1]
		jpeg_img = Image.open(file)
		jpeg_image = transforms.Compose(
			[transforms.ToTensor()])(jpeg_img).to(device)

		normed_jpeg_img = transforms.Normalize(
			[0, 0, 0], [1, 1, 1])(jpeg_image)[None]
		
		mask, _ = gradcam(normed_jpeg_img)
		heatmap, result = visualize_cam(mask, jpeg_image)
			
		save_image(result, directory.joinpath(jpeg_name))


if __name__ == '__main__':

	viz(
		device = Config.device,
		csv_files = Config.args.CSV_files,
		batch_size = Config.args.batch_size, 
		path2weights = Config.args.Path2Weights,
		Train_folder = Config.args.Train_folder,
		visualization = Config.args.Visualization)