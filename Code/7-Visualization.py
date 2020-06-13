from torchvision.utils import save_image
from torchvision import transforms   
import torchvision                          
import torch
from gradcam.utils import visualize_cam
from gradcam import GradCAM
from itertools import islice
from PIL import Image
import Model_Utils
import Config
import Utils                
import numpy as np
import csv

def viz(device, batch_size, csv_files, path2weights, Train_folder, visualization):

	model = Model_Utils.create_model()
	model.load_state_dict(torch.load(path2weights))
	config = dict(model_type = 'resnet', arch = model, layer_name = 'layer4')
	gradcam = GradCAM.from_config(**config) 
	model.eval()
	
	files = sorted([f for f in (csv_files.joinpath('Annotations')).iterdir() if f.is_file()])
	
	for file in files:
		wsi_name = str(file).split("/")[-1][:-4]
		directory = Utils.create_folder(visualization.joinpath('/'.join(["Patchs", wsi_name])))
		
		with open(file, "r") as reader:
			for line in islice(reader, 1, None):
				patch_name = line.split("\t")[0]

				paths_list, tiff_list = Utils.parse_dir(Train_folder, "tiff") 
				for tiff in tiff_list:
					tiff_name = str(tiff).split('/')[-1]
					if tiff_name == patch_name:
						tiff_img = Image.open(tiff)
						tiff_image = transforms.Compose([transforms.ToTensor()])(tiff_img).to(device)
						normed_tiff_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(tiff_image)[None]
						
						mask, _ = gradcam(normed_tiff_img)
						heatmap, result = visualize_cam(mask, tiff_image)
							
						save_image(result, directory.joinpath(tiff_name))

if __name__ == '__main__':

	viz(
		device = Config.device,
		csv_files = Config.args.CSV_files,
	    batch_size = Config.args.batch_size, 
	    path2weights = Config.args.Path2Weights,
		Train_folder = Config.args.Train_folder,
		visualization = Config.args.Visualization)