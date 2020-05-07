import Utils
import Compute_mean_std
import Code_from_deepslide
##########################
import torch
from torch import nn 
from torchvision import models 
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms                      
################################
from matplotlib import pyplot as plt
#################################### 

def create_model():
                                                
    model = models.resnet18(pretrained = True)  
    num_ftrs = model.fc.in_features             
    model.fc = nn.Linear(num_ftrs, 6)              
    model_summary = summary(model, (3,224,224))

    return model

def get_data_transforms(Train, path):

	if Train:
		mean, std = Compute_mean_std.compute_mean_std(path)
		data_transforms = transforms.Compose(transforms = [ 
			transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Code_from_deepslide.Random90Rotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)])
	else :
		mean, std = Compute_mean_std.compute_mean_std(path)
		data_transforms = transforms.Compose(transforms = [ 
			transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)])

	return data_transforms

def load_data(path, shuffle, batch_size, Train = True):

	images_dataset = datasets.ImageFolder(root = str(path), transform = get_data_transforms(Train, path)) 
	dataloaders = torch.utils.data.DataLoader(dataset = images_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 8)

	return dataloaders, images_dataset

def get_current_lr(opt):
	
	current_lr = None
	for group in opt.param_groups:
		current_lr = group["lr"]

	return current_lr

def plot_graphs(loss_history, metric_history, num_epochs):

	plt.title("Train-Val Loss")
	plt.plot(range(1, num_epochs + 1), loss_history['train'], label = 'train')
	plt.plot(range(1, num_epochs + 1), loss_history['val'], label = 'val')
	plt.ylabel("Loss")
	plt.xlabel("Epochs")
	plt.legend()
	plt.show()

	plt.title("Train-Val Accuracy")
	plt.plot(range(1,num_epochs + 1), metric_history['train'], label = 'train')
	plt.plot(range(1,num_epochs + 1), metric_history['val'], label = 'val')
	plt.ylabel("Accuracy")
	plt.xlabel("Epochs")
	plt.legend()
	plt.show()

def save_work(epoch, save_interval, checkpoints_folder, model, opt, scheduler, val_metric):

	if epoch % save_interval == 0:
		output_path = checkpoints_folder.joinpath(f"resnet18_e{epoch}_val{val_metric:.5f}.pt")
		Utils.create_folder(checkpoints_folder)

		torch.save(obj = {
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": opt.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"epoch": epoch + 1}, f = str(output_path))