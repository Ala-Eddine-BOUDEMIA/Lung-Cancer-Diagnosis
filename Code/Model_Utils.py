import Utils
############
import numpy as np
import pandas as pd
import seaborn as sns
#####################
import torch
from torch import nn 
from torchvision import models 
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms                      
##################################
from matplotlib import pyplot as plt
####################################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#################################################
from torch.utils.tensorboard import SummaryWriter
#################################################

def create_model():
                                                
    model = models.resnet18(pretrained = True)  
    num_ftrs = model.fc.in_features             
    model.fc = nn.Linear(num_ftrs, 5)              
    model_summary = summary(model, (3,224,224))

    return model

def get_data_transforms(Train, path):

	if Train:
		data_transforms = transforms.Compose(transforms = [ 
			transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.2),
            transforms.ToTensor()])
	else :
		data_transforms = transforms.Compose(transforms = [ 
			transforms.ToTensor()])

	return data_transforms

def load_data(path, shuffle, batch_size, Train = True):

	images_dataset = datasets.ImageFolder(root = str(path), 
										transform = get_data_transforms(Train, path)) 
	dataloaders = torch.utils.data.DataLoader(dataset = images_dataset, batch_size = batch_size, 
											shuffle = shuffle, num_workers = 8)

	return dataloaders, images_dataset

def get_current_lr(opt):
	
	current_lr = None
	for group in opt.param_groups:
		current_lr = group["lr"]

	return current_lr

def save_work(epoch, save_interval, checkpoints_folder, model, opt, scheduler, val_metric):

	if epoch % save_interval == 0:
		output_path = checkpoints_folder.joinpath(f"resnet18_e{epoch}_val{val_metric:.5f}.pt")
		Utils.create_folder(checkpoints_folder)
		print(f"Saving work to: {output_path}")

		torch.save(obj = {
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": opt.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"epoch": epoch + 1}, f = str(output_path))

def c_m(actual, predicted, classes):

	remap_classes = {}
	for i in range(len(classes)):
		remap_classes[i] = classes[i]

	actual = np.array(pd.Series(actual).replace(remap_classes))
	predicted = np.array(pd.Series(predicted).replace(remap_classes))

	cm = confusion_matrix(actual, predicted, labels = classes)

	f1, ax1 = plt.subplots(1,1)
	sns.heatmap(cm, annot = True, cmap = "YlGnBu")
	ax1.set_xlabel('Predicted labels')
	ax1.set_ylabel('True labels')
	ax1.set_title('Confusion Matrix')

	return f1, cm

def c_r(actual, predicted, classes):

	remap_classes = {}
	for i in range(len(classes)):
		remap_classes[i] = classes[i]

	actual = np.array(pd.Series(actual).replace(remap_classes))
	predicted = np.array(pd.Series(predicted).replace(remap_classes))
	
	cr = classification_report(actual, predicted, labels = classes, target_names = classes, output_dict = True)
	cr = pd.DataFrame(cr)
	f2, ax2 = plt.subplots(1,1)
	ax2 = sns.heatmap(cr, annot = True, cmap = "YlGnBu")
	ax2.set_title('Classification report')

	return f2, cr

def pr_curve(class_index, test_probs, test_preds, classes, global_step = 0):

    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    tb_pr = SummaryWriter("Tensorboard")
    tb_pr.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step = global_step)
    tb_pr.close()