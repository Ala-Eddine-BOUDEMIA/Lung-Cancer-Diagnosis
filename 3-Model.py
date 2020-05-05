###########
import Utils                
import Config
import Compute_mean_std
import Code_from_deepslide
##########################
import csv
import time 
import copy
import pandas as pd                                                            
###################
from matplotlib import pyplot as plt
####################################  
import torch                                
import torchvision                          
from torch import nn                        
from torch import optim                        
from torchvision import models            
from torchsummary import summary
from torchvision import datasets            
from torchvision import transforms                      
from torch.optim import lr_scheduler              
from torch.optim.lr_scheduler import ExponentialLR      
##################################################

def get_data_transforms(Train, Train_Patches = Config.args.Train_Patches, Validation_Patches = Config.args.Validation_Patches):

	if Train:
		mean, std = Compute_mean_std.compute_mean_std(Train_Patches)
		data_transforms = transforms.Compose(transforms = [ 
			transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Code_from_deepslide.Random90Rotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)])
	else :
		mean, std = Compute_mean_std.compute_mean_std(Validation_Patches)
		data_transforms = transforms.Compose(transforms = [ 
			transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)])

	return data_transforms

def load_data(path, shuffle, Train = True, batch_size = Config.args.batch_size):

	images_dataset = datasets.ImageFolder(root = str(path), transform = get_data_transforms(Train)) 
	dataloaders = torch.utils.data.DataLoader(dataset = images_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 8)

	return dataloaders, images_dataset

def create_model(device = Config.device):
                                                
    model = models.resnet18(pretrained = True)  
    num_ftrs = model.fc.in_features             
    model.fc = nn.Linear(num_ftrs, 6)              
    model_summary = summary(model, (3,224,224))
    model.to(device)

    return model

def get_current_lr(opt):

	current_lr = None
	for group in opt.param_groups:
		current_lr = group["lr"]

	return current_lr  

def train_val(num_epochs = Config.args.num_epochs, device = Config.device,
	sanity_check = Config.args.Sanity_Check, loss_function = nn.CrossEntropyLoss(), 
	weight_decay = Config.args.weight_decay, path2weights = Config.args.Path2Weights, 
	checkpoints_folder = Config.args.Checkpoints_folder, save_interval = Config.args.Save_interval, 
	resume_checkpoint = Config.args.Resume_checkpoint, checkpoint_file =Config.args.Checkpoint_file,
	learning_rate = Config.args.learning_rate, learning_rate_decay = Config.args.learning_rate_decay, 
	Train_Patches_path = Config.args.Train_Patches, Validation_Patches_path = Config.args.Validation_Patches):

	since = time.time()

	train_loader, train_set = load_data(path = Train_Patches_path, shuffle = True)
	val_loader, val_set = load_data(path = Validation_Patches_path, shuffle = False, Train = False)

	model = create_model()
	best_model = copy.deepcopy(model.state_dict())
	best_loss = float("inf")

	opt = optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay = weight_decay)
	scheduler = lr_scheduler.ExponentialLR(optimizer = opt, gamma = learning_rate_decay)

	if resume_checkpoint:
		ckpt = torch.load(f = checkpoint_file)
		model.load_state_dict(state_dict = ckpt["model_state_dict"])
		opt.load_state_dict(state_dict = ckpt["optimizer_state_dict"])
		scheduler.load_state_dict(state_dict = ckpt["scheduler_state_dict"])
		start_epoch = ckpt["epoch"]
		print("Model loaded from: ", checkpoint_file)
	else:
		start_epoch = 0
	
	loss_history = {"train": [], "val": []}
	metric_history = {"train": [], "val": []}

	for epoch in range(num_epochs):

		current_lr = get_current_lr(opt)
		print('Epoch {}/{}, current lr={}'.format(epoch + 1, num_epochs, current_lr))

		model.train()
		train_running_loss = 0.0
		train_runing_metric = 0.0
		train_all_labels = []
		train_all_predictions = []

		for i, (inputs, labels) in enumerate(train_loader):

			train_inputs = inputs.to(device)
			train_labels = labels.to(device)

			train_outputs = model(train_inputs)
			__, predicted = torch.max(train_outputs.data, dim = 1)
			corrects = predicted.eq(train_labels.view_as(predicted)).sum().item()
			
			train_loss = loss_function(train_outputs, train_labels) 

			opt.zero_grad()
			train_loss.backward()
			opt.step()

			train_running_loss += train_loss # in deepslide they multiplied it by train_inputs.size(0)

			if corrects is not None:
				train_runing_metric += corrects

			for x in predicted.numpy():
				train_all_labels.append(x)
			for x in train_labels.numpy():
				train_all_predictions.append(x)

		Code_from_deepslide.calculate_confusion_matrix(train_all_labels, train_all_predictions)
		
		train_len_data = len(train_set)
		train_loss = train_running_loss / float(train_len_data)
		train_metric = train_runing_metric / float(train_len_data)
		loss_history["train"].append(train_loss)
		metric_history["train"].append(train_metric)

		model.eval()
		val_running_loss = 0.0
		val_runing_metric = 0.0
		val_all_labels = []
		val_all_predictions = []

		for i, (inputs, labels) in enumerate(val_loader):

			val_inputs = inputs.to(device)
			val_labels = labels.to(device)

			with torch.no_grad():
				val_outputs = model(val_inputs)
				val_loss = loss_function(val_outputs, val_labels) 
				__, predicted = torch.max(val_outputs.data, dim = 1)
				corrects = predicted.eq(val_labels.view_as(predicted)).sum().item()

				if val_loss < best_loss:
					best_loss = val_loss
					best_model = copy.deepcopy(model.state_dict())
					torch.save(model.state_dict(), path2weights) 
					print("Copied best model weights")

			val_running_loss += val_loss # in deepslide they multiplied it by val_inputs.size(0)

			if corrects is not None:
				val_runing_metric += corrects

			for x in predicted.numpy():
				val_all_labels.append(x)
			for x in train_labels.numpy():
				val_all_predictions.append(x)

			if sanity_check is True:
				break

		Code_from_deepslide.calculate_confusion_matrix(val_all_labels, val_all_predictions)

		val_len_data = len(val_set)
		val_loss = val_running_loss / float(val_len_data)
		val_metric = val_runing_metric / float(val_len_data)
		loss_history["val"].append(val_loss)
		metric_history["val"].append(val_metric)

		scheduler.step()

		if epoch % save_interval == 0:
			output_path = checkpoints_folder.joinpath(f"resnet18_e{epoch}_val{val_metric:.5f}.pt")
			Utils.create_folder(checkpoints_folder)

			torch.save(obj = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1}, f = str(output_path))

		print("train loss: %.6f, val loss: %.6f, accuracy: %.2f"%(train_loss, val_loss, 100*val_metric))
	
	print(f"\ntraining complete in " f"{(time.time() - since) // 60:.2f} minutes")

	model.load_state_dict(best_model)

	return model, loss_history, metric_history

def predict(path2weights = Config.args.Path2Weights, Test_Patches_path = Config.args.Test_Patches, device = Config.device):
	
	model = create_model()
	model.load_state_dict(torch.load(path2weights))
	
	model.eval()

	path_list, jpeg_paths = Utils.parse_dir(input_directories = Test_Patches_path, forme = "jpeg")

	for path in path_list:
		test_loader, test_set = load_data(path = str(path), shuffle = False, batch_size = 1)
		test_len_data = len(test_set)
		predictions = str(path) + "/predictions.csv"

		with open(predictions, "w") as w:
			writer = csv.writer(w, delimiter = "\t")
			writer.writerow(["Image name", "Prediction", "Confidence"])
			for i, (inputs, labels) in enumerate(test_loader, 0):
				
				test_inputs = inputs.to(device)
				test_labels = labels.to(device)

				test_outputs = model(test_inputs)
				_, predicted = torch.max(test_outputs.data, 1)
				confidence = torch.max(nn.Softmax(dim = 1)(test_outputs))
				
				#https://stackoverflow.com/questions/56699048/how-to-get-the-filename-of-a-sample-from-a-dataloader	
				sample_name, _ = test_loader.dataset.samples[i]
				sample_name = sample_name.split("/")[-1][:-5]
				
				writer.writerow([sample_name, predicted.item(), confidence.item()*100])

def plot_graphs(loss_history, metric_history, num_epochs = Config.args.num_epochs):

	plt.title("Train-Val Loss")
	plt.plot(range(1, num_epochs + 1), loss_history["train"], label = "train")
	plt.plot(range(1, num_epochs + 1), loss_history["val"], label = "val")
	plt.ylabel("Loss")
	plt.xlabel("Training Epochs")
	plt.legend()
	plt.show()

	plt.title("Train-Val Accuracy")
	plt.plot(range(1,num_epochs + 1), metric_history["train"], label = "train")
	plt.plot(range(1,num_epochs + 1), metric_history["val"], label = "val")
	plt.ylabel("Accuracy")
	plt.xlabel("Training Epochs")
	plt.legend()
	plt.show()

if __name__ == '__main__':

    loss_history, metric_history = train_val()
    plot_graphs(loss_history, metric_history)
    predict()