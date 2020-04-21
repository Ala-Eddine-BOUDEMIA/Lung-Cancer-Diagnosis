###########
import copy
import time                 
import Config                                                            
#############
from matplotlib import pyplt as plt
###################################
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

def load_data(path, batch_size, shuffle):

	data_transforms = transforms.Compose(transforms = [transforms.ToTensor()])

	images_dataset = datasets.ImageFolder(root = str(path), transform = data_transforms) 
	dataloaders = torch.utils.data.DataLoader(dataset = images_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 8)

	return dataloaders

def create_model(device):
                                                
    model = models.resnet18(pretrained = True)  
    num_ftrs = model.fc.in_features             
    model.fc = nn.Linear(num_ftrs, 6)              
    model_summary = summary(model, (3,224,224))
    model.to(device)

    return model

def optimizer(model, learning_rate = Config.args.learning_rate, 
	learning_rate_decay = Config.args.learning_rate_decay, 
	weight_decay = Config.args.weight_decay):

	opt = optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay = weight_decay)
	scheduler = lr_scheduler.ExponentialLR(optimizer = opt, gamma = learning_rate_decay)

	current_lr = None
	for group in opt.param_groups:
		current_lr = group["lr"]

	return opt, scheduler, current_lr

def metrics_batch(output, target):
	
	predicted = torch.argmax(output, dim = 1, keepdim = True)
	corrects = predicted.eq(target.view_as(predicted)).sum().item()

	return corrects 

def loss_batch(output, target, optimizer, loss_function = nn.CrossEntropyLoss()):

 	loss = loss_function(output, target) 
 	metric_b = metrics_batch(output, target)

 	if optimizer is None:
 		optimizer.zero_grad()
 		loss.backward()
 		optimizer.step()

 	return loss.item(), metric_b   

def loss_epoch(model, device, loader, optimizer, sanity_check):

	running_loss = 0.0
	runing_metric = 0.0
	
	for i, (inputs, labels) in enumerate(loader):

		inputs = inputs.to(device)
		labels = labels.to(device)

		outputs = model(inputs)
		loss_b, metric_b = loss_batch(outputs, labels, optimizer)
		running_loss += loss_b

		if metric_b is not None:
			runing_metric += metric_b

		if sanity_check is True:
			break

	len_data = len(loader)
	loss = running_loss / float(len_data)
	metric = runing_metric / float(len_data)

	return loss, metric

def Train_Val(num_epochs = Config.args.num_epochs, batch_size = Config.args.batch_size, 
	path2weights = Config.args.Path2Weights, Train_Patches_path = Config.args.Train_Patches, 
	Validation_Patches_path = Config.args.Validation_Patches, sanity_check = Config.args.Sanity_Check):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	train_loader = load_data(path = Train_Patches_path, batch_size = batch_size, shuffle = True)
	val_loader = load_data(path = Validation_Patches_path, batch_size = batch_size, shuffle = False)

	model = create_model(device)
	best_model = copy.deepcopy(model.state_dict())
	best_loss = float("inf")

	opt, scheduler, current_lr = optimizer(model)

	loss_history = {"train": [], "val": []}
	metric_history = {"train": [], "val": []}


	for epoch in range(num_epochs):

		print('Epoch {}/{}, current lr={}'.format(epoch + 1, num_epochs, current_lr))
		
		model.train()
		train_loss, train_metric = loss_epoch(model, device, train_loader, opt, sanity_check = sanity_check)
		loss_history["train"].append(train_loss)
		metric_history["train"].append(train_metric)

		model.eval()
		with torch.no_grad():
			val_loss, val_metric = loss_epoch(model, device, val_loader, opt, sanity_check = sanity_check)
			loss_history["val"].append(val_loss)
			metric_history["val"].append(val_metric)

			if val_loss < best_loss:
				best_loss = val_loss
				best_model = copy.deepcopy(model.state_dict())
				torch.save(model.state_dict(), path2weights) 
				print("Copied best model weights")

		scheduler.step()

		print("train loss: %.6f, val loss: %.6f, accuracy: %.2f"%(train_loss, val_loss, 100*val_metric))

	model.load_state_dict(best_model)

	return model, loss_history, metric_history

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
    model, loss_history, metric_history = Train_Val()
    plot_graphs(loss_history, metric_history)