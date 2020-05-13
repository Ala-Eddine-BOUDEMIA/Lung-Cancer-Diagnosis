import Utils
import Config
import Model_Utils
import Code_from_deepslide
##########################
import csv
import time 
import copy
import datetime
import pandas as pd                                                            
################### 
import torch                                
import torchvision                          
from torch import nn                        
from torch import optim                                   
from torch.optim import lr_scheduler              
from torch.optim.lr_scheduler import ExponentialLR      
################################################## 
import sys
def train_val(num_epochs = Config.args.num_epochs, device = Config.device,
	sanity_check = Config.args.Sanity_Check, loss_function = nn.CrossEntropyLoss(), 
	batch_size = Config.args.batch_size, diagnostic_path = Config.args.Diagnostics,
	weight_decay = Config.args.weight_decay, path2weights = Config.args.Path2Weights, 
	checkpoints_folder = Config.args.Checkpoints_folder, save_interval = Config.args.Save_interval, 
	resume_checkpoint = Config.args.Resume_checkpoint, checkpoint_file =Config.args.Checkpoint_file,
	learning_rate = Config.args.learning_rate, learning_rate_decay = Config.args.learning_rate_decay, 
	Train_Patches_path = Config.args.Train_Patches, Validation_Patches_path = Config.args.Validation_Patches,
	diagnostics_directory = Config.args.Diagnostics_Directory):

	Utils.create_folder(diagnostics_directory)
	
	print(f"Training started at: {datetime.datetime.now()}")
	since = time.time()
	
	print("\nLoading training data ...")
	train_loader, train_set = Model_Utils.load_data(path = Train_Patches_path, shuffle = True, batch_size = batch_size)
	print("\nLoading validation data ...")
	val_loader, val_set = Model_Utils.load_data(path = Validation_Patches_path, shuffle = False, batch_size = batch_size, Train = False)

	print("\nCreating the model ...")
	model = Model_Utils.create_model()
	model.to(device)
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

	with open(diagnostic_path, 'w') as file:
		writer = csv.writer(file, delimiter = '\t')
		writer.writerow(["Date", "Epoch", "Batch size", "Train loss", "Train accuracy", "Val loss", "Val accuracy"])
		for epoch in range(num_epochs):

			current_lr = Model_Utils.get_current_lr(opt)
			print('Epoch {}/{}, current lr={}'.format(epoch + 1, num_epochs, current_lr))

			model.train()
			train_running_loss, train_runing_metric = 0.0, 0.0
			train_all_labels, train_all_predictions = [], []

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

				train_running_loss += train_loss.item() # in deepslide they multiplied it by train_inputs.size(0)

				if corrects is not None:
					train_runing_metric += corrects

				for x in predicted.numpy():
					train_all_labels.append(x)
				for x in train_labels.numpy():
					train_all_predictions.append(x)

			cm = Code_from_deepslide.calculate_confusion_matrix(train_all_labels, train_all_predictions)
			cm.to_csv(str(diagnostics_path)+f"/cm_{epoch}_train.csv", sep='\t')
			
			train_len_data = len(train_set)
			train_loss = train_running_loss / float(train_len_data)
			train_metric = train_runing_metric / float(train_len_data)
			loss_history["train"].append(train_loss)
			metric_history["train"].append(train_metric)

			model.eval()
			val_running_loss, val_runing_metric = 0.0, 0.0
			val_all_labels, val_all_predictions = [], []

			for i, (inputs, labels) in enumerate(val_loader):

				val_inputs = inputs.to(device)
				val_labels = labels.to(device)

				with torch.no_grad():
					val_outputs = model(val_inputs)
					__, predicted = torch.max(val_outputs.data, dim = 1)
					corrects = predicted.eq(val_labels.view_as(predicted)).sum().item()
					
					val_loss = loss_function(val_outputs, val_labels) 

					if val_loss < best_loss:
						best_loss = val_loss
						best_model = copy.deepcopy(model.state_dict())
						torch.save(model.state_dict(), path2weights) 
						print("Copied best model weights")

				val_running_loss += val_loss.item() # in deepslide they multiplied it by val_inputs.size(0)

				if corrects is not None:
					val_runing_metric += corrects

				for x in predicted.numpy():
					val_all_labels.append(x)
				for x in train_labels.numpy():
					val_all_predictions.append(x)

				if sanity_check is True:
					break

			cm = Code_from_deepslide.calculate_confusion_matrix(val_all_labels, val_all_predictions)
			cm.to_csv(str(diagnostics_path)+f"/cm_{epoch}_val.csv", sep='\t')

			val_len_data = len(val_set)
			val_loss = val_running_loss / float(val_len_data)
			val_metric = val_runing_metric / float(val_len_data)
			loss_history["val"].append(val_loss)
			metric_history["val"].append(val_metric)

			print("train loss: %.6f, val loss: %.6f, accuracy: %.2f"%(train_loss, val_loss, 100*val_metric))
			
			scheduler.step()

			Model_Utils.save_work(epoch, save_interval, checkpoints_folder, model, opt, scheduler, val_metric)
			writer.writerow([datetime.datetime.now(), epoch+1, batch_size, train_loss, train_metric, val_loss, val_metric])
	
	print(f"\ntraining complete in: {(time.time() - since) // 60:.2f} minutes")

	model.load_state_dict(best_model)

	Model_Utils.plot_graphs(loss_history, metric_history, num_epochs)

	return model

if __name__ == '__main__':

    model = train_val()