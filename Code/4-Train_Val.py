from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR      
from torch.optim import lr_scheduler              
from torch import optim                                   
from torch import nn                        
import torchvision                          
import torch                                
import Model_Utils
import Config
import Utils
import pandas as pd                                                            
import numpy as np
import datetime
import time 
import copy
import csv

def train_val(device, classes, num_epochs, batch_size, loss_function, best_weights, weight_decay, path2weights, 
	sanity_check, learning_rate, save_interval, diagnostic_path, checkpoint_file, Train_Patches_path, 
	resume_checkpoint, checkpoints_folder, learning_rate_decay, Validation_Patches_path, diagnostics_directory):

	Utils.create_folder(diagnostics_directory)
	Utils.create_folder(best_weights)
	
	print(f"Training started at: {datetime.datetime.now()}")
	since = time.time()
	
	print("\nLoading training data ...")
	train_loader, train_set = Model_Utils.load_data(path = Train_Patches_path, shuffle = True, 
													batch_size = batch_size)

	print("\nLoading validation data ...")
	val_loader, val_set = Model_Utils.load_data(path = Validation_Patches_path, shuffle = False, 
													batch_size = batch_size, Train = False)
	print("\nCreating the model ...")
	model = Model_Utils.create_model()
	best_model = copy.deepcopy(model.state_dict())
	best_loss = float("inf")

	opt = optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay = weight_decay)
	scheduler = lr_scheduler.ExponentialLR(optimizer = opt, gamma = learning_rate_decay)
	
	train_images, train_labels = next(iter(train_loader))
	train_grid = torchvision.utils.make_grid(train_images)
	val_images, val_labels = next(iter(val_loader))
	val_grid = torchvision.utils.make_grid(val_images)
	
	tb_images = SummaryWriter("Tensorboard/Train_Val")
	tb_images.add_image("train_images", train_grid)
	tb_images.add_image("Validation_images", val_grid)
	tb_images.close()

	tb_model = SummaryWriter("Tensorboard/Model")
	tb_model.add_graph(model, train_images)
	tb_model.close()

	model.to(device)

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
		writer.writerow(["Date", "Epoch", "Batch size", "Train loss", 
						"Train accuracy", "Val loss", "Val accuracy"])

		train_tb_loss, val_tb_loss = 0.0, 0.0
		tb = SummaryWriter("Tensorboard/Model")

		for epoch in range(num_epochs):

			current_lr = Model_Utils.get_current_lr(opt)
			print('\nEpoch {}/{}, current lr={}'.format(epoch + 1, num_epochs, current_lr))

			model.train()
			train_running_loss, train_runing_metric = 0.0, 0.0
			train_all_labels, train_all_predictions = [], []

			for i, (inputs, labels) in enumerate(train_loader):

				train_inputs = inputs.to(device)
				train_labels = labels.to(device)

				with torch.set_grad_enabled(mode = True):

					train_outputs = model(train_inputs)
					__, predicted = torch.max(train_outputs.data, dim = 1)
					train_loss = loss_function(train_outputs, train_labels) 
					train_loss.backward()
					opt.step()

				corrects = predicted.eq(train_labels.view_as(predicted)).sum().item()
				train_running_loss += train_loss.item()

				train_tb_loss += train_loss.item()
				if i % 1000 == 999:
					tb.add_scalar("Training loss", train_tb_loss / 1000, epoch * len(train_loader) + i)
					train_tb_loss = 0.0
				
				if corrects is not None:
					train_runing_metric += corrects

				for x in predicted.cpu().numpy():
					train_all_labels.append(x)
				for x in train_labels.cpu().numpy():
					train_all_predictions.append(x)

			tb_metrics = SummaryWriter("Tensorboard/Train_Val")
			cm_train_heatmap, cm_train = Model_Utils.c_m(train_all_labels, train_all_predictions, classes)
			cr_train_heatmap, cr_train = Model_Utils.c_r(train_all_labels, train_all_predictions, classes)
			tb_metrics.add_figure("Train Confusion matrix epoch: " + str(epoch), cm_train_heatmap)
			tb_metrics.add_figure("Train Classification report epoch: " + str(epoch), cr_train_heatmap)
			np.savetxt(str(diagnostics_directory)+f"/cm_{epoch}_train.csv", cm_train, delimiter = '\t')
			cr_train.to_csv(str(diagnostics_directory)+f"/cr_{epoch}_train.csv", sep = '\t')
			
			train_len_data = len(train_set)
			train_loss = train_running_loss / float(train_len_data)
			train_metric = train_runing_metric / float(train_len_data)
			loss_history["train"].append(train_loss)
			metric_history["train"].append(train_metric)

			if torch.cuda.is_available():
				torch.cuda.empty_cache()

			model.eval()
			val_running_loss, val_runing_metric = 0.0, 0.0
			val_all_labels, val_all_predictions = [], []

			for i, (inputs, labels) in enumerate(val_loader):

				val_inputs = inputs.to(device)
				val_labels = labels.to(device)

				with torch.set_grad_enabled(mode = False):
					val_outputs = model(val_inputs)
					__, predicted = torch.max(val_outputs.data, dim = 1)
					val_loss = loss_function(val_outputs, val_labels) 
					
				corrects = predicted.eq(val_labels.view_as(predicted)).sum().item()
					
				if val_loss < best_loss:
					best_loss = val_loss
					best_model = copy.deepcopy(model.state_dict())
					torch.save(model.state_dict(), path2weights) 
					print("Copied best model weights")

				val_running_loss += val_loss.item()

				val_tb_loss += val_loss.item()
				if i % 1000 == 999:
					tb.add_scalar("Validation loss", val_tb_loss / 1000, epoch * len(val_loader) + i)
					val_tb_loss = 0.0

				if corrects is not None:
					val_runing_metric += corrects

				for x in predicted.detach().cpu().numpy():
					val_all_labels.append(x)
				for x in train_labels.detach().cpu().numpy():
					val_all_predictions.append(x)

				if sanity_check is True:
					break

			cm_val_heatmap, cm_val = Model_Utils.c_m(val_all_labels, val_all_predictions, classes)
			cr_val_heatmap, cr_val = Model_Utils.c_r(val_all_labels, val_all_predictions, classes)
			tb_metrics.add_figure("Validation Confusion matrix epoch: " + str(epoch), cm_val_heatmap)
			tb_metrics.add_figure("Validation Classification report epoch: " + str(epoch), cr_val_heatmap)			
			tb_metrics.close()
			np.savetxt(str(diagnostics_directory)+f"/cm_{epoch}_val.csv", cm_val, delimiter = '\t')
			cr_val.to_csv(str(diagnostics_directory)+f"/cr_{epoch}_val.csv", sep = '\t')

			val_len_data = len(val_set)
			val_loss = val_running_loss / float(val_len_data)
			val_metric = val_runing_metric / float(val_len_data)
			loss_history["val"].append(val_loss)
			metric_history["val"].append(val_metric)

			if torch.cuda.is_available():
            	torch.cuda.empty_cache()

			print("train loss: %.6f, val loss: %.6f, accuracy: %.2f"%(train_loss, val_loss, val_metric))
			
			scheduler.step()

			Model_Utils.save_work(epoch, save_interval, checkpoints_folder, model, opt, scheduler, val_metric)
			writer.writerow([datetime.datetime.now(), epoch+1, batch_size, 
							train_loss, train_metric, val_loss, val_metric])
		tb.close()
	
	print(f"\ntraining complete in: {(time.time() - since) // 60:.2f} minutes")

	model.load_state_dict(best_model)

	return model

if __name__ == '__main__':

	model = train_val(
	device = Config.device,
	classes = Config.args.Classes,
	num_epochs = Config.args.num_epochs, 
	batch_size = Config.args.batch_size, 
	loss_function = nn.CrossEntropyLoss(), 
	best_weights = Config.args.BestWeights,
	weight_decay = Config.args.weight_decay, 
	path2weights = Config.args.Path2Weights, 
	sanity_check = Config.args.Sanity_Check, 
	learning_rate = Config.args.learning_rate, 
	save_interval = Config.args.Save_interval, 
	diagnostic_path = Config.args.Diagnostics,
	checkpoint_file = Config.args.Checkpoint_file,
	Train_Patches_path = Config.args.Train_Patches, 
	resume_checkpoint = Config.args.Resume_checkpoint, 
	checkpoints_folder = Config.args.Checkpoints_folder, 
	learning_rate_decay = Config.args.learning_rate_decay, 
	Validation_Patches_path = Config.args.Validation_Patches,
	diagnostics_directory = Config.args.Diagnostics_Directory)