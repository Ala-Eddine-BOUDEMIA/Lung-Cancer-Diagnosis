###########
import time                 
import Config
import random                                           
import numpy as np          
import pandas as pd         
from PIL import Image        
#####################
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
                                            
def calculate_confusion_matrix(all_labels, all_predicts):                     
                                                  
    classes = ["lepidic", "acinar", "papillary", "micropapillary", "solid", "normal"]       
    num_classes = len(classes)                                    
    remap_classes = {}  

    for i in range(len(classes)):    

        remap_classes[i] = classes[i]                               
                                                        
        pd.options.display.float_format = "{:.2f}".format                         
        pd.options.display.width = 0                                  
                                                      
        actual = pd.Series(data = pd.Series(all_labels).replace(remap_classes), name="Actual")      
        predicted = pd.Series(data = pd.Series(all_predicts).replace(remap_classes), name="Predicted")  
                                                        
        confusion_matrix = pd.crosstab(index=actual, columns=predicted, normalize="index")        
        print(confusion_matrix)                                                                                         

def create_model():                             
                                                
    model = models.resnet18(pretrained = True)  
    num_ftrs = model.fc.in_features             
    model.fc = nn.Linear(num_ftrs, 6)              
    model_summary = summary(model, (3,224,224))

    return model                                                                     

def Train_Val(weight_decay = Config.args.weight_decay, loss_func = nn.CrossEntropyLoss(),
        lr = Config.args.learning_rate,  learning_rate_decay = Config.args.learning_rate_decay, 
        num_epochs = Config.args.num_epochs, batch_size = Config.args.batch_size): 
	
	since = time.time()

    data_transforms = transforms.Compose(transforms=[transforms.ToTensor()])
    train_image_datasets = datasets.ImageFolder(root = str(Config.args.Train_Patches), transform = data_transforms)  
    train_loader = torch.utils.data.DataLoader(dataset = train_image_datasets, batch_size = batch_size, shuffle = True)
    val_image_datasets = datasets.ImageFolder(root = str(Config.args.Validation_Patches), transform = data_transforms)
    val_loader = torch.utils.data.DataLoader(dataset = val_image_datasets, batch_size = batch_size, shuffle = False)

    model = create_model()    
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = learning_rate_decay) 

    for epoch in range(num_epochs):

        model.train(mode = True)

        train_running_loss = 0.0
        train_corrects  = 0
	        
	    for i, (train_inputs, train_labels) in enumerate(train_loader):
	            
	            optimizer.zero_grad()

	            with torch.no_grad():

	            	train_outputs = model(train_inputs)
	            	_, predicted = torch.max(train_outputs.data, 1)
		            train_loss = loss_func(train_outputs, train_labels)
		            train_loss.backward()
		            optimizer.step()
			
			train_corrects += (predicted == train_labels).sum().item()
            train_running_loss += train_loss.item() * train_inputs.size(0)

			start = i * batch_size
            end = start + batch_size

			train_all_labels[start:end] = train_labels
            train_all_predicts[start:end] = predicted

			calculate_confusion_matrix(all_labels = train_all_labels.numpy(), all_predicts = train_all_predicts.numpy())

			train_loss = train_running_loss / len(train_image_datasets)
        	train_acc = train_corrects / len(train_image_datasets)

        val_running_loss = 0.0
        val_corrects = 0

        for i, (val_inputs, val_labels) in enumerate(val_loader):

            with torch.no_grad():
                
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_loss = loss_func(val_outputs, val_labels)

            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += (predicted == labels).sum().item()

            start = i * batch_size
            end = start + batch_size

            val_all_labels[start:end] = val_labels
            val_all_predicts[start:end] = val_preds

        calculate_confusion_matrix(all_labels = val_all_labels.numpy(), all_predicts = val_all_predicts.numpy())

        val_loss = val_running_loss / len(val_image_datasets)
        val_acc = val_running_corrects / len(val_image_datasets) 
        
        scheduler.step()
		
		current_lr = None
        for group in optimizer.param_groups:
            current_lr = group["lr"]
            
        print(f"Epoch {epoch} with lr "
              f"{current_lr:.15f}: "
              f"t_loss: {train_loss:.4f} "
              f"t_acc: {train_acc:.4f} "
              f"v_loss: {val_loss:.4f} "
              f"v_acc: {val_acc:.4f} ")

    print(f"\ntraining complete in "
          f"{(time.time() - since) // 60:.2f} minutes")

if __name__ == '__main__':
    Train_Val()