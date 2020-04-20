###########
import time                 
import Config
import random                                           
import numpy as np          
import pandas as pd         
from PIL import Image        
##########################
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

def Train(weight_decay = Config.args.weight_decay, loss_func = nn.CrossEntropyLoss(),
        lr = Config.args.learning_rate,  learning_rate_decay = Config.args.learning_rate_decay, 
        num_epochs = Config.args.num_epochs, batch_size = Config.args.batch_size): 

    data_transforms = transforms.Compose(transforms=[transforms.ToTensor()])
    image_datasets = datasets.ImageFolder(root = str(Config.args.Train_Patches), transform = data_transforms)  
    train_loader = torch.utils.data.DataLoader(dataset = image_datasets, batch_size = batch_size, shuffle = True)

    model = create_model()    
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = learning_rate_decay) 

    for epoch in range(num_epochs):

        model.train(mode = True)

        train_running_loss = 0.0
        correct_outputs  = 0
        
        for i, (train_inputs, labels) in enumerate(train_loader):
            
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            train_loss = loss_func(train_outputs, labels)
            train_loss.backward()
            optimizer.step()

            train_running_loss += train_loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, train_running_loss / 100))
                train_running_loss = 0.0

        scheduler.step()

if __name__ == '__main__':
    Train()