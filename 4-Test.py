import Utils                
import Config
import Load_Data
import Create_Model
import Code_from_deepslide
##########################
import numpy
############
import torch
############

def predict(path2weights = Config.args.Path2Weights, Test_Patches_path = Config.args.Test_Patches, device = Config.device):
	
	model = Create_Model.create_model()
	model.load_state_dict(torch.load(path2weights))
	
	model.eval()

	test_loader, test_set = Load_Data.load_data(path = Test_Patches_path, shuffle = False, Train = False)
	test_len_data = len(test_set)

	test_all_labels, test_all_predictions = [], []
	test_running_metric = 0.0

	for i, (inputs, labels) in enumerate(test_loader):
		
		test_inputs = inputs.to(device)
		test_labels = labels.to(device)

		test_outputs = model(test_inputs)
		_, predicted = torch.max(test_outputs.data, 1)

		for x in predicted.numpy():
			test_all_labels.append(x)
		for x in test_labels.numpy():
			test_all_predictions.append(x)

	Code_from_deepslide.calculate_confusion_matrix(test_all_labels, test_all_predictions)

if __name__ == '__main__':

    predict()