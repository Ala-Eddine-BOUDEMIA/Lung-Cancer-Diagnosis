import Utils                
import Config
import Model_Utils
import Code_from_deepslide
##########################
import torch
############

def test(batch_size = Config.args.batch_size, device = Config.device,
	path2weights = Config.args.Path2Weights, Test_Patches_path = Config.args.Test_Patches):
	
	model = Model_Utils.create_model()
	model.load_state_dict(torch.load(path2weights))
	
	model.eval()

	test_loader, test_set = Model_Utils.load_data(path = Test_Patches_path, shuffle = False, batch_size = batch_size, Train = False)
	test_len_data = len(test_set)

	test_all_labels, test_all_predictions = [], []
	test_running_metric = 0.0

	for i, (inputs, labels) in enumerate(test_loader):
		
		test_inputs = inputs.to(device)
		test_labels = labels.to(device)

		test_outputs = model(test_inputs)
		__, predicted = torch.max(test_outputs.data, 1)
		corrects = (predicted == test_labels).sum().item()

		if corrects is not None:
			test_running_metric += corrects

		for x in predicted.numpy():
			test_all_labels.append(x)
		for x in test_labels.numpy():
			test_all_predictions.append(x)

	Code_from_deepslide.calculate_confusion_matrix(test_all_labels, test_all_predictions)

	print(f"Accuracy of the network on the {test_len_data} test images: {100 * test_running_metric / test_len_data}")

if __name__ == '__main__':

    test()