import Utils                
import Config
import Model_Utils
import Code_from_deepslide
##########################
import csv
##########
import torch
from torch import nn 
####################

def test(batch_size = Config.args.batch_size, device = Config.device, path2weights = Config.args.Path2Weights,
	predictions_directory = Config.args.Predictions_Directory, Test_Patches_path = Config.args.Test_Patches, 
	prediction_file = Config.args.Predictions, diagnostics_directory = Config.args.Diagnostics_Directory):

	Utils.create_folder(predictions_directory)
	
	model = Model_Utils.create_model()
	model.load_state_dict(torch.load(path2weights))
	
	model.eval()

	print("\nLoading testing data ...")
	test_loader, test_set = Model_Utils.load_data(path = Test_Patches_path, shuffle = False, batch_size = batch_size, Train = False)
	test_len_data = len(test_set)

	test_all_labels, test_all_predictions, names, confidence_stats, preds = [], [], [], [], []
	test_running_metric, confidence_running_metric = 0.0, 0.0

	for i, (inputs, labels) in enumerate(test_loader):
		
		test_inputs = inputs.to(device)
		test_labels = labels.to(device)

		start = i * batch_size 
		end = start + batch_size

		for j in range(start, end):
			sample_name, _ = test_loader.dataset.samples[j]
			sample_name = sample_name.split("/")[-1][:-5]
			names.append(sample_name)

		test_outputs = model(test_inputs)
		__, predicted = torch.max(test_outputs.data, 1)
		corrects = (predicted == test_labels).sum().item()
		test_running_metric += corrects
		confidences = nn.Softmax(dim = 1)(test_outputs)

		for p in predicted:
			preds.append(p.item())

		for confidence in confidences:
			confidence = torch.max(confidence).item()
			confidence_running_metric += confidence
			confidence_stats.append(confidence)
		
		for x in predicted.numpy():
			test_all_labels.append(x)
		for x in test_labels.numpy():
			test_all_predictions.append(x)
	
	with open(prediction_file, "w") as f:
		writer = csv.writer(f, delimiter = "\t")
		writer.writerow(["Patch name", "Prediction", "Confidence"])
		for x in range(0, test_len_data):
			writer.writerow([names[x], preds[x], f"{100*confidence_stats[x]:.6}"])

	cm = Code_from_deepslide.calculate_confusion_matrix(test_all_labels, test_all_predictions)
	cm.to_csv(str(diagnostics_directory)+f"/cm_test.csv", sep='\t')

	print(f"Accuracy of the network on the {test_len_data} test images: {100 * test_running_metric / test_len_data}\n"
		f"Averge confidence of the model on the {test_len_data} test images: {100 * confidence_running_metric / test_len_data}\n")

if __name__ == '__main__':

    test()