from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F   
from torch import nn    
import torchvision                          
import torch
import Model_Utils
import Config
import Utils                
import numpy as np
import csv

def test(device, classes, batch_size, path2weights, prediction_file, 
	Test_Patches_path, predictions_directory, diagnostics_directory):

	Utils.create_folder(predictions_directory)

	model = Model_Utils.create_model()
	model.load_state_dict(torch.load(path2weights, map_location=torch.device('cpu')))
	model.eval()

	print("\nLoading testing data ...")
	test_loader, test_set = Model_Utils.load_data(path = Test_Patches_path, shuffle = False, 
												batch_size = batch_size, Train = False)
	test_len_data = len(test_set)

	tb_images = SummaryWriter("Tensorboard/Test")
	test_images, test_labels = next(iter(test_loader))
	test_grid = torchvision.utils.make_grid(test_images)
	tb_images.add_image("test_images", test_grid)
	tb_images.close()

	test_all_labels, test_all_predictions, class_probs, class_preds = [], [], [], [] 
	names, preds, confidence_stats = [], [], []
	test_running_metric, confidence_running_metric = 0.0, 0.0

	model.to(device)

	for i, (inputs, labels) in enumerate(test_loader):

		test_inputs = inputs.to(device)
		test_labels = labels.to(device)

		start = i * len(inputs)
		end = start + len(labels)

		for j in range(start, end):
			sample_name, _ = test_loader.dataset.samples[j]
			sample_name = sample_name.split("/")[-1][:-5]
			names.append(sample_name)

		test_outputs = model(test_inputs)
		__, predicted = torch.max(test_outputs.data, 1)
		corrects = (predicted == test_labels).sum().item()
		test_running_metric += corrects
		confidences = [F.softmax(el, dim=0) for el in test_outputs]

		for p in predicted:
			preds.append(p.item())

		for confidence in confidences:
			confidence = torch.max(confidence).item()
			confidence_running_metric += confidence
			confidence_stats.append(confidence)

		class_probs.append(confidences)
		class_preds.append(predicted)
		test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
		test_preds = torch.cat(class_preds)
		
		for x in predicted.detach().cpu().numpy():
			test_all_predictions.append(x)
		for x in test_labels.detach().cpu().numpy():
			test_all_labels.append(x)

	for i in range(len(classes)):
		Model_Utils.pr_curve(i, test_probs, test_preds, classes)

	tb_metrics = SummaryWriter("Tensorboard/Test")
	cm_test_heatmap, cm_test = Model_Utils.c_m(np.array(test_all_labels), np.array(test_all_predictions), classes)
	cr_test_heatmap, cr_test = Model_Utils.c_r(np.array(test_all_labels), np.array(test_all_predictions), classes)
	tb_metrics.add_figure("Test Confusion matrix: ", cm_test_heatmap)
	tb_metrics.add_figure("Test Classification report: ", cr_test_heatmap)	
	tb_metrics.close()

	np.savetxt(str(diagnostics_directory)+f"/cm_test.csv", cm_test, delimiter = '\t')
	cr_test.to_csv(str(diagnostics_directory)+f"/cr_test.csv", sep = '\t')
	
	with open(prediction_file, "w") as f:
		writer = csv.writer(f, delimiter = "\t")
		writer.writerow(["Patch name", "Prediction", "Confidence"])
		for x in range(0, test_len_data):
			writer.writerow([names[x], preds[x], f"{100*confidence_stats[x]:.6}"])

	print(f"Accuracy of the network on the: {test_len_data}",
		f" test images: {100 * test_running_metric / test_len_data}\n"
		f"Averge confidence of the model on the: {test_len_data}",
		f" test images: {100 * confidence_running_metric / test_len_data}\n")

if __name__ == '__main__':

	test(
	device = Config.device,
	classes = Config.args.Classes, 
	batch_size = Config.args.batch_size, 
	path2weights = Config.args.Path2Weights,
	prediction_file = Config.args.Predictions, 
	Test_Patches_path = Config.args.Test_Patches, 
	predictions_directory = Config.args.Predictions_Directory, 
	diagnostics_directory = Config.args.Diagnostics_Directory)