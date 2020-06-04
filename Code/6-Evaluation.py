import Utils
import Config 
#############
import csv
import operator
###############
from itertools import islice
############################

def get_prediction(csv_files, predictions, output_folder, predictions_dir):

	predictions_cleaned = predictions_dir.joinpath("predictions_cleaned.csv")
	with open(predictions_cleaned, "w") as w:
		writer = csv.writer(w, delimiter = "\t")
		writer.writerow(["Name", "Prediction", "Confidence"])
		with open(predictions, "r") as preds:
			for preds_line in islice(preds, 1, None):
				patch_name, predicted, confidence = preds_line.split("\t")
				confidence = confidence.replace('\n','')
				patch_name = patch_name.split("_")
				if len(patch_name) == 5:
					writer.writerow(["_".join(patch_name), predicted, confidence])

	with open(predictions_cleaned, "r") as preds:
		wsi_names = []
		for preds_line in islice(preds, 1, None):
			wsi_name = preds_line.split("\t")[0].split("_")[0]
			if wsi_name not in wsi_names:
				wsi_names.append(wsi_name)

	for wsi_name in wsi_names:
		classe = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
		conf = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}	
		patches_number = 0

		with open(predictions_cleaned, "r") as preds:
			for preds_line in islice(preds, 1, None):
				patch_name, prediction, confidence = preds_line.split("\t")
				confidence = float(confidence.replace('\n',''))
				if wsi_name == patch_name.split("_")[0]:
					if confidence > 90:
						classe[prediction] += 1 
						conf[prediction] += confidence
						patches_number += 1

			sorted_classe = sorted(classe.items(), key = operator.itemgetter(1), reverse = True)
			print(f"\nThe WSI : {wsi_name}")
			for sc in sorted_classe:
				pourcentage = 100 * sc[1]/float(patches_number)
				confidence_total = conf[sc[0]]/(sc[1] + 1e-7)
				print(f"\nCancer subtype: {sc[0]}",
					f"\nRatio: {pourcentage}",
					f"\nConfidence: {confidence_total}")

if __name__ == '__main__':
	get_prediction(
	csv_files = Config.args.CSV_files, 
	predictions = Config.args.Predictions,
	output_folder = Config.args.Predictions,
	predictions_dir = Config.args.Predictions_Directory)