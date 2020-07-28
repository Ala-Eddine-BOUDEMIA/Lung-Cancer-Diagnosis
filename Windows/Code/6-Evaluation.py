from itertools import islice
import operator
import Config 
import Utils
import csv

def get_prediction(csv_files, predictions, output_folder, predictions_dir):

	predictions_cleaned = predictions_dir.joinpath("predictions_cleaned.csv")
	with open(predictions_cleaned, "w") as w:
		writer = csv.writer(w, delimiter = "\t")
		writer.writerow(["Name", "Prediction", "Confidence"])
		
		with open(predictions, "r") as preds:
			wsi_names = []
			
			for preds_line in islice(preds, 1, None):
				
				if preds_line.split('\t') != ['\n']:
					patch_name, predicted, confidence = preds_line.split("\t")
					wsi_name = patch_name.split("_")[0]
				
					if wsi_name not in wsi_names:
						wsi_names.append(wsi_name)
					confidence = confidence.replace('\n','')
					patch_name = patch_name.split("_")
				
					if len(patch_name) == 5:
						writer.writerow(["_".join(patch_name), predicted, confidence])
	
	for wsi_name in wsi_names:
		classe = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
		conf = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}	
		classe_str = {"0": "ACINAR", "1": "CRIB", "2": "MICROPAP", "3": "NC", "4": "SOLID"}
		patches_number = 0

		with open(predictions_cleaned, "r") as preds:
			
			for preds_line in islice(preds, 1, None):
				
				if preds_line.split('\t') != ['\n']:
					patch_name, prediction, confidence = preds_line.split("\t")
					confidence = float(confidence.replace('\n',''))
					
					if wsi_name == patch_name.split("_")[0] and confidence >= 00:
							classe[prediction] += 1 
							conf[prediction] += confidence
							patches_number += 1

			sorted_classe = sorted(classe.items(), key = operator.itemgetter(1), reverse = True)
			for sc in sorted_classe:
				pourcentage = 100 * sc[1]/(float(patches_number) + 1e-7)
				confidence_total = conf[sc[0]]/(sc[1] + 1e-7)
				
				if pourcentage > 0.0:
					print(f"\nThe WSI : {wsi_name}",
						f"\nCancer subtype: {classe_str[sc[0]]}",
						f"\nRatio: {pourcentage}",
						f"\nConfidence: {confidence_total}")

					with open(predictions_dir.joinpath(wsi_name + ".csv"), "w") as w:
						writer = csv.writer(w, delimiter = "\t")
						writer.writerow(["Subtype", "Ratio", "Confidence"])
						writer.writerow([classe_str[sc[0]], pourcentage, confidence_total])

if __name__ == '__main__':
	get_prediction(
		csv_files = Config.args.CSV_files, 
		predictions = Config.args.Predictions,
		output_folder = Config.args.Predictions,
		predictions_dir = Config.args.Predictions_Directory)