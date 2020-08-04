import csv
import operator
from itertools import islice

import Utils
import Config 


def get_prediction(csv_files, predictions, predictions_dir):

	predictions_cleaned = predictions_dir.joinpath("predictions_cleaned.csv")
	with open(predictions_cleaned, "w") as w:
		writer = csv.writer(w, delimiter = "\t")
		writer.writerow(["Name", "Prediction", "Confidence"])
		
		with open(predictions, "r") as preds:
			wsi_names = []
			for preds_line in islice(preds, 0, None):
				if preds_line.split('\t') != ['\n']:
					patch_name, predicted, confidence = preds_line.split("\t")
					wsi_name = patch_name.split("_")[0]
					if wsi_name not in wsi_names:
						wsi_names.append(wsi_name)
			
					confidence = confidence.replace('\n','')
					patch_name = patch_name.split("_")
					if len(patch_name) == 5:
						writer.writerow(["_".join(patch_name), 
							predicted, confidence])
	
	for wsi_name in wsi_names:
		classe = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
		conf = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}	
		classe_str = {"0": "ACINAR", "1": "CRIB", "2": "MICROPAP", 
					"3": "NC", "4": "SOLID"}
		patches_number = 0

		with open(predictions_cleaned, "r") as preds:
			for preds_line in islice(preds, 1, None):
				if preds_line.split('\t') != ['\n']:
					patch_name, prediction, confidence = preds_line.split("\t")
					confidence = float(confidence.replace('\n',''))
			
				if wsi_name == patch_name.split("_")[0] and confidence >= 0.0:
					classe[prediction] += 1 
					conf[prediction] += confidence
					patches_number += 1

			sorted_classe = sorted(
				classe.items(), key = operator.itemgetter(1), reverse = True)
			
			with open(predictions_dir.joinpath(wsi_name + ".csv"), "w") as w:
				writer = csv.writer(w, delimiter = "\t")
				writer.writerow(["Subtype", "Patches", 
					"Total patches", "Ratio", "Confidence"])
				for sc in sorted_classe:
					try:
						pourcentage = 100 * sc[1]/(float(patches_number))
						confidence_total = conf[sc[0]]/(sc[1])
					except:
						pourcentage = 0
						confidence_total = 0
				
					if pourcentage > 0:
						print(f"\nThe WSI : {wsi_name}",
							f"\nCancer subtype: {classe_str[sc[0]]}",
							f"\nRatio: {sc[1]} / {patches_number}",
							f" = {pourcentage} ",
							f"\nConfidence: {confidence_total}")

						writer.writerow([classe_str[sc[0]], sc[1], 
							patches_number, pourcentage, confidence_total])


if __name__ == '__main__':
	
	get_prediction(
		csv_files = Config.args.CSV_files, 
		predictions = Config.args.Predictions,
		predictions_dir = Config.args.Predictions_Directory)