import csv
from itertools import islice

import Utils
import Config 


def get_prediction(predictions, predictions_dir):

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
					if len(patch_name) == 1:
						writer.writerow([patch_name, predicted, confidence])


if __name__ == '__main__':
	
	get_prediction(
		predictions = Config.args.Predictions,
		predictions_dir = Config.args.Predictions_Directory)