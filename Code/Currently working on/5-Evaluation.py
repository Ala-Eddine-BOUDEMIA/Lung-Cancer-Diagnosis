import Utils
import Config 
#############
import csv
import operator
###############
from itertools import islice
############################

def get_prediction(Train_folder, output_folder):

	path_list, csv_paths = Utils.parse_dir(Train_folder, "csv")
	
	for csv_path in csv_paths:

		if str(csv_path).split("/")[-2] == 'Test_patches':
			image_names_csv_file = csv_path
		elif str(csv_path).split("/")[-2] == 'Predictions':
			predictions_csv_file = csv_path

	with open(image_names_csv_file, "r") as names:
		with open(predictions_csv_file, "r") as preds:
			for name_line in islice(names, 0, None):
				classe = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
				conf = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
				wsi_name, w, h, patches_number, increment_x, increment_y = name_line.split("\t")
				with open(str(output_folder) + "/" + wsi_name + ".csv", "w") as w:
					writer = csv.writer(w, delimiter = '\t')
					#writer.writerow(["name", "prediction", "confidence"])
					for preds_line in islice(preds, 0, None):
						patch_name, prediction, confidence = preds_line.split("\t")
						confidence = confidence.replace("\n","")
						if patch_name.split("_")[0] == wsi_name:
							classe[prediction] += 1 
							conf[prediction] += float(confidence)
							#writer.writerow([patch_name, prediction, confidence])
					#writer.writerow("\n")
					sorted_classe = sorted(classe.items(), key = operator.itemgetter(1), reverse = True)
					writer.writerow(["Classe", "Ratio", "Confidence"])
					for sc in sorted_classe:
						pourcentage = 100 * sc[1]/float(patches_number)
						confidence_total = conf[sc[0]]/(sc[1] + 1e-7)
						if confidence_total > 50:
							writer.writerow([sc[0], pourcentage, confidence_total])
							print(f"The WSI : {wsi_name}",
									f"\nCancer subtype: {sc[0]}",
									f"\nRatio: {pourcentage}",
									f"\nConfidence: {confidence_total}")

if __name__ == '__main__':
	get_prediction(
	Train_folder = Config.args.Train_folder, 
	output_folder = Config.args.Predictions)