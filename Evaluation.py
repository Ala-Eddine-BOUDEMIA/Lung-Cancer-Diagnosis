import Utils
import Config 
#############
from itertools import islice
############################

def get_prediction(Tets_Patches_path = Config.args.Test_Patches):

	path_list, csv_paths = Utils.parse_dir(Tets_Patches_path, "csv")
	for csv_path in csv_paths:
		wsi_name = str(csv_path).split("/")[-2]
		with open(csv_path, "r") as prediction_file:
			len_list = 0
			preds = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
			for line in islice(prediction_file, 1, None):
				line_items = line.split("\t")
				class_predicted = line_items[1]
				confidence = line_items[2].replace("\n", "")
				if confidence > "10":
					preds[class_predicted] += 1
				len_list +=1
		
		for i, classe in enumerate(preds):

			with open(str(Config.args.Predictions) + "/" + wsi_name + ".csv", "a") as writer:
				writer.write(f"classe : {classe} - pourcentage : {preds[classe]/len_list * 100}\n")

if __name__ == '__main__':
	get_prediction()