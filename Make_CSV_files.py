import csv
import Utils
import Config
from itertools import islice
############################

def Make_csv_labels_files():
	
	FoldersToBeCreated = [Config.args.Train_labels, Config.args.Validation_labels, Config.args.Test_labels]
	extraction_paths = [Config.args.Train_WSI, Config.args.Validation_WSI, Config.args.Test_WSI]

	for i in range (len(FoldersToBeCreated)):
		x = Utils.create_folder(FoldersToBeCreated[i])

		file_name = str(x)+"/"+str(x).split('/')[-1]+'.csv' 

		with open (file_name, "w", newline = "") as file:

			writer = csv.writer(file, delimiter = "\t")
			writer.writerow(["id", "submitter_id", "entity_type", "entity_id", "category",
							"classification", "created_datetime", "status", "notes", "image_name"])

			path_list, wsi_paths, annotation_paths = Utils.parse_dir(extraction_paths[i])
			for i in range(len(annotation_paths)):
				image_name = str(wsi_paths[i]).split("/")[-1][:-4]
				with open(annotation_paths[i]) as f:
					for line in islice(f, 1, None): 
						l = line.split("\t")
						l.append(image_name)
						writer.writerow(l)