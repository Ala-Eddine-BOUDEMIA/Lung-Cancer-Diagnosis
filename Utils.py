import Imports 
##############

def create_folder(output_folder):

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	return output_folder

def parse_dir(input_directories, forme):

	path_list = sorted([d for d in input_directories.iterdir() if d.is_dir()])
	paths = []

	for i in range(len(path_list)):
		input_directorie = path_list[i]
		path = sorted(list(input_directorie.glob('**/*.' + forme)))
		
		for j in range(len(path)):
			paths.append(path[j])

	return(path_list, paths)

def Make_csv_labels_files(extraction_paths):
	
	FoldersToBeCreated = [Config.args.Validation_labels, Config.args.Test_labels, Config.args.Train_labels]

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