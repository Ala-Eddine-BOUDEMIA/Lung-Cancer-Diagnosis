import Imports
##############

def split():
	
	path_list, wsi_paths = Utils.parse_dir(Imports.Config.args.All_WSI, "svs")

	FoldersToBeCreated = [Imports.Config.args.Validation_WSI, 
						Imports.Config.args.Test_WSI, 
						Imports.Config.args.Train_WSI]

	for x in FoldersToBeCreated:
		Utils.create_folder(x)

	Val_set_size = Imports.Config.args.Validation_WSI_Size
	Test_set_size = Imports.Config.args.Test_WSI_Size
	Train_set_size = len(path_list) - Val_set_size - Test_set_size

	r = []
	for (dataset, set_size) in ((FoldersToBeCreated[0], Val_set_size), 
								(FoldersToBeCreated[1], Test_set_size), 
								(FoldersToBeCreated[2], Train_set_size)):
		length = 0
		while length < set_size :
			n = random.randint(0,len(path_list)-1)
			if n not in r :
				if path_list[n].exists() == True:
					src = str(path_list[n])
					dst = Path(dataset)
					shutil.move(src, dst)
					r.append(n)
					length += 1

	#Utils.Make_csv_labels_files(FoldersToBeCreated)