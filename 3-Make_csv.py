import csv
from pathlib import Path 
from itertools import islice 

input_directorie = Path('/home/ala/Desktop/PFE/Input_Images_WSI/Train_set/')	
annotation_paths = sorted(list(input_directorie.glob('**/*.txt')))

with open ("/home/ala/Desktop/PFE/Input_Images_WSI/Train_set/Train_set.csv", "w", newline = "") as file:
	writer = csv.writer(file, delimiter = "\t")
	writer.writerow(["id", "submitter_id", "entity_type", "entity_id", "category",
						"classification", "created_datetime", "status", "notes"])
	
	for i in annotation_paths:
		with open(i) as fin:
			for line in islice(fin, 1, None): 
				l = line.split("\t")
				writer.writerow(l)