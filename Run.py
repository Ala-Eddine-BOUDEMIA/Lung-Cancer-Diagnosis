import Model
import Split
import Processing
#################

def Main():

	print("__________Run.py__________")
	ans = 0
	while ans < 1 or ans > 4:
		
		print("1: Split.py \n2: Processing.py \n3: Model.py \n4: Quit \n")
		ans = input("Enter your choice :\n")
		
		if ans == 1:
			Split.split()

		elif ans == 2:
			Processing.generate_patches()

		elif ans == 3:
			print("__________Model.py__________")
			x = " "

			while x != 'a' or x != "b":

				print("a: train_val \nb: predict")
				x = input("Enter your choice :\n")

				if x == "a":
					loss_history, metric_history = Model.train_val(
						device = Imports.Config.device,
						num_epochs = Imports.Config.args.num_epochs, 
						loss_function = Imports.nn.CrossEntropyLoss(), 
						weight_decay = Imports.Config.args.weight_decay, 
						path2weights = Imports.Config.args.Path2Weights, 
						sanity_check = Imports.Config.args.Sanity_Check, 
						save_interval = Imports.Config.args.Save_interval, 
						learning_rate = Imports.Config.args.learning_rate, 
						checkpoint_file = Imports.Config.args.Checkpoint_file,
						Train_Patches_path = Imports.Config.args.Train_Patches, 
						resume_checkpoint = Imports.Config.args.Resume_checkpoint, 
						checkpoints_folder = Imports.Config.args.Checkpoints_folder, 
						learning_rate_decay = Imports.Config.args.learning_rate_decay, 
						Validation_Patches_path = Imports.Config.args.Validation_Patches)

					plot_graphs(loss_history, metric_history, Imports.Config.args.num_epochs)

				elif x == "b":
					Model.predict(
						device = Imports.Config.device,
						path2weights = Imports.Config.args.Path2Weights, 
						Test_Patches_path = Imports.Config.args.Test_Patches)
				else:
					print("a: train_val \n2: predict")
					x = input("Enter your choice :\n")

		elif ans == 4:
			sys.exit(0)
		else:
			print("1: Split.py \n2: Processing.py \n3: Model.py \n4: Quit \n")
			ans = input("Enter your choice :\n")

if __name__ == '__main__':

	Main()
	
