import Imports
##############

#########################################################################################################
# This implementation is based on the discussion from: 													#
#	https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9			#
#########################################################################################################
def compute_mean_std(folderpath):

	class MyDataset(Dataset):

		def __init__(self, folder):

			self.data = []
			path_list, jpeg_paths = Utils.parse_dir(folder, "jpeg")
			for path in jpeg_paths:
				self.data.append(path)

		def __getitem__(self, index):

			return ToTensor()(Image.open(self.data[index]).convert("RGB"))

		def __len__(self):

			return len(self.data)

	def online_mean_and_sd(loader):

		cnt = 0
		fst_moment = torch.empty(3)
		snd_moment = torch.empty(3)

		for data in loader:
			b, c, h, w = data.shape
			nb_pixels = b * h * w

			sum_ = torch.sum(data, dim=[0, 2, 3])
			sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
			fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
			snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

			cnt += nb_pixels

		return fst_moment.tolist(), torch.sqrt(snd_moment - fst_moment**2).tolist()

	return online_mean_and_sd(loader = torch.utils.data.DataLoader(dataset = MyDataset(folder = folderpath), 
    	batch_size = 1, num_workers = 1, shuffle = False))