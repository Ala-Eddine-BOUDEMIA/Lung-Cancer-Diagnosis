import Utils                
import Config
#############
from PIL import Image
#####################
import sys

def repiece(Predictions = Config.args.Predictions):

	path_list, csv_list = Utils.parse_dir(Predictions, "csv")
	print(csv_list)
	sys.exit(0)

	print("Repiece patches together")
	compression_factor = 3
	new_im = Image.new('RGB',(int(W/incr_x),int(H/incr_y)))

	a, b, x, y, w_tot, h_tot = 0, 0, 0, 0, 0, 0
	j = 1
	while j < i+1 :
	#for i in range(1,i):
		image = Image.open(input_path+str(j)+'.jpeg')
		(w, h) = image.size
		w_size = int(w/compression_factor)
		h_size = int(h/compression_factor)
		w_tot += w_size
		h_tot += h_size
		image = image.resize((w_size,h_size), Image.ANTIALIAS)
		new_im.paste(image, box = (x, y))
		
		if b < (incr_y-1):
			b += 1
			y += h_size
		else :
			if a < (incr_x-1):
				a += 1
				b = 0
				x += w_size
				y = 0
		j += 1

if __name__ == '__main__':
	repiece()