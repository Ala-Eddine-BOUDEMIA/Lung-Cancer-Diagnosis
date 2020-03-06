# -*- coding: UTF-8 -*-
#!/usr/bin/python

from PIL import Image

def repiece(input_path, i, W , H, incr_x, incr_y):
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

	#new_im.paste(image, box = (x, y)) #find something better

	img = Image.new('RGB',(int(w_tot/incr_x),int(h_tot/incr_y)))
	img.paste(new_im)
	#new_im = new_im.resize((int(w_tot/5),int(h_tot/5)), Image.ANTIALIAS)
	img.save("/home/ala/Desktop/PFE/Output_Images_jpeg/IMAGE.jpeg")

