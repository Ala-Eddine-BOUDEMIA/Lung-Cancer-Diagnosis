# -*- coding: UTF-8 -*-
#!/usr/bin/python

import Read_patch as rp
import Repiece

input_path_rp = '/home/ala/Desktop/PFE/Input_Images_WSI/image.svs'	
(i, w, h, incr_x, incr_y) = rp.read_patch(input_path_rp)

input_path_repiece = '/home/ala/Desktop/PFE/Output_Patches/image'
Repiece.repiece(input_path_repiece, i, w, h, incr_x, incr_y)