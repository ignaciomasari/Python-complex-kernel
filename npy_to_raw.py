import spectral as sp
import numpy as np
import os

train_header = sp.io.envi.read_envi_header("./Input/San_Francisco/TrainMap_SF.hdr")

file_path = "./Output/San_Francisco/complex_sym_3_complete_powell_pddp2000_mrf3_l_c.npy"
im_array = np.load(file_path)
sp.envi.save_image(file_path.replace('npy','hdr'), im_array, dtype=np.uint8, ext='.raw',force=True)
os.remove(file_path)