import spectral as sp
import numpy as np
import os

# train_header = sp.io.envi.read_envi_header()

filename = './Input/Baltrum_Island_L_S_reduced/BI_label_test_corrected'

image = sp.io.envi.open(filename + ".hdr", filename + ".raw").load()
np.save(filename + ".npy", image.squeeze())