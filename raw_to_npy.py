import spectral as sp
import numpy as np
import os

# train_header = sp.io.envi.read_envi_header()
image = sp.io.envi.open('./Input/Baltrum_Island/label_test.hdr', './Input/Baltrum_Island/label_test.raw').load()
np.save('./Input/Baltrum_Island/label_test.npy', image.squeeze())