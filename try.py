import rasterio
import numpy as np
import spectral as sp
import math
import matplotlib.pyplot as plt

# %% Global variables
USE_CHECKPOINT = False
SAVE_CHECKPOINT = True
PDDP_FLAG = False
USE_POWELL = True
USE_MRF = True
MRF_ITERATIONS = 3
OPTIMIZE_LAMBDA = False
MRF_SOLVER = 'GC'# 'GC', 'ICM'
KERNEL_TYPE = 'complex_sym_3' #'real', 'complex', 'complex_sym', 'complex_positive', 'complex_sym_2', 'complex_sym_3'
DATASET = 'San_Francisco'#'San Francisco', 'Flevoland'


# %%

def module_phase_to_complex(module, phase_difference, cross_phase = None):
    module_sqrt = np.sqrt(module)
    if phase_difference.max()>4:#if it is greater than pi (with some margin), then it is in degrees
        phase_difference_rads_1_2 = phase_difference.squeeze() * math.pi / 180 / 2
    else:
        phase_difference_rads_1_2 = phase_difference.squeeze() / 2
    cross_phase = cross_phase.squeeze()

    if cross_phase is None:
        real_image = np.zeros((module.shape[0], module.shape[1], 2))
        immaginary_image = np.zeros((module.shape[0], module.shape[1], 2))
    else:
        real_image = np.zeros((module.shape[0], module.shape[1], 3))
        immaginary_image = np.zeros((module.shape[0], module.shape[1], 3))

        real_image[:,:,2] = module_sqrt[:,:,2] * np.cos(cross_phase)
        immaginary_image[:,:,2] = module_sqrt[:,:,2] * np.sin(cross_phase)
            
    real_image[:,:,0] = module_sqrt[:,:,0] * np.cos(phase_difference_rads_1_2)
    immaginary_image[:,:,0] = module_sqrt[:,:,0] * np.sin(phase_difference_rads_1_2)

    real_image[:,:,1] = module_sqrt[:,:,1] * np.cos(phase_difference_rads_1_2)
    immaginary_image[:,:,1] = - module_sqrt[:,:,1] * np.sin(phase_difference_rads_1_2)

    return np.concatenate((np.expand_dims(real_image, axis=-1), np.expand_dims(immaginary_image, axis=-1)), axis=-1)

def Load_dataset(Dataset_name, use_checkpoint_flag=False, save_checkpoint_flag=False, PDDP_flag=True):
    if (Dataset_name == 'San_Francisco'):
        if (use_checkpoint_flag):
            train_map = np.load('Input/San_Francisco/train_map.npy')
            complex_image = np.load('Input/San_Francisco/complex_image.npy')
            n_classes = train_map.max()
        else:
            path_module = "./Input/San_Francisco/San_Francisco_c"
            module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

            path_phase = "./Input/San_Francisco/San_Francisco_phase_difference"
            phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
            
            path_cross_phase = "./Input/San_Francisco/San_Francisco_c_cross_phase"
            cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

            path_train_map = "./Input/San_Francisco/TrainMap_SF"
            train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())
            n_classes = train_map.max()

            complex_image = module_phase_to_complex(module_image, phase_diference, cross_phase)
            # complex_image = module_phase_to_complex(module_image, phase_diference)

            # PDDP
            if PDDP_flag:
                target = np.array((4000,4000,4000))
                train_map_PDDP_mask = np.zeros(shape=train_map.squeeze().shape, dtype=bool)
                for i in range(1,n_classes+1):
                    selected = cluster(complex_image, train_map, i, target[i-1])
                    print(f"label:{i}, samples:{selected.sum()}")
                    train_map_PDDP_mask = np.logical_or(train_map_PDDP_mask,selected)
                train_map = train_map * np.expand_dims(train_map_PDDP_mask, axis=-1)

        if save_checkpoint_flag:
            np.save('./Input/San_Francisco/train_map.npy',train_map)
            np.save('./Input/San_Francisco/complex_image.npy', complex_image)

    elif (Dataset_name=='Flevoland'):
        if (use_checkpoint_flag):
            train_map = np.load('Input/Flevoland/train_map.npy')
            complex_image = np.load('Input/Flevoland/complex_image.npy')
            n_classes = train_map.max()
        else:
            path_module = "./Input/Flevoland/Flevoland_l"
            # path_module = "./Input/Flevoland/Flevoland_l_db_pp"
            module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

            path_phase = "./Input/Flevoland/Flevoland_l_phase_difference"
            phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
            
            path_cross_phase = "./Input/Flevoland/Flevoland_l_cross_phase"
            cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

            # complex_image = module_phase_to_complex(module_image, phase_diference)
            complex_image = module_phase_to_complex(module_image, phase_diference, cross_phase)

            # path_module = "./Input/Flevoland/Flevoland_c"
            # # path_module = "./Input/Flevoland/Flevoland_c_db_pp"
            # module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

            # path_phase = "./Input/Flevoland/Flevoland_c_phase_difference"
            # phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
            
            # path_cross_phase = "./Input/Flevoland/Flevoland_c_cross_phase"
            # cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

            # # complex_image = module_phase_to_complex(module_image, phase_diference)
            # complex_image_c = module_phase_to_complex(module_image, phase_diference, cross_phase)

            # complex_image = np.concatenate([complex_image_l, complex_image_c], axis=2)

            path_train_map = "./Input/Flevoland/TrainMap_FL"
            train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())
            n_classes = train_map.max()

            # PDDP
            if PDDP_flag:
                target = np.ones((n_classes)) * 220
                train_map_PDDP_mask = np.zeros(shape=train_map.squeeze().shape, dtype=bool)
                for i in range(1,n_classes+1):
                    selected = cluster(complex_image, train_map, i, target[i-1])
                    print(f"label:{i}, samples:{selected.sum()}")
                    train_map_PDDP_mask = np.logical_or(train_map_PDDP_mask,selected)
                train_map = train_map * np.expand_dims(train_map_PDDP_mask, axis=-1)

        if save_checkpoint_flag:
            np.save('./Input/Flevoland/train_map.npy',train_map)
            np.save('./Input/Flevoland/complex_image.npy', complex_image)

    elif (Dataset_name == 'San_Francisco_3_4'):
        if (use_checkpoint_flag):
            train_map = np.load('Input/San_Francisco/train_map_3_4.npy')
            complex_image = np.load('Input/San_Francisco/complex_image_3_4.npy')
            n_classes = train_map.max()
        else:
            path_module = "./Input/San_Francisco/San_Francisco_c_3_4"
            module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

            path_phase = "./Input/San_Francisco/San_Francisco_phase_difference_3_4"
            phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
            
            path_cross_phase = "./Input/San_Francisco/San_Francisco_c_cross_phase_3_4"
            cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

            path_train_map = "./Input/San_Francisco/TrainingMap_SF_3_4"
            train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())
            n_classes = train_map.max()

            complex_image = module_phase_to_complex(module_image, phase_diference, cross_phase)
            # complex_image = module_phase_to_complex(module_image, phase_diference)

            # PDDP
            if PDDP_flag:
                target = np.array((2000,2000,2000))
                train_map_PDDP_mask = np.zeros(shape=train_map.squeeze().shape, dtype=bool)
                for i in range(1,n_classes+1):
                    selected = cluster(complex_image, train_map, i, target[i-1])
                    print(f"label:{i}, samples:{selected.sum()}")
                    train_map_PDDP_mask = np.logical_or(train_map_PDDP_mask,selected)
                train_map = train_map * np.expand_dims(train_map_PDDP_mask, axis=-1)


    return complex_image, train_map, n_classes

# Define the input .raw file path
raw_file = './Output/Flevoland/complex_MAP_complete.raw'

# Define the input .hdr file path
hdr_file = raw_file.replace('raw', 'hdr')

complex_image, train_map, n_classes = Load_dataset(Dataset_name='Flevoland', PDDP_flag=False)

module_image = np.sqrt(np.abs(complex_image[:,:,:,0] * complex_image[:,:,:,1]))
print(module_image.min())
print(module_image.max())

# for i in range(module_image.shape[2]):
# module_image = (module_image.max() - module_image) / (module_image.max() - module_image.min())

module_image_int = np.array(module_image * 255, dtype=np.uint8)

# Define the output .tif file path
# output_file = raw_file.replace('raw', 'tif')
output_file = './Output/Flevoland/composite.tif'

# Read the ENVI header file to extract necessary information
with open(hdr_file, 'r') as f:
    header_lines = f.readlines()

# Parse the header information
header_info = {}
for line in header_lines:
    if '=' not in line:
        continue
    key, value = line.strip().split('=')
    header_info[key.strip()] = value.strip()

# Extract the necessary information from the header
width = int(header_info['samples'])
height = int(header_info['lines'])
count = 3#int(header_info['bands'])
dtype = header_info['data type']

# Define the projection (assuming WGS84)
projection = {'init': 'epsg:4326'}  # Replace with the desired projection code

# Create a new raster dataset
with rasterio.open(
    output_file,
    'w',
    driver='GTiff',
    width=width,
    height=height,
    count=count,
    dtype='uint8',
    crs=projection,
) as dst:
    # Read the raw data and write it to the TIFF dataset
    # raw_data = np.asarray(sp.io.envi.open(hdr_file, raw_file).asarray())
    # print(raw_data.shape)
    # with open(raw_file, 'rb') as src_file:
    #     raw_data = src_file.read()
    #     print(raw_data)
    # dst.write(np.moveaxis(raw_data,-1,0))
    dst.write(np.moveaxis(module_image_int,-1,0))
