import sys
import spectral as sp
import numpy as np
import maxflow
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
from HiPart.clustering import PDDP
from itertools import product, combinations

# %% Global variables
USE_CHECKPOINT = False
SAVE_CHECKPOINT = True
PDDP_FLAG = True
USE_POWELL = False
USE_MRF = True
MRF_ITERATIONS = 10
OPTIMIZE_LAMBDA = False
MRF_SOLVER = 'GC'# 'GC', 'ICM'
KERNEL_TYPE = 'complex_sym_3' #'real', 'complex', 'complex_sym', 'complex_positive', 'complex_sym_2', 'complex_sym_3'
DATASET = 'San_Francisco'#'San_Francisco', 'Flevoland'

# %% Various Functions

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

def module_kernel(gamma):
    def gaussian(X, Y):
        #not using modules but all values as different channels
        X_matrix = np.reshape(X,(X.shape[0],1,-1))
        Y_matrix = np.reshape(Y,(1,Y.shape[0],-1))
        D = np.sum((X_matrix - Y_matrix) ** 2, axis=-1)

        return np.exp(- D * gamma)
    return gaussian
    
def module_kernel_mrf(gamma):
    def gaussian(X, Y):
        #not using modules but all values as different channels
        X_matrix = np.reshape(X,(X.shape[0],1,-1))
        Y_matrix = np.reshape(Y,(1,Y.shape[0],-1))

        D = np.sum((X_matrix[:,:,:X_matrix.shape[2] - 2] - Y_matrix[:,:,:Y_matrix.shape[2] - 2]) ** 2, axis=-1)
        # return np.exp(- D * gamma) + lambda_ * X_matrix[:,:,X_matrix.shape[2] - 2] * Y_matrix[:,:,Y_matrix.shape[2] - 2]
        #instead of multiplying by lambda every time, I pre-multiply everything by lambda
        return np.exp(- D * gamma) + X_matrix[:,:,X_matrix.shape[2] - 2] * Y_matrix[:,:,Y_matrix.shape[2] - 2]
    return gaussian

def complex_kernel(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype='complex128')
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype='complex128')
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        D_c = np.sum((X_comp_matrix - Y_comp_matrix.conj()) ** 2, axis=-1)
        res = np.exp(- D_c * gamma)

        return res.real
    return gaussian

def complex_kernel_mrf(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype='complex128')
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype='complex128')
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        D_c = np.sum((X_comp_matrix[:,:,:X_comp_matrix.shape[2] - 1] - Y_comp_matrix[:,:,:Y_comp_matrix.shape[2] - 1].conj()) ** 2, axis=-1)
        res = np.exp(- D_c * gamma) + X_comp_matrix[:,:,X_comp_matrix.shape[2] - 1] * Y_comp_matrix[:,:,Y_comp_matrix.shape[2] - 1]
        #instead of multiplying by lambda every time, I pre-multiply everything by lambda

        return res.real
    return gaussian

def complex_symmetrical_kernel_mrf(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype='complex128')
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype='complex128')
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        D_c = np.sum(((X_comp_matrix[:,:,:X_comp_matrix.shape[2] - 1] - Y_comp_matrix[:,:,:Y_comp_matrix.shape[2] - 1].conj()) ** 2 +
                     (X_comp_matrix[:,:,:X_comp_matrix.shape[2] - 1] + Y_comp_matrix[:,:,:Y_comp_matrix.shape[2] - 1].conj()) ** 2 ) / 2, axis=-1)
        res = np.exp(- D_c * gamma) + X_comp_matrix[:,:,X_comp_matrix.shape[2] - 1] * Y_comp_matrix[:,:,Y_comp_matrix.shape[2] - 1]
        #instead of multiplying by lambda every time, I pre-multiply everything by lambda

        return res.real
    return gaussian

def complex_symmetrical_2_kernel_mrf(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype='complex128')
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype='complex128')
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        D_c = np.sum(((X_comp_matrix[:,:,:X_comp_matrix.shape[2] - 1] - Y_comp_matrix[:,:,:Y_comp_matrix.shape[2] - 1].conj()) ** 2 -
                     (X_comp_matrix[:,:,:X_comp_matrix.shape[2] - 1] + Y_comp_matrix[:,:,:Y_comp_matrix.shape[2] - 1].conj()) ** 2 ) / 2, axis=-1)
        res = np.exp(- D_c * gamma) + X_comp_matrix[:,:,X_comp_matrix.shape[2] - 1] * Y_comp_matrix[:,:,Y_comp_matrix.shape[2] - 1]
        #instead of multiplying by lambda every time, I pre-multiply everything by lambda

        return res.real
    return gaussian

def complex_symmetrical_3_kernel_mrf(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype='complex128')
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype='complex128')
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        X_comp_matrix_swaped = X_comp_matrix.copy()
        X_comp_matrix_swaped.real = X_comp_matrix.imag
        X_comp_matrix_swaped.imag = X_comp_matrix.real
        Y_comp_matrix_swaped = Y_comp_matrix.copy()
        Y_comp_matrix_swaped.real = Y_comp_matrix.imag
        Y_comp_matrix_swaped.imag = Y_comp_matrix.real
        D_c = np.sum(((X_comp_matrix[:,:,:X_comp_matrix.shape[2] - 1] - Y_comp_matrix[:,:,:Y_comp_matrix.shape[2] - 1].conj()) ** 2 +
                     (X_comp_matrix_swaped[:,:,:X_comp_matrix_swaped.shape[2] - 1] - Y_comp_matrix_swaped[:,:,:Y_comp_matrix_swaped.shape[2] - 1].conj()) ** 2 ) / 2,
                      axis=-1)
        res = np.exp(- D_c * gamma) + X_comp_matrix[:,:,X_comp_matrix.shape[2] - 1] * Y_comp_matrix[:,:,Y_comp_matrix.shape[2] - 1]
        #instead of multiplying by lambda every time, I pre-multiply everything by lambda

        return res.real
    return gaussian

def complex_positive_kernel_mrf(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype='complex128')
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype='complex128')
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        D_c = np.sum((X_comp_matrix[:,:,:X_comp_matrix.shape[2] - 1] + Y_comp_matrix[:,:,:Y_comp_matrix.shape[2] - 1].conj()) ** 2 , axis=-1)
        res = np.exp(- D_c * gamma) + X_comp_matrix[:,:,X_comp_matrix.shape[2] - 1] * Y_comp_matrix[:,:,Y_comp_matrix.shape[2] - 1]
        #instead of multiplying by lambda every time, I pre-multiply everything by lambda

        return res.real
    return gaussian

def svm_optimization_problem(x, *args): 
    # args[0]: complex_flatten_filtered
    # args[1]: train_map_flatten_filtered
    # C = math.exp(x[0])
    # gamma = 0.5 * math.exp(-x[1])
    C = x[0]
    gamma = x[1]
    clf = svm.SVC(kernel=kernel_function(gamma=gamma), C=C)
    clf.fit(X=args[0][0], y=args[0][1])
    # CHANGE score to nuSVM upperbound
    score = clf.score(X=args[0][0], y=args[0][1])
    return 1/score

def nusvm_optimization_problem(x, *args): 
    # args[0]: complex_flatten_filtered
    # args[1]: train_map_flatten_filtered
    # C = math.exp(x[0])
    # gamma = 0.5 * math.exp(-x[1])
    nu = x[0]
    gamma = x[1]
    clf = svm.NuSVC(kernel=kernel_function(gamma=gamma), nu=nu)
    clf.fit(X=args[0][0], y=args[0][1])
    # CHANGE score to nuSVM upperbound
    score = clf.score(X=args[0][0], y=args[0][1])
    return 1/score

def cluster(image, map, value, target):

    indeces = np.where(map.squeeze() == value)
    X = image[indeces]
    tot_samples = X.shape[0]

    if tot_samples == 0:
        return np.zeros(map.squeeze().shape, dtype=bool)

    X_reshaped = np.reshape(X,(tot_samples,-1))
    clustering = PDDP(min_sample_split=2, max_clusters_number=int(target)).fit_predict(X_reshaped)
    
    #In the case that less than "target" clusters were found, only unique centers are used (less than "target")
    clustering_unique_values = np.unique(clustering)
    center_sample = np.zeros((clustering_unique_values.size,X_reshaped.shape[1]))
    
    bool_list = np.zeros((tot_samples), dtype=bool)
    for i in range(center_sample.shape[0]):
        cluster_samples = X_reshaped[clustering == clustering_unique_values[i]]
        cluster_samples_indx = np.where(clustering == clustering_unique_values[i])[0]
        center = np.mean(cluster_samples, axis=0)
        distances_to_center = np.sum((center - cluster_samples)**2, axis=1)
        center_sample[i] = cluster_samples[np.argmin(distances_to_center),:]
        bool_list[cluster_samples_indx[np.argmin(distances_to_center)]] = True
    
    bool_map = np.zeros(map.squeeze().shape, dtype=bool)
    bool_map[indeces[0][bool_list],indeces[1][bool_list]] = True

    return bool_map

def EpsilonCompute(predicted_image, i, j):
    
    epsilon = 0
    count = 0
    center_class = predicted_image[i,j]
    for coordinate in product(list(range(-1,2)),repeat=2):
        new_i = i + coordinate[0]
        new_j = j + coordinate[1]
        if (new_i >= 0) and (new_i < predicted_image.shape[0]) and (new_j >= 0) and (new_j < predicted_image.shape[1]):
            count +=1
            if predicted_image[new_i,new_j] == center_class:
                epsilon +=1
            else:
                epsilon -=1

    # The center point should be disregarded
    epsilon -= 1
    count -=1
    return epsilon/count
    
def HoKashyap(matrix, max_iterations):
    rho = 1
    eps = 1e-4
    delta = eps * 2
    matrix_sharp = np.linalg.pinv(np.transpose(matrix) @ matrix) @ np.transpose(matrix)
    col = matrix.shape[1]
    Lambda = np.ones(col)

    b = matrix @ Lambda
    b[b<0] = 1

    deltas = np.zeros(max_iterations)
    for i in range(max_iterations):
        if delta < eps:
            break
        delta = 0
        
        Lambda_old = Lambda
        Lambda = matrix_sharp @ b
        err = matrix @ Lambda - b
        b[err>0] += rho * err[err>0]

        err_max = rho * err.max()
        if err_max > delta :
            delta = err_max

        dif_max = (np.abs(Lambda - Lambda_old)).max()
        if dif_max > delta:
            delta = dif_max
        deltas[i] = delta
    
    if Lambda.min() <= 0:
        Lambda = np.ones(col)
    
    Lambda = Lambda / Lambda.min()

    for idx, value in enumerate(Lambda):
        Lambda[idx] = min(value, 10)

    return Lambda

def SVM_MRF_HoKashyap(clf, spatial_image, train_map_pair_flatten, pair):

    spatial_image_flatten = np.reshape(spatial_image, (-1,spatial_image.shape[2],spatial_image.shape[3]))
    spatial_flatten_filtered = spatial_image_flatten[train_map_pair_flatten != 0]

    L = spatial_flatten_filtered.shape[0]
    E = np.zeros((L, 2))

    E[:,0] = clf.decision_function(spatial_flatten_filtered)

    sumepsilon = 0
    for j, index in enumerate(clf.support_):
        index_in_spatial_image_flatten = np.where(train_map_pair_flatten != 0)[0][index]
        row = index_in_spatial_image_flatten // spatial_image.shape[1]
        column = index_in_spatial_image_flatten % spatial_image.shape[1]
        sumepsilon += clf.dual_coef_[0,j] * spatial_image[row,column,-1,0]

    for i in range(L):
        index_in_spatial_image_flatten = np.where(train_map_pair_flatten != 0)[0][i]
        row = index_in_spatial_image_flatten // spatial_image.shape[1]
        column = index_in_spatial_image_flatten % spatial_image.shape[1]
        pred = clf.predict(np.reshape(spatial_image[row,column],(1,spatial_image.shape[2],spatial_image.shape[3])))
        if pred == pair[0]:
            E[i,1] = -1
        else:
            E[i,1] = 1
        E[i,1] *= sumepsilon * spatial_image[row,column,-1,0]
    
    Lambda_vector = HoKashyap(E, max_iterations = 1000000)
    
    return Lambda_vector

def SVM_MRF_HoKashyap_as_Gab(clf, spatial_image, train_map_pair_flatten, pair):

    spatial_image_flatten = np.reshape(spatial_image, (-1,spatial_image.shape[2],spatial_image.shape[3]))
    spatial_flatten_filtered = spatial_image_flatten[train_map_pair_flatten != 0]

    L = spatial_flatten_filtered.shape[0]

    E = np.zeros((L, 2))
    # sumepsilon = 0
    # for i, index in enumerate(clf.support_):
    #     index_in_spatial_image_flatten = np.where(train_map_pair_flatten != 0)[0][index]
    #     row = index_in_spatial_image_flatten // spatial_image.shape[1]
    #     column = index_in_spatial_image_flatten % spatial_image.shape[1]
    #     sumepsilon += clf.dual_coef_[0,i] * spatial_image[row,column,3,0]

    E[:,0] = clf.decision_function(spatial_flatten_filtered)

    for i in range(L):
        index_in_spatial_image_flatten = np.where(train_map_pair_flatten != 0)[0][i]
        row = index_in_spatial_image_flatten // spatial_image.shape[1]
        column = index_in_spatial_image_flatten % spatial_image.shape[1]
        pred = clf.predict(np.reshape(spatial_image[row,column],(1,spatial_image.shape[2],spatial_image.shape[3])))
        if pred == pair[0]:
            E[i,1] = -1
        else:
            E[i,1] = 1
        E[i,1] *= spatial_image[row,column,-1,0]
    
    Lambda_vector = HoKashyap(E, max_iterations = 1000000)
    
    return Lambda_vector

def SVM_MRF_GraphCuts(decision_image, lambda_):
    # Compute data cost
    data_cost = np.concatenate([(- decision_image)[:, :, np.newaxis], decision_image[:, :, np.newaxis]], axis=2).astype(np.float32)
    data_cost *= lambda_[0]

    # Compute smooth cost
    # smooth_cost = np.zeros_like(data_cost)
    # smooth_cost[:-1, :, 0] = np.expand_dims(np.sum(np.square(np.diff(decision_image, axis=0)), axis=-1), axis=-1)
    # smooth_cost[:, :-1, 1] = np.expand_dims(np.sum(np.square(np.diff(decision_image, axis=1)), axis=-1), axis=-1)
    # smooth_cost *= lambda_[1]

    # Create graph
    # g = maxflow.Graph[int]()
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(decision_image.shape[:2])

    # Add data cost
    # g.add_grid_edges(nodeids, weights=data_cost)
    g.add_grid_tedges(nodeids, data_cost[:,:,0,0], data_cost[:,:,1,0])

    # # Add smooth cost
    # g.add_grid_edges(nodeids, structure=np.array([[-1, 0], [0, -1]]), weights=smooth_cost[:, :-1].ravel(), symmetric=True)
    # g.add_grid_edges(nodeids, structure=np.array([[0, -1], [-1, 0]]), weights=smooth_cost[:-1, :].ravel(), symmetric=True)

    structure = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])
    
    g.add_grid_edges(nodeids, structure=structure, weights=lambda_[1], symmetric=True)

    # Perform graph cut
    energy = g.maxflow()

    # Get the labels
    labels = g.get_grid_segments(nodeids)

    # Compute energy
    # energy = g.get_grid_energy(nodeids, data_cost)

    # Return labeling and energy
    return energy, labels

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

            complex_image_c = module_phase_to_complex(module_image, phase_diference, cross_phase)
            # complex_image = module_phase_to_complex(module_image, phase_diference)

            path_module = "./Input/San_Francisco/San_Francisco_l"
            module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

            path_phase = "./Input/San_Francisco/San_Francisco_l_phase_difference"
            phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
            
            path_cross_phase = "./Input/San_Francisco/San_Francisco_l_cross_phase"
            cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

            complex_image_l = module_phase_to_complex(module_image, phase_diference, cross_phase)
            complex_image = np.concatenate([complex_image_l, complex_image_c], axis=2)

            path_train_map = "./Input/San_Francisco/TrainMap_SF"
            train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())
            n_classes = train_map.max()

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
            complex_image_l = module_phase_to_complex(module_image, phase_diference, cross_phase)

            path_module = "./Input/Flevoland/Flevoland_c"
            # path_module = "./Input/Flevoland/Flevoland_c_db_pp"
            module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

            path_phase = "./Input/Flevoland/Flevoland_c_phase_difference"
            phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
            
            path_cross_phase = "./Input/Flevoland/Flevoland_c_cross_phase"
            cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

            # complex_image = module_phase_to_complex(module_image, phase_diference)
            complex_image_c = module_phase_to_complex(module_image, phase_diference, cross_phase)

            complex_image = np.concatenate([complex_image_l, complex_image_c], axis=2)

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

#%% Main
if __name__=="__main__":
    
    complex_image, train_map, n_classes = Load_dataset(DATASET, USE_CHECKPOINT, SAVE_CHECKPOINT, PDDP_FLAG)   

    # Compute flatten imgs
    complex_image_flatten = np.reshape(complex_image, (-1,complex_image.shape[2],complex_image.shape[3]))
    train_map_flatten = np.reshape(train_map,(-1))
    complex_flatten_filtered = complex_image_flatten[train_map_flatten != 0]
    train_map_flatten_filtered = train_map_flatten[train_map_flatten != 0]
        
    print(f"Dataset: {DATASET} loaded")

    # Set kernel function
    if (USE_MRF):
        match KERNEL_TYPE:
            case 'real':
                kernel_function = module_kernel_mrf
            case 'complex':
                kernel_function = complex_kernel_mrf
            case 'complex_sym':
                kernel_function = complex_symmetrical_kernel_mrf
            case 'complex_positive':
                kernel_function = complex_positive_kernel_mrf
            case 'complex_sym_2':
                kernel_function = complex_symmetrical_2_kernel_mrf
            case 'complex_sym_3':
                kernel_function = complex_symmetrical_3_kernel_mrf
    else:
        match KERNEL_TYPE:
            case 'real':
                kernel_function = module_kernel
            case 'complex':
                kernel_function = complex_kernel
            case 'complex_sym':
                pass

    #%% Parameter tunning

    print("Parameter tunning...")

    if USE_POWELL:
        x0 = np.array((1,1))
        sol = minimize(
                svm_optimization_problem,
                x0=x0, 
                args=[complex_flatten_filtered, train_map_flatten_filtered],
                method="Powell",
                # bounds=((-4,3),(-1.4,1))
                bounds=((0.02,20),(0.05,4))
        )

        x = sol.x
        # C = math.exp(x[0])
        # gamma = 0.5 * math.exp(-x[1])
        C = x[0]
        gamma = x[1]
        print(f"C:{C}    gamma:{gamma}")
    else:
        # C = 2.718
        # gamma = 0.35
        if DATASET=='Flevoland':
            C = 15.892858723624235
            gamma = 3.9358827531948766
        else:
            C = 15.892858723624235
            gamma = 3.9358827531948766

    #%% Predict
    Predict_batches = 512
    # only because i use the real
    # gamma=0.35
    # C=2.718
    # clf = svm.SVC(kernel=module_kernel_mrf(gamma=gamma, lambda_=lambda_), C=C)
    clf = svm.SVC(kernel=kernel_function(gamma=gamma), C=C)
    
    if not USE_MRF:
        MRF_ITERATIONS = 1

    lambda_ = [1,1] #handpicked

    spatial_image = np.zeros((complex_image.shape[0], complex_image.shape[1], complex_image.shape[2] + 1, complex_image.shape[3]))
    spatial_image[:,:,:complex_image.shape[2],:] = complex_image

    vote_map = np.zeros((train_map.shape[0],train_map.shape[1], n_classes), dtype=np.uint8)

    classes_labels = [x+1 for x in range(n_classes)]
    for pair in combinations(classes_labels,2):
        print(f'pair:{pair}')

        if (train_map!=pair[0]).all():
            vote_map[:, pair[1] - 1] +=1
            continue
        elif (train_map!=pair[1]).all():
            vote_map[:, pair[0] - 1] +=1
            continue

        energy_min = sys.float_info.max
        train_map_pair = np.zeros(train_map.shape, dtype=np.uint8)
        train_map_pair[train_map==pair[0]] = pair[0]
        train_map_pair[train_map==pair[1]] = pair[1]
        train_map_pair_flatten = np.reshape(train_map_pair,(-1))
        train_map_pair_flatten_filtered = train_map_pair_flatten[train_map_pair_flatten != 0]
        classification_optimal = np.zeros((train_map.shape[0],train_map.shape[1]), dtype=np.uint8)

        for iter in range(MRF_ITERATIONS):
            print(f'Iteration:{iter}')
            spatial_image_flatten = np.reshape(spatial_image, (-1, spatial_image.shape[2], spatial_image.shape[3]))
            spatial_flatten_filtered = spatial_image_flatten[train_map_pair_flatten != 0]

            clf.fit(X=spatial_flatten_filtered, y=train_map_pair_flatten_filtered)
            #ONLY for try
            # lambda_ = SVM_MRF_HoKashyap(clf, spatial_image, train_map_pair_flatten)

            y_predicted_flatten = np.zeros((train_map_pair_flatten.shape), dtype=np.uint8) 
            decision_flatten = np.zeros((train_map_pair_flatten.shape)) 
            for division in range(Predict_batches):
                decision_flatten[
                    decision_flatten.size // Predict_batches * division : decision_flatten.size // Predict_batches * (division + 1)
                    ] = clf.decision_function(spatial_image_flatten[
                    decision_flatten.size // Predict_batches * division : decision_flatten.size // Predict_batches * (division + 1)
                    ])                 
            if (decision_flatten.size % Predict_batches != 0):
                decision_flatten[decision_flatten.size // Predict_batches * Predict_batches:] = clf.decision_function(spatial_image_flatten[decision_flatten.size // Predict_batches * Predict_batches:])
            sign_decision = np.sign(decision_flatten)
            y_predicted_flatten[sign_decision == -1] = pair[0]
            y_predicted_flatten[sign_decision == 1] = pair[1]
            decision_image = np.reshape(decision_flatten,(train_map.shape))
            y_predicted_image = np.reshape(y_predicted_flatten,(train_map.shape))

            if not USE_MRF:
                continue

            if (iter<MRF_ITERATIONS - 1): #in the last run it is not necessary
                for i in range(spatial_image.shape[0]):
                    for j in range(spatial_image.shape[1]):                    
                        spatial_image[i,j,-1,0] = EpsilonCompute(y_predicted_image,i,j)

            if OPTIMIZE_LAMBDA:
                if (iter == 0):
                    #calculate lambda_
                    lambda_ = SVM_MRF_HoKashyap_as_Gab(clf, spatial_image, train_map_pair_flatten, pair)
                    print(f'Lambda:{lambda_}')
                    #remove
                    # lambda_1 = SVM_MRF_HoKashyap(clf, spatial_image, train_map_pair_flatten, pair)
                    # print(f'Lambda:{lambda_1}')
                    # spatial_image[:,:,-1,0] *= lambda_[1]#? aca si creo que es??
                    continue
            
            # spatial_image[:,:,-1,0] *= lambda_[1]#? aca si creo que es??
            if MRF_SOLVER == 'GC':
                energy, classification = SVM_MRF_GraphCuts(decision_image, lambda_)
                if energy < energy_min:
                    energy_min = energy
                    classification_optimal = classification

                        
        if MRF_SOLVER == 'ICM' or not USE_MRF:
            vote_map[y_predicted_image.squeeze() == pair[0],pair[0] - 1] += 1
            vote_map[y_predicted_image.squeeze() == pair[1],pair[1] - 1] += 1
        elif MRF_SOLVER == 'GC':
            vote_map[classification_optimal == 0,pair[0] - 1] += 1
            vote_map[classification_optimal == 1,pair[1] - 1] += 1

    max_vote = np.argmax(vote_map, axis=2) + 1
    np.save(f"./Output/{DATASET}/complex_sym_3_complete_powell_pddp2000_mrf10_l_c.npy", max_vote)
    plt.imshow(max_vote * 255. / max_vote.max())
    plt.show()

    a = 0