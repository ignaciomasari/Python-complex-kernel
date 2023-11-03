import sys
import os
import math
import warnings
import time
# import concurrent.futures
from itertools import product, combinations

import spectral as sp
import numpy as np
import maxflow
import matplotlib.pyplot as plt
import pdfo
from sklearn import svm
from scipy.optimize import minimize
from pathos.multiprocessing import ProcessingPool as PPool
from HiPart.clustering import PDDP

# If using intel herdware, use optimization
try:
    from sklearnex import patch_sklearn    
except ImportError:
    print("No Intel extension")
else:
    patch_sklearn()

# %% Global variables
USE_CHECKPOINT = False
SAVE_CHECKPOINT = False
PDDP_FLAG = True
PDDP_TARGET = 1000
USE_POWELL = False
USE_MRF = True
MRF_ITERATIONS = 1
OPTIMIZE_LAMBDA = False
MRF_SOLVER = 'GC'# 'GC', 'ICM'
KERNEL_TYPE = 'complex_sym' #'real', 'complex', 'complex_sym'
DATASET = 'Baltrum_Island'#'San_Francisco', 'Flevoland', 'Baltrum_Island'

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

    return np.concatenate((np.expand_dims(real_image, axis=-1), np.expand_dims(immaginary_image, axis=-1)), axis=-1, dtype=np.float32)

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
        X_comp = np.array(X[:,:,0], dtype=np.complex64)
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype=np.complex64)
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        D_c = np.sum((X_comp_matrix - Y_comp_matrix.conj()) ** 2, axis=-1)
        res = np.exp(- D_c * gamma)

        return res.real
    return gaussian

def complex_kernel_mrf(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype=np.complex64)
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype=np.complex64)
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        D_c = np.sum((X_comp_matrix[:,:,:X_comp_matrix.shape[2] - 1] - Y_comp_matrix[:,:,:Y_comp_matrix.shape[2] - 1].conj()) ** 2, axis=-1)
        res = np.exp(- D_c * gamma) + X_comp_matrix[:,:,X_comp_matrix.shape[2] - 1] * Y_comp_matrix[:,:,Y_comp_matrix.shape[2] - 1]
        #instead of multiplying by lambda every time, I pre-multiply everything by lambda

        return res.real
    return gaussian

def complex_symmetrical_kernel(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype=np.complex64)
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype=np.complex64)
        Y_comp.imag = Y[:,:,1]
        X_comp_matrix = np.reshape(X_comp,(X_comp.shape[0],1,-1))
        Y_comp_matrix = np.reshape(Y_comp,(1,Y_comp.shape[0],-1))
        X_comp_matrix_swaped = X_comp_matrix.copy()
        X_comp_matrix_swaped.real = X_comp_matrix.imag
        X_comp_matrix_swaped.imag = X_comp_matrix.real
        Y_comp_matrix_swaped = Y_comp_matrix.copy()
        Y_comp_matrix_swaped.real = Y_comp_matrix.imag
        Y_comp_matrix_swaped.imag = Y_comp_matrix.real
        D_c = np.sum(((X_comp_matrix - Y_comp_matrix.conj()) ** 2 +
                     (X_comp_matrix_swaped - Y_comp_matrix_swaped.conj()) ** 2 ) / 2,
                      axis=-1)
        res = np.exp(- D_c * gamma)        

        return res.real
    return gaussian

def complex_symmetrical_kernel_mrf(gamma):
    def gaussian(X, Y):
        X_comp = np.array(X[:,:,0], dtype=np.complex64)
        X_comp.imag = X[:,:,1]
        Y_comp = np.array(Y[:,:,0], dtype=np.complex64)
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

def svm_optimization_problem(x, *args): 
    # args[0]: complex_flatten_filtered
    # args[1]: train_map_flatten_filtered
    # C = math.exp(x[0])
    # gamma = 0.5 * math.exp(-x[1])
    # print(args)
    C = x[0]
    gamma = x[1]
    kernel_function = args[2]
    clf = svm.SVC(kernel=kernel_function(gamma=gamma), C=C)
    clf.fit(X=args[0], y=args[1])
    score = clf.score(X=args[0], y=args[1])
    return 1/score

def svm_optimization_problem_linearized(x, *args): 
    # args[0]: complex_flatten_filtered
    # args[1]: train_map_flatten_filtered
    C = math.exp(x[0])
    gamma = 0.5 * math.exp(-x[1])
    # C = x[0]
    # gamma = x[1]
    clf = svm.SVC(kernel=kernel_function(gamma=gamma), C=C)
    clf.fit(X=args[0][0], y=args[0][1])
    # CHANGE score to nuSVM upperbound
    score = clf.score(X=args[0][0], y=args[0][1])
    return 1/score

def svm_optimization_problem_SB_linearized(x, *args): 
    # args[0]: complex_flatten_filtered
    # args[1]: train_map_flatten_filtered
    # args[2]: kernel_function
    # bounds=((-4,3),(-9,5)),
    if x[0]<-4 or x[0]>4 or x[1]<-4 or x[1]>4:
        return len(args[1])

    C = math.exp(x[0])
    gamma = 0.5 * math.exp(-x[1])
    # C = x[0]
    # gamma = x[1]
    kernel_function = args[2](gamma=gamma)
    clf = svm.SVC(kernel=kernel_function, C=C, decision_function_shape='ovo')
    SB = multiclass_span_estimate(clf, kernel_function, args[0], args[1])
    return SB

def svm_optimization_problem_SB(x, *args): 
    # args[0]: complex_flatten_filtered
    # args[1]: train_map_flatten_filtered
    # args[2]: kernel_function
    C = x[0]
    gamma = x[1]
    kernel_function = args[2](gamma=gamma)
    clf = svm.SVC(kernel=kernel_function, C=C, decision_function_shape='ovo')
    SB = multiclass_span_estimate(clf, kernel_function, args[0], args[1])
    return SB

def svm_optimization_problem_SB_vectors(x, *args): 
    # args[0]: complex_flatten_filtered
    # args[1]: train_map_flatten_filtered
    # args[2]: kernel_function
    C = x[0]
    gamma = x[1]
    kernel_function = args[2]
    SB = multiclass_span_estimate_vectors(kernel_function, C, gamma, args[0], args[1])
    return SB

def fast_span_estimate(svm, kernel_function, X_tr, Y_tr):
    """Fast estimate of the span of the support vectors.
    
    Parameters
    ----------
    K : array-like, shape (n_samples, n_samples)
        Kernel matrix.
    Y : array-like, shape (n_samples,)
        Labels.
    alpha : array-like, shape (n_samples,)
        Lagrange multipliers.
    b : float
        Lagrange threshold.
    C : float
        soft margin parameter, aka upper bound on alpha.
        
    Returns
    -------
    span : float
        Estimate of the fraction of leave-one-out errors.
    """
        
    alpha = svm.dual_coef_.squeeze()
    K = kernel_function(X_tr, X_tr)
    C = svm.C

    # Compute the outputs on the training points
    output = Y_tr * svm.decision_function(X_tr)

    # Find the indices of the support vectors of first and second category
    # eps = 1e-6
    # sv1 = (np.logical_and(alpha > alpha.max()*eps, alpha < C*(1-eps))).squeeze()
    # sv2 = (alpha > C*(1-eps)).squeeze()
    sv1 = (alpha < C).squeeze()
    sv2 = (alpha == C).squeeze()

    # Degenerate case: if sv1 is empty, then we assume nothing changes
    # (loo = training error)
    l = sv1.sum()
    if l==0:
        SB = np.mean(output < 0)
        return SB

    # Compute the invert of KSV
    KSV = np.ones((l+1,l+1))
    # print("Chequear K")

    for ix, x in enumerate(svm.support_[sv1]):
        for iy, y in enumerate(svm.support_[sv1]):
            KSV[ix,iy] = K[x,y]

    KSV[-1,-1] = 0
    KSV += np.eye(l+1) * 1e-8  # a small ridge is added to be sure that the matrix is invertible
    invKSV = np.linalg.inv(KSV)

    # Compute the span for all support vectors
    n_sv = len(alpha.squeeze()) # Number of support vectors
    span = np.zeros(n_sv) # Initialize the vector
    tmp = np.diag(invKSV)
    span[:l] = np.reciprocal(tmp[:-1]) # Span of sv of first category

    # If there exists sv of second category, compute their span
    if sv2.sum() != 0:
        # print("Chequear K")

        V= np.ones((l+1,n_sv-l))
        for ix, x in enumerate(svm.support_[sv1]):
            for iy, y in enumerate(svm.support_[sv2]):
                V[ix,iy] = K[x,y]        

        KSV2 = np.zeros((n_sv-l,n_sv-l))
        for ix, x in enumerate(svm.support_[sv2]):
            for iy, y in enumerate(svm.support_[sv2]):
                KSV2[ix,iy] = K[x,y]

        span[l:] = np.diag(KSV2) - np.diag(V.T @ invKSV @ V)

    n_samples = len(Y_tr)
    alpha_x_span = np.zeros((n_samples), dtype=np.float64)
    alpha_x_span[svm.support_[sv1]] = alpha[sv1] * span[:l]
    alpha_x_span[svm.support_[sv2]] = alpha[sv2] * span[l:]

    # Estimate the fraction of loo error
    SB = np.mean((output - alpha_x_span) < 0)

    return SB

def multiclass_span_estimate(svm, kernel_function, X_multiclass, Y_multiclass):
    
    SB=0
    classes_labels = np.unique(Y_multiclass)
    
    for pair in combinations(classes_labels,2):
        Y_ovo_large = np.zeros((Y_multiclass.shape), dtype=np.int8)
        Y_ovo_large[Y_multiclass==pair[0]] = 1
        Y_ovo_large[Y_multiclass==pair[1]] = -1
        Y_ovo = Y_ovo_large[Y_ovo_large!=0]
        X_ovo = X_multiclass[Y_ovo_large!=0]
        svm.fit(X=X_ovo, y=Y_ovo)

        SB += fast_span_estimate(svm, kernel_function, X_ovo, Y_ovo)

    return SB

def multiclass_span_estimate_vectors(kernel_function, vector_C, vector_gamma, X_multiclass, Y_multiclass):
    
    SB=0
    classes_labels = np.unique(Y_multiclass)
    
    for index, pair in enumerate(combinations(classes_labels,2)):
        svc = svm.SVC(kernel=kernel_function(gamma=vector_gamma[index]), C=vector_C[index], decision_function_shape='ovo')
        Y_ovo_large = np.zeros((Y_multiclass.shape), dtype=np.int8)
        Y_ovo_large[Y_multiclass==pair[0]] = 1
        Y_ovo_large[Y_multiclass==pair[1]] = -1
        Y_ovo = Y_ovo_large[Y_ovo_large!=0]
        X_ovo = X_multiclass[Y_ovo_large!=0]
        svc.fit(X=X_ovo, y=Y_ovo)

        SB += fast_span_estimate(svc, kernel_function, X_ovo, Y_ovo)

    return SB

def cluster(image, map, value, target):

    indeces = np.where(map.squeeze() == value)
    X = image[indeces]
    tot_samples = X.shape[0]

    if tot_samples == 0:
        return np.zeros(map.squeeze().shape, dtype=np.bool_)

    X_reshaped = np.reshape(X,(tot_samples,-1))
    clustering = PDDP(min_sample_split=2, max_clusters_number=int(target)).fit_predict(X_reshaped)
    
    #In the case that less than "target" clusters were found, only unique centers are used (less than "target")
    clustering_unique_values = np.unique(clustering)
    center_sample = np.zeros((clustering_unique_values.size,X_reshaped.shape[1]))
    
    bool_list = np.zeros((tot_samples), dtype=np.bool_)
    for i in range(center_sample.shape[0]):
        cluster_samples = X_reshaped[clustering == clustering_unique_values[i]]
        cluster_samples_indx = np.where(clustering == clustering_unique_values[i])[0]
        center = np.mean(cluster_samples, axis=0)
        distances_to_center = np.sum((center - cluster_samples)**2, axis=1)
        center_sample[i] = cluster_samples[np.argmin(distances_to_center),:]
        bool_list[cluster_samples_indx[np.argmin(distances_to_center)]] = True
    
    bool_map = np.zeros(map.squeeze().shape, dtype=np.bool_)
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

def Load_dataset(Dataset_name, use_checkpoint_flag=False, save_checkpoint_flag=False, PDDP_flag=True, PDDP_target=1000, plot=False):
    if use_checkpoint_flag:
        try:
            train_map = np.load(f'Input/{Dataset_name}/train_map.npy')
            print("Total number of samples: ", (train_map!=0).sum())
            complex_image = np.load(f'Input/{Dataset_name}/complex_image_checkpoint.npy')
            n_classes = train_map.max()
    
        except FileNotFoundError:
            print("Checkpoint not found")
            use_checkpoint_flag = False

    if not use_checkpoint_flag:
        print("Loading dataset...")
        
        path_module = f"./Input/{Dataset_name}/{Dataset_name}_c"
        module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

        path_phase = f"./Input/{Dataset_name}/{Dataset_name}_phase_difference"
        phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
        
        path_cross_phase = f"./Input/{Dataset_name}/{Dataset_name}_c_cross_phase"
        cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

        complex_image_c = module_phase_to_complex(module_image, phase_diference, cross_phase)
        # complex_image = module_phase_to_complex(module_image, phase_diference)

        path_module = f"./Input/{Dataset_name}/{Dataset_name}_l"
        module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

        path_phase = f"./Input/{Dataset_name}/{Dataset_name}_l_phase_difference"
        phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
        
        path_cross_phase = f"./Input/{Dataset_name}/{Dataset_name}_l_cross_phase"
        cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

        complex_image_l = module_phase_to_complex(module_image, phase_diference, cross_phase)
        # print("SOLO POR AHORA SOLO IMAGEN L")
        # complex_image = complex_image_l
        complex_image = np.concatenate([complex_image_l, complex_image_c], axis=2)

        path_train_map = f"./Input/{Dataset_name}/TrainMap"
        train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())
        n_classes = train_map.max()

        # PDDP
        if PDDP_flag:
            target = np.ones((n_classes), dtype=np.int32) * int(PDDP_target)
            train_map_PDDP_mask = np.zeros(shape=train_map.squeeze().shape, dtype=np.bool_)
            for i in range(1,n_classes+1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    selected = cluster(complex_image, train_map, i, target[i-1])
                print(f"label:{i}, samples:{selected.sum()}")
                train_map_PDDP_mask = np.logical_or(train_map_PDDP_mask,selected)
            train_map = train_map * np.expand_dims(train_map_PDDP_mask, axis=-1)

        if save_checkpoint_flag:
            np.save(f'./Input/{Dataset_name}/train_map.npy',train_map)
            np.save(f'./Input/{Dataset_name}/complex_image_checkpoint.npy', complex_image)
                
    if plot:
        fig= plt.figure(figsize=(3,2))
        fig.add_subplot(2,3,1)
        # plt.imshow(complex_image[:,:,0,0], cmap='gray')
        plt.hist(complex_image[:,:,0,0].flatten(), bins=1000)
        fig.add_subplot(2,3,2)
        # plt.imshow(complex_image[:,:,1,0], cmap='gray')
        plt.hist(complex_image[:,:,1,0].flatten(), bins=1000)
        fig.add_subplot(2,3,3)
        # plt.imshow(complex_image[:,:,2,0], cmap='gray')
        plt.hist(complex_image[:,:,2,0].flatten(), bins=1000)
        fig.add_subplot(2,3,4)
        # plt.imshow(complex_image[:,:,0,1], cmap='gray')
        plt.hist(complex_image[:,:,0,1].flatten(), bins=1000)
        fig.add_subplot(2,3,5)
        # plt.imshow(complex_image[:,:,1,1], cmap='gray')
        plt.hist(complex_image[:,:,1,1].flatten(), bins=1000)
        fig.add_subplot(2,3,6)
        # plt.imshow(complex_image[:,:,2,1], cmap='gray')
        plt.hist(complex_image[:,:,2,1].flatten(), bins=1000)
        plt.show()

    return complex_image, train_map, n_classes

def Load_dataset_T3(Dataset_name, use_checkpoint_flag=False, save_checkpoint_flag=False, PDDP_flag=True):
    if (Dataset_name == 'San_Francisco'):
        if (use_checkpoint_flag):
            train_map = np.load('Input/San_Francisco/train_map.npy')
            complex_image = np.load('Input/San_Francisco/complex_T3_log_checkpoint.npy')
            n_classes = train_map.max()
        else:
            print("Loading LOG dataset...")
            path_module = "./Input/San_Francisco/T3_modules_log.npy"
            # print("Loading dataset...")
            # path_module = "./Input/San_Francisco/T3_modules.npy"
            module_image = np.load(path_module)

            path_phase = "./Input/San_Francisco/T3_phases.npy"
            phase_image = np.load(path_phase)

            complex_image = np.zeros((module_image.shape[0], module_image.shape[1], module_image.shape[2], 2), dtype='float32')
            
            for band in range(module_image.shape[2]):
                complex_image[:,:,band,0] = module_image[:,:,band] * np.cos(phase_image[:,:,band])
                complex_image[:,:,band,1] = module_image[:,:,band] * np.sin(phase_image[:,:,band])

            path_train_map = "./Input/San_Francisco/TrainMap_SF"
            train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())
            n_classes = train_map.max()

            # PDDP
            if PDDP_flag:
                target = np.array((1000,1000,1000))
                train_map_PDDP_mask = np.zeros(shape=train_map.squeeze().shape, dtype=bool)
                for i in range(1,n_classes+1):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        selected = cluster(complex_image, train_map, i, target[i-1])
                    print(f"label:{i}, samples:{selected.sum()}")
                    train_map_PDDP_mask = np.logical_or(train_map_PDDP_mask,selected)
                train_map = train_map * np.expand_dims(train_map_PDDP_mask, axis=-1)

        if save_checkpoint_flag:
            np.save('./Input/San_Francisco/train_map.npy',train_map)
            np.save('./Input/San_Francisco/complex_T3_log_checkpoint.npy', complex_image)

    elif (Dataset_name=='Flevoland'):
        pass

    elif (Dataset_name == 'San_Francisco_3_4'):
        pass

    return complex_image, train_map, n_classes

def Load_dataset_image(Dataset_name, use_checkpoint_flag=False, save_checkpoint_flag=False, PDDP_flag=True, PDDP_target=1000, plot=False):
    
    if use_checkpoint_flag:
        try:
            train_map = np.load(f'Input/{Dataset_name}/train_map.npy')
            print("Total number of samples: ", (train_map!=0).sum())
            complex_image = np.load(f'Input/{Dataset_name}/complex_image_log_checkpoint.npy')
            n_classes = train_map.max()
    
        except FileNotFoundError:
            print("Checkpoint not found")
            use_checkpoint_flag = False

    if not use_checkpoint_flag:
        if (Dataset_name == 'San_Francisco' or Dataset_name == 'Flevoland'):
            print("Loading c_ENVI image dataset...")
            path_module = f"./Input/{Dataset_name}/image_c_ENVI.npy"
            image_c = np.load(path_module)
            print("Loading l_ENVI image dataset...")
            path_module = f"./Input/{Dataset_name}/image_l_ENVI.npy"
            image_l = np.load(path_module)
            image = np.concatenate([image_c, image_l], axis=2)

        else:
            print("Loading LOG image dataset...")
            path_module = f"./Input/{Dataset_name}/image_log.npy"
            # print("Loading no_LOG image dataset...")
            # path_module = f"./Input/{Dataset_name}/image_no_log.npy"
            image = np.load(path_module)

        complex_image = np.zeros((image.shape[0], image.shape[1], int(image.shape[2] / 2), 2), dtype=np.float32)
        
        for band in range(complex_image.shape[2]):
            complex_image[:,:,band,0] = image[:,:,2 * band]
            complex_image[:,:,band,1] = image[:,:,2 * band + 1]

        path_train_map = f"./Input/{Dataset_name}/TrainMap"
        train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())
        n_classes = train_map.max()

        # PDDP
        if PDDP_flag:
            target = np.ones((n_classes), dtype=np.int32) * int(PDDP_target)
            train_map_PDDP_mask = np.zeros(shape=train_map.squeeze().shape, dtype=np.bool_)
            for i in range(1,n_classes+1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    selected = cluster(complex_image, train_map, i, target[i-1])
                print(f"label:{i}, samples:{selected.sum()}")
                train_map_PDDP_mask = np.logical_or(train_map_PDDP_mask,selected)
            train_map = train_map * np.expand_dims(train_map_PDDP_mask, axis=-1)

        if save_checkpoint_flag:
            np.save(f'./Input/{Dataset_name}/train_map.npy',train_map)
            np.save(f'./Input/{Dataset_name}/complex_image_log_checkpoint.npy', complex_image)
                
    if plot:
        fig= plt.figure(figsize=(3,2))
        fig.add_subplot(2,3,1)
        plt.imshow(complex_image[:,:,0,0], cmap='gray')
        # plt.hist(complex_image[:,:,0,0].flatten(), bins=1000)
        fig.add_subplot(2,3,2)
        plt.imshow(complex_image[:,:,1,0], cmap='gray')
        # plt.hist(complex_image[:,:,1,0].flatten(), bins=1000)
        fig.add_subplot(2,3,3)
        plt.imshow(complex_image[:,:,2,0], cmap='gray')
        # plt.hist(complex_image[:,:,2,0].flatten(), bins=1000)
        fig.add_subplot(2,3,4)
        plt.imshow(complex_image[:,:,0,1], cmap='gray')
        # plt.hist(complex_image[:,:,0,1].flatten(), bins=1000)
        fig.add_subplot(2,3,5)
        plt.imshow(complex_image[:,:,1,1], cmap='gray')
        # plt.hist(complex_image[:,:,1,1].flatten(), bins=1000)
        fig.add_subplot(2,3,6)
        plt.imshow(complex_image[:,:,2,1], cmap='gray')
        # plt.hist(complex_image[:,:,2,1].flatten(), bins=1000)
        plt.show()


    return complex_image, train_map, n_classes

#%% Main
if __name__=="__main__":
    
    # complex_image, train_map, n_classes = Load_dataset_T3(DATASET, USE_CHECKPOINT, SAVE_CHECKPOINT, PDDP_FLAG)   
    complex_image, train_map, n_classes = Load_dataset_image(DATASET, USE_CHECKPOINT, SAVE_CHECKPOINT, PDDP_FLAG, PDDP_TARGET, plot=False)
    # complex_image, train_map, n_classes = Load_dataset(DATASET, USE_CHECKPOINT, SAVE_CHECKPOINT, PDDP_FLAG, PDDP_TARGET, plot=False)   

    # Compute flatten imgs
    complex_image_flatten = np.reshape(complex_image, (-1,complex_image.shape[2],complex_image.shape[3]))
    train_map_flatten = np.reshape(train_map,(-1))
    complex_flatten_filtered = complex_image_flatten[train_map_flatten != 0]
    train_map_flatten_filtered = train_map_flatten[train_map_flatten != 0]
        
    print(f"Dataset: {DATASET} loaded")

    # Set kernel function
    
    match KERNEL_TYPE:
        case 'real':
            kernel_function = module_kernel
        case 'complex':
            kernel_function = complex_kernel
        case 'complex_sym':
            kernel_function = complex_symmetrical_kernel

    #%% Parameter tunning

    print("Parameter tunning...")

    if USE_POWELL:
        
        x0 = np.array((1,1))
        start_time = time.time()
        
        vector_C=[]
        vector_gamma=[]
        vector_func = []

        vector_C_2=[]
        vector_gamma_2=[]
        vector_func_2 = []

        vector_C_3=[]
        vector_gamma_3=[]
        vector_func_3 = []

        args = []
        args.append(complex_flatten_filtered)
        args.append(train_map_flatten_filtered)
        args.append(kernel_function)        

        for pair in combinations(np.unique(train_map_flatten_filtered),2):
            args[0] = complex_flatten_filtered[np.logical_or(train_map_flatten_filtered==pair[0],train_map_flatten_filtered==pair[1])]
            args[1] = train_map_flatten_filtered[np.logical_or(train_map_flatten_filtered==pair[0],train_map_flatten_filtered==pair[1])]        

            sol = pdfo.pdfo(
                    svm_optimization_problem_SB,
                    # try_problem,
                    x0=x0, 
                    args=args,
                    # method="Powell",
                    # bounds=((-4,3),(-1.4,1))
                    bounds=((0.018,55),(0.009,27)),
                    options={'maxfev': 5000}
            )
            vector_C.append(sol.x[0])
            vector_gamma.append(sol.x[1])
            vector_func.append(sol.fun)

            # sol2 = pdfo.pdfo(
            #         svm_optimization_problem_SB_linearized,
            #         # try_problem,
            #         x0=x0, 
            #         args=args,
            #         # method="Powell",
            #         # bounds=((-4,3),(-1.4,1))
            #         # bounds=((0.02,20),(0.05,5)),
            #         options={'maxfev': 5000, 'rhobeg': 1e6}
            # )
            # vector_C_2.append(math.exp(sol2.x[0]))
            # vector_gamma_2.append(0.5 * math.exp(-sol2.x[1]))
            # vector_func_2.append(sol2.fun)

            # sol3 = pdfo.pdfo(
            #         svm_optimization_problem_SB_linearized,
            #         # try_problem,
            #         x0=x0, 
            #         args=args,
            #         method="newuoa",
            #         # bounds=((-4,3),(-1.4,1)),
            #         # bounds=((-4,3),(-9,5)),
            #         # bounds=((0.02,20),(0.05,5)),
            #         options={'maxfev': 5000, 'rhobeg': 1e6}
            # )

            # vector_C_3.append(math.exp(sol3.x[0]))
            # vector_gamma_3.append(0.5 * math.exp(-sol3.x[1]))
            # vector_func_3.append(sol3.fun)

        end_time = time.time()

        print(f"C:{vector_C},    gamma:{vector_gamma},    functional:{vector_func}")
        # print(f"C2:{vector_C_2},    gamma2:{vector_gamma_2},    functional2:{vector_func_2}")
        # print(f"C3:{vector_C_3},    gamma3:{vector_gamma_3},    functional3:{vector_func_3}")

        # SF
        # PDFO normal and bounded
        # C:19.999646807282154,    gamma:4.393900644686247,    functional:0.2525
        # Elapsed time:  527.6078419685364
        # PDFO linearized uobyqa
        # C:1.7384389244009992,    gamma:7.8297855558966445,    functional:0.2985
        # Elapsed time:  3757.362387895584
        # PDFO linearized newuoa
        # C:1.9286615988319975,    gamma:7.899505640594436,    functional:0.2995
        # Elapsed time:  3855.4465351104736
        # sklearn normal bounded method="Powell"
        # C:16.194806934256125,    gamma:4.151997286521436,    functional:0.2535
        # Elapsed time:  2567.53133559227
        # sklearn linearized bounded(implicitly) method="Powell"

        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time) 

    else:
        # C = 2.718
        # gamma = 0.35
        if DATASET=='Flevoland':
            C = 15.892858723624235
            gamma = 3.9358827531948766
        elif DATASET=='San_Francisco':
            # C = 19.999646807282154
            # gamma = 4.393900644686247            
            # C = 1.7384389244009992
            # gamma = 7.8297855558966445    
            C = 1.9286615988319975
            gamma = 7.899505640594436  
        elif DATASET=='Baltrum_Island':
            print('Using gamma and C calculated for 1000 samples per class, complex_symmetrical_kernel')
            vector_C = [
                1.5534315080034524, 0.8980182074803726, 0.8958528935136156, 1.3234968291938622, 0.9201043933724966, 0.6820760378602203, 0.899273970418126, 1.3335669521755285, 1.8597110796824916, 1.5580647303855018, 1.0204982634523605, 1.7082188701027168, 0.49247829026690376, 1.3391743458435834, 1.2575245460928468, 1.7722479454253572, 0.4159955678730737, 0.870049203431152, 1.1578532697155013, 1.2225514146728562, 0.5189232295787986, 1.03090539168909, 2.2760244982287836, 1.2130287589877307, 1.0193246553009974, 1.3430576032866888, 1.5498014650566474, 1.2462780512837908,
                1.0189065252730392, 0.8068355415154345, 1.1073277113815152, 0.9680889028941043, 1.0180539068814463, 1.091067123145986, 1.3812605014834554, 1.0109556809839357, 1.018, 0.5778419301522035, 0.7787557009226063, 1.0203908143283222, 1.0957890341228655, 1.11707869346684, 1.2709907824960003, 1.0068556777940527, 1.925483612662153, 1.967647820177333, 1.0495146046260324, 1.7970449425412223, 1.0096834125332805, 1.1838722062254197, 0.9905294565737481, 0.34276929146434004, 1.3352414923387075, 1.3719035600725422, 2.204302520216902, 0.7896110695191801, 1.0182247988913964,
                1.0754539570717994, 2.540856791840597, 1.4006743123486463, 1.4102528259488225, 0.9399620890447109, 0.5647402984274333, 1.8675680881204695, 1.5796240487399906, 0.8255211817064665
                ]
            vector_gamma = [
                3.7547471535149652, 2.2160614206021894, 3.1918230221617567, 3.1932428069001793, 2.4971602276667424, 4.2725029796803335, 2.824474255584262, 2.942678139439175, 2.2166178127725544, 1.6035577540285166, 2.131214726140104, 2.9606581057356296, 2.95074269127979, 3.7857654141875567, 2.697192721320872, 2.7156220721844253, 3.8429768684897785, 2.592827509881049, 3.0347532752109188, 2.192407938923933, 2.853056797146671, 1.8932383689340986, 2.7819815596245494, 2.0314160934108676,
                2.0289774520503507, 2.2061884320368432, 2.455020266003772, 2.598026267028483, 2.2085778484521623, 2.415463973287333, 2.053950639365205, 2.0060196668498556, 2.209084226172492, 2.1228485393093104, 2.1189203063541533, 2.041097715801574, 2.009, 2.396342373002821, 2.2844514022026794, 2.0099647518143673, 2.0558048706790273, 2.1086111629236344, 3.224688111841129, 2.540939386862531, 1.7211470422699657, 1.4837619385505791, 2.14686525920356, 2.6172402389824208, 2.2325761399068718,
                2.575887736322413, 2.010224930342031, 2.8492521511715454, 3.1659891503816064, 3.9902946936057364, 2.4234014501117693, 1.5900168922778763, 2.0088906132621065, 2.227814897737445, 3.618919276916787, 2.171570065435414, 2.886807072425783, 2.0761312967207135, 2.0238789633277356, 2.679229694934223, 2.4976085393400647, 2.3669745139141325
                ]

    #%% Predict    
    # Predict_batches = 8192
    Predict_batches = 256 * os.cpu_count()   

    start_time = time.time()    

    if USE_MRF:
        lambda_ = [1,1] #handpicked
        spatial_image = np.zeros((complex_image.shape[0], complex_image.shape[1], complex_image.shape[2] + 1, complex_image.shape[3]), dtype=np.float32)
        spatial_image[:,:,:complex_image.shape[2],:] = complex_image
        input_image = spatial_image

        match KERNEL_TYPE:
            case 'real':
                kernel_function = module_kernel_mrf
            case 'complex':
                kernel_function = complex_kernel_mrf
            case 'complex_sym':
                kernel_function = complex_symmetrical_kernel_mrf

    else:
        MRF_ITERATIONS = 1
        MRF_SOLVER = 'ICM'
        OPTIMIZE_LAMBDA = False
        input_image = complex_image

    vote_map = np.zeros((train_map.shape[0],train_map.shape[1], n_classes), dtype=np.uint8)
    classes_labels = np.unique(train_map_flatten_filtered)

    for index, pair in enumerate(combinations(classes_labels,2)):
        print(f'pair:{pair}, gamma:{vector_gamma[index]}, C:{vector_C[index]}')

        if (train_map!=pair[0]).all():
            vote_map[:, pair[1] - 1] +=1
            continue
        elif (train_map!=pair[1]).all():
            vote_map[:, pair[0] - 1] +=1
            continue

        clf = svm.SVC(kernel=kernel_function(gamma=vector_gamma[index]), C=vector_C[index])

        train_map_pair = np.zeros(train_map.shape, dtype=np.uint8)
        train_map_pair[train_map==pair[0]] = pair[0]
        train_map_pair[train_map==pair[1]] = pair[1]
        train_map_pair_flatten = np.reshape(train_map_pair,(-1))
        train_map_pair_flatten_filtered = train_map_pair_flatten[train_map_pair_flatten != 0]

        if USE_MRF:
            energy_min = sys.float_info.max
            classification_optimal = np.zeros((train_map.shape[0],train_map.shape[1]), dtype=np.uint8)

        for iter in range(MRF_ITERATIONS):

            print(f'Iteration:{iter}')
            input_image_flatten = np.reshape(input_image, (-1, input_image.shape[2], input_image.shape[3]))
            input_image_flatten_filtered = input_image_flatten[train_map_pair_flatten != 0]

            clf.fit(X=input_image_flatten_filtered, y=train_map_pair_flatten_filtered)
            #ONLY for try
            # lambda_ = SVM_MRF_HoKashyap(clf, input_image, train_map_pair_flatten)

            y_predicted_flatten = np.zeros((train_map_pair_flatten.shape), dtype=np.uint8) 
            decision_flatten = np.zeros((train_map_pair_flatten.shape), dtype=np.float32)

            input_arrays = [input_image_flatten[
                    decision_flatten.size // Predict_batches * division : decision_flatten.size // Predict_batches * (division + 1)
                    ] for division in range(Predict_batches)]

            if (decision_flatten.size % Predict_batches != 0):
                input_arrays.append(input_image_flatten[decision_flatten.size // Predict_batches * Predict_batches:])

            with PPool(nodes = int(os.cpu_count() * 0.8)) as executor:
                results = executor.map(clf.decision_function, input_arrays)

            for division, result in enumerate(results):
                if division < Predict_batches: 
                    decision_flatten[
                        decision_flatten.size // Predict_batches * division : decision_flatten.size // Predict_batches * (division + 1)
                        ] = result
                else:
                    decision_flatten[decision_flatten.size // Predict_batches * Predict_batches:] = result

            sign_decision = np.sign(decision_flatten)
            y_predicted_flatten[sign_decision == -1] = pair[0]
            y_predicted_flatten[sign_decision == 1] = pair[1]
            decision_image = np.reshape(decision_flatten,(train_map.shape))
            y_predicted_image = np.reshape(y_predicted_flatten,(train_map.shape))

            if (iter<MRF_ITERATIONS - 1): #in the last run it is not necessary
                for i in range(input_image.shape[0]):
                    for j in range(input_image.shape[1]):                    
                        input_image[i,j,-1,0] = EpsilonCompute(y_predicted_image,i,j)

            if OPTIMIZE_LAMBDA:
                if (iter == 0):
                    #calculate lambda_
                    lambda_ = SVM_MRF_HoKashyap_as_Gab(clf, input_image, train_map_pair_flatten, pair)
                    print(f'Lambda:{lambda_}')
                    #remove
                    # lambda_1 = SVM_MRF_HoKashyap(clf, input_image, train_map_pair_flatten, pair)
                    # print(f'Lambda:{lambda_1}')
                    # input_image[:,:,-1,0] *= lambda_[1]#? aca si creo que es??
                    # continue
            
            # input_image[:,:,-1,0] *= lambda_[1]#? aca si creo que es??
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

    np.save(f"./Output/{DATASET}/{KERNEL_TYPE}_powell{USE_POWELL}_pddp{PDDP_TARGET}_mrf_{MRF_SOLVER}{MRF_ITERATIONS}.npy", max_vote)
    plt.imshow(max_vote * 255. / max_vote.max())
    plt.show()
