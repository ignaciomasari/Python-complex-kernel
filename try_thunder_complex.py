import spectral as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm
from thundersvm import SVC
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
from timeit import default_timer as timer
from HiPart.clustering import PDDP
from HiPart.clustering import DePDDP
from itertools import product, combinations

ICM_ITERATIONS = 3
PDDP_flag = True
use_Powell = True
use_MRF = False

def module_phase_to_complex(module, phase_difference, cross_phase = None):
    module_sqrt = np.sqrt(module)
    phase_difference_rads_1_2 = phase_difference.squeeze() * math.pi / 180 / 2
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

        # X_comp = np.array(np.zeros(shape=(X.shape[0], X.shape[1])), dtype='complex128')
        # X_matrix = np.reshape(X,(X.shape[0],1,-1))
        # Y_matrix = np.reshape(Y,(1,Y.shape[0],-1))
        # D = np.sum((X_matrix - Y_matrix) ** 2, axis=-1)

        return res.real
    return gaussian

def module_kernel_mrf(gamma, lambda_):
    def gaussian(X, Y):
        #not using modules but all values as different channels
        X_matrix = np.reshape(X,(X.shape[0],1,-1))
        Y_matrix = np.reshape(Y,(1,Y.shape[0],-1))

        D = np.sum((X_matrix[:,:,:X_matrix.shape[2] - 2] - Y_matrix[:,:,:Y_matrix.shape[2] - 2]) ** 2, axis=-1)
        return np.exp(- D * gamma) + lambda_ * X_matrix[:,:,X_matrix.shape[2] - 2] * Y_matrix[:,:,Y_matrix.shape[2] - 2]
    return gaussian

def svm_optimization_problem(x, *args): 
    # args[0]: complex_flatten_filtered
    # args[1]: train_map_flatten_filtered
    C = math.exp(x[0])
    gamma = 0.5 * math.exp(-x[1])
    clf = SVC(kernel=complex_kernel(gamma=gamma), C=C, gpu_id=0)
    clf.fit(X=args[0][0], y=args[0][1])
    # CHANGE score to nuSVM upperbound
    score = clf.score(X=args[0][0], y=args[0][1])
    return score

def cluster(image, map, value, target):

    indeces = np.where(map.squeeze() == value)
    X = image[indeces]
    tot_samples = X.shape[0]

    X_reshaped = np.reshape(X,(tot_samples,-1))
    clustering = PDDP(min_sample_split=2, max_clusters_number=int(target)).fit_predict(X_reshaped)#why target no funciona???
    
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
    # CHECK IF CALCULATES OK
    epsilon = 0
    count = 0
    center_class = predicted_image[i,j]
    for coordinate in product(list(range(-1,2)),repeat=2):
        new_i = i + coordinate[0]
        new_j = j + coordinate[1]
        if (new_i >= 0) and (new_i < predicted_image.shape[0]) and (new_j >= 0) and (new_j < predicted_image.shape[1]):
            count +=1
            if predicted_image[i,j] == center_class:
                epsilon +=1
            else:
                epsilon -=1

    # The center point should be disregarded
    epsilon -= 1
    count -=1
    return epsilon/count

if __name__=="__main__":
    #%% Load data
    path_module = "./Input/San_Francisco/San_Francisco_c_3_4"
    module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

    path_phase = "./Input/San_Francisco/San_Francisco_phase_difference_3_4"
    phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
    
    path_cross_phase = "./Input/San_Francisco/San_Francisco_c_cross_phase_3_4"
    cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

    path_train_map = "./Input/San_Francisco/TrainMap_SF_3_4"
    train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())
    n_classes = train_map.max()

    complex_image = module_phase_to_complex(module_image, phase_diference, cross_phase)

    #%% PDDP
    if PDDP_flag:
        target = np.array((2000,2000,2000))
        train_map_PDDP_mask = np.zeros(shape=train_map.squeeze().shape, dtype=bool)
        for i in range(1,n_classes+1):
            selected = cluster(complex_image, train_map, i, target[i-1])
            train_map_PDDP_mask = np.logical_or(train_map_PDDP_mask,selected)
        train_map = train_map * np.expand_dims(train_map_PDDP_mask, axis=-1)

    #%% Compute flatten imgs
    complex_image_flatten = np.reshape(complex_image, (-1,3,2))
    train_map_flatten = np.reshape(train_map,(-1))

    complex_flatten_filtered = complex_image_flatten[train_map_flatten != 0]
    train_map_flatten_filtered = train_map_flatten[train_map_flatten != 0]

    #%% Parameter tunning

    if use_Powell:
  
        x0 = np.array((1,1))
        # start = timer()
        sol = minimize(
                svm_optimization_problem,
                x0=x0, 
                args=[complex_flatten_filtered, train_map_flatten_filtered],
                method="Powell",
                bounds=((-2.3,1.6),(-0.7,1.6))
        )
        # stop = timer()
        # print(f'time in miliseconds:{stop - start}')
        x = sol.x


        # clf = svm.SVC(kernel=module_kernel(gamma=0.35), C=2.718)
        # clf.fit(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        # score = clf.score(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        # # y_predicted = clf.predict(complex_flatten_filtered)
        # # print(confusion_matrix(y_pred=y_predicted, y_true=train_map_flatten_filtered))

        # clf_nu = svm.NuSVC(kernel=module_kernel(gamma=0.35), C=2.718)
        # clf_nu.fit(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        # score2 = clf_nu.score(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        # # y_predicted_nu = clf_nu.predict(complex_flatten_filtered)
        # # print(confusion_matrix(y_pred=y_predicted_nu, y_true=train_map_flatten_filtered))

        # print(f"score SVM:{score}")
        # print(f"score nuSVM:{score2}")

        C = math.exp(x[0])
        gamma = 0.5 * math.exp(-x[1])
    else:
        # C = 2.7
        # gamma = 0.28
        C = 0.10148
        gamma = 0.1032

    #%% Predict
    
    if (use_MRF): # MRF
    
        lambda_ = 0.5 #handpicked

        # clf = svm.SVC(kernel=module_kernel_mrf(gamma=gamma, lambda_=lambda_), C=C)
        clf = svm.SVC(kernel=module_kernel_mrf(gamma=gamma, lambda_=lambda_), C=C)

        # y_predicted = clf.fit_predict(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        spatial_image = np.zeros((complex_image.shape[0], complex_image.shape[1], complex_image.shape[2] + 1, complex_image.shape[3]))
        spatial_image[:,:,:complex_image.shape[2],:] = complex_image

        vote_map = np.zeros((train_map.shape[0],train_map.shape[1], n_classes))

        classes_labels = [x+1 for x in range(n_classes)]

        for pair in combinations(classes_labels,2):
            train_map_pair = np.zeros(train_map.shape)
            train_map_pair[train_map==pair[0]] = pair[0]
            train_map_pair[train_map==pair[1]] = pair[1]
            train_map_pair_flatten = np.reshape(train_map_pair,(-1))
            train_map_pair_flatten_filtered = train_map_pair_flatten[train_map_pair_flatten != 0]

            for iter in range(ICM_ITERATIONS):
                spatial_image_flatten = np.reshape(spatial_image, (-1,4,2))
                spatial_flatten_filtered = spatial_image_flatten[train_map_pair_flatten != 0]

                clf.fit(X = spatial_flatten_filtered, y=train_map_pair_flatten_filtered)

                try:
                    y_predicted_flatten = clf.predict(spatial_image_flatten)
                except MemoryError as e:
                    y_predicted_flatten = np.zeros((train_map_pair_flatten.shape)) 
                    partition = 50
                    for division in range(partition):
                        y_predicted_flatten[
                            y_predicted_flatten.size // partition * division : y_predicted_flatten.size // partition * (division + 1)
                            ] = clf.predict(spatial_image_flatten[
                            y_predicted_flatten.size // partition * division : y_predicted_flatten.size // partition * (division + 1)
                            ]) 
                            
                    y_predicted_flatten[y_predicted_flatten.size // partition * partition:] = clf.predict(spatial_image_flatten[y_predicted_flatten.size // partition * partition:])
                y_predicted_image = np.reshape(y_predicted_flatten,(train_map.shape))

                for i in range(spatial_image.shape[0]):
                    for j in range(spatial_image.shape[1]):
                        
                        spatial_image[i,j,3,0] = EpsilonCompute(y_predicted_image,i,j)

                        if (iter==ICM_ITERATIONS - 1):
                            pass
                # plt.imshow(y_predicted_image * 255. / y_predicted_image.max())
                # plt.show()
                            
            vote_map[y_predicted_image.squeeze() == pair[0],pair[0] - 1] += 1
            vote_map[y_predicted_image.squeeze() == pair[1],pair[1] - 1] += 1


        max_vote = np.argmax(vote_map, axis=2) + 1
        plt.imshow(max_vote * 255. / max_vote.max())
        plt.show()
    else: # No MRF
        
        clf = svm.SVC(kernel=complex_kernel(gamma=gamma), C=C)
        clf.fit(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        score = clf.score(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        y_predicted = clf.predict(complex_flatten_filtered)
        print(confusion_matrix(y_pred=y_predicted, y_true=train_map_flatten_filtered))

        clf = svm.SVC(kernel=module_kernel(gamma=0.35), C=2.718)
        clf.fit(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        score = clf.score(X=complex_flatten_filtered, y=train_map_flatten_filtered)
        y_predicted = clf.predict(complex_flatten_filtered)
        print(confusion_matrix(y_pred=y_predicted, y_true=train_map_flatten_filtered))


    a = 0
    # print(x)
    # plt.imshow(y_predicted/3)
    # plt.show()
    # np.save("./y_predicted.npy", y_predicted)
