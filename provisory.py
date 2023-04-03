import spectral as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
from timeit import default_timer as timer
from HiPart.clustering import PDDP
from HiPart.clustering import DePDDP


# If using intel herdware, use optimization
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("intel optimized")
except ImportError:
    print("no intel extension")

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

def module_kernel(sigma):
    def gaussian(X, Y):
        # X_matrix = np.tile(X[:,:,0],X.shape[0])
        # X_matrix = np.reshape(X_matrix,(X.shape[0],X.shape[0],-1))
        # Y_matrix = np.tile(Y[:,:,0],Y.shape[0])
        # Y_matrix = np.reshape(Y_matrix,(Y.shape[0],Y.shape[0],-1))
        # Y_matrix_T = np.transpose(Y_matrix, axes=(1,0,2))
        # D = np.sum((X_matrix - Y_matrix_T) ** 2, axis=-1)
        X_matrix = np.reshape(X,(X.shape[0],1,-1))
        Y_matrix = np.reshape(Y,(1,Y.shape[0],-1))
        D = np.sum((X_matrix - Y_matrix) ** 2, axis=-1)

        return np.exp(- D * sigma)
    return gaussian

def try_kernel(X,Y):
    print(X.shape)
    print(Y.shape)
    X_matrix = np.tile(X,X.shape[0])
    X_matrix = np.reshape(X_matrix,(X.shape[0],X.shape[0],-1))
    Y_matrix = np.tile(Y,Y.shape[0])
    Y_matrix = np.reshape(Y_matrix,(Y.shape[0],Y.shape[0],-1))
    Y_matrix_T = np.transpose(Y_matrix, axes=(1,0,2))
    D = np.sum((X_matrix - Y_matrix_T) ** 2, axis=-1)
    kernel = np.exp(- D * 1)
    return kernel

def svm_optimization_problem(x, *args): 
    # args[0] = complex_flatten_filtered
    # args[1] = train_map_flatten_filtered
    # probar tambien linealizando estos parametros
    # x[0] = C
    # x[1] = sigma
    clf = svm.SVC(kernel=module_kernel(sigma=x[1]), C=x[0])
    clf.fit(X=args[0][0], y=args[0][1])
    score = clf.score(X=args[0][0], y=args[0][1])
    return score

def cluster(X, target):
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
            
    return bool_list


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
    C = train_map.max()

    complex_image = module_phase_to_complex(module_image, phase_diference, cross_phase)

    complex_image_flatten = np.reshape(complex_image, (-1,3,2))
    train_map_flatten = np.reshape(train_map,(-1))

    complex_flatten_filtered = complex_image_flatten[train_map_flatten != 0]
    train_map_flatten_filtered = train_map_flatten[train_map_flatten != 0]

    #%% PDDP
    target = np.array((2000,2000,2000))
    train_map_PDDP_mask = np.zeros(shape=train_map_flatten_filtered.shape, dtype=bool)
    for i in range(1, C + 1):
        selected_centers = cluster(complex_flatten_filtered[train_map_flatten_filtered == i], target[i-1])
        # selected = PDDP(train_map_flatten_filtered[train_map_flatten_filtered == i], target[i-1])
        count = 0
        for idx in range(train_map_flatten_filtered.size):
            if train_map_flatten_filtered[idx]==i:
                train_map_PDDP_mask[idx] = selected_centers[count]
                count +=1
            if count == selected_centers.size:
                break
        breakpoint = True



    #%% Run SVM
    
    # clf = svm.SVC(kernel=module_kernel(sigma=1))
    # clf.fit(X=complex_flatten_filtered[train_map_PDDP_mask], y=train_map_flatten_filtered[train_map_PDDP_mask])
    # clf.fit(X=complex_flatten_filtered, y=train_map_flatten_filtered)
    # score = clf.score(X=complex_flatten_filtered, y=train_map_flatten_filtered)
    # y_predicted = clf.predict(complex_flatten_filtered)
    # print(confusion_matrix(y_pred=y_predicted, y_true=train_map_flatten_filtered))

    # clf_nu = svm.NuSVC(kernel=module_kernel(sigma=1))
    # clf_nu.fit(X=complex_flatten_filtered, y=train_map_flatten_filtered)
    # score2 = clf_nu.score(X=complex_flatten_filtered, y=train_map_flatten_filtered)
    # y_predicted_nu = clf_nu.predict(complex_flatten_filtered)
    # print(confusion_matrix(y_pred=y_predicted_nu, y_true=train_map_flatten_filtered))

    x0 = np.array((1,1))
    start = timer()
    x = minimize(
            svm_optimization_problem,
            x0=x0, 
            args=[complex_flatten_filtered[train_map_PDDP_mask], train_map_flatten_filtered[train_map_PDDP_mask]],
            method="Powell",
            bounds=((0.001,10),(0.001,10))
    )
    stop = timer()
    print(f'time in miliseconds:{stop - start}')

    print(x)
    a = 0
    # plt.imshow(y_predicted/3)
    # plt.show()
    # np.save("./y_predicted.npy", y_predicted)