import spectral as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
from timeit import default_timer as timer

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

def module_kernel(gamma):
    def gaussian(X, Y):
        X_matrix = np.tile(X[:,:,0],X.shape[0])
        X_matrix = np.reshape(X_matrix,(X.shape[0],X.shape[0],-1))
        Y_matrix = np.tile(Y[:,:,0],Y.shape[0])
        Y_matrix = np.reshape(Y_matrix,(Y.shape[0],Y.shape[0],-1))
        Y_matrix_T = np.transpose(Y_matrix, axes=(1,0,2))
        D = np.sum((X_matrix - Y_matrix_T) ** 2, axis=-1)
        return np.exp(- D * gamma)
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
    # C = x[0]
    # gamma = x[1]
    C = math.exp(x[0])
    gamma = 0.5 * math.exp(-x[1])
    clf = svm.SVC(kernel=module_kernel(gamma=gamma), C=C)
    clf.fit(X=args[0][0], y=args[0][1])
    score = clf.score(X=args[0][0], y=args[0][1])
    return score


if __name__=="__main__":
    path_module = "./Input/San_Francisco/San_Francisco_c_3_4"
    module_image = np.asarray(sp.io.envi.open(path_module + '.hdr', path_module + '.raw').asarray())

    path_phase = "./Input/San_Francisco/San_Francisco_phase_difference_3_4"
    phase_diference = np.asarray(sp.io.envi.open(path_phase + '.hdr', path_phase + '.raw').asarray())
    
    path_cross_phase = "./Input/San_Francisco/San_Francisco_c_cross_phase_3_4"
    cross_phase = np.asarray(sp.io.envi.open(path_cross_phase + '.hdr', path_cross_phase + '.raw').asarray())

    path_train_map = "./Input/San_Francisco/TrainMap_SF_3_4"
    train_map =  np.asarray(sp.io.envi.open(path_train_map + '.hdr', path_train_map + '.raw').asarray())

    complex_image = module_phase_to_complex(module_image, phase_diference, cross_phase)

    complex_image_flatten = np.reshape(complex_image, (-1,3,2))
    train_map_flatten = np.reshape(train_map,(-1))

    complex_flatten_filtered = complex_image_flatten[train_map_flatten != 0]
    train_map_flatten_filtered = train_map_flatten[train_map_flatten != 0]
        
    # clf2 = svm.SVC(kernel=try_kernel)
    # clf2.fit(X=complex_flatten_filtered[:,:,0].squeeze(), y=train_map_flatten_filtered)
    # y_predicted = clf2.predict(complex_flatten_filtered[:,:,0].squeeze())
    # print(confusion_matrix(y_pred=y_predicted, y_true=train_map_flatten_filtered))
    
    # clf = svm.SVC(kernel=module_kernel(gamma=1))
    # clf.fit(X=complex_flatten_filtered, y=train_map_flatten_filtered)
    # score = clf.score(X=complex_flatten_filtered, y=train_map_flatten_filtered)
    # # y_predicted = clf.predict(complex_flatten_filtered)
    # # print(confusion_matrix(y_pred=y_predicted, y_true=train_map_flatten_filtered))

    # clf_nu = svm.NuSVC(kernel=module_kernel(gamma=1))
    # clf_nu.fit(X=complex_flatten_filtered, y=train_map_flatten_filtered)
    # score2 = clf_nu.score(X=complex_flatten_filtered, y=train_map_flatten_filtered)
    # # y_predicted_nu = clf_nu.predict(complex_flatten_filtered)
    # # print(confusion_matrix(y_pred=y_predicted_nu, y_true=train_map_flatten_filtered))

    x0 = np.array((1,1))
    start = timer()
    sol = minimize(
            svm_optimization_problem,
            x0=x0, 
            args=[complex_flatten_filtered, train_map_flatten_filtered],
            method="Powell",
            bounds=((-2.3,1.6),(-0.7,1.6))
    )
    stop = timer()
    print(f'time in miliseconds:{stop - start}')

    print(x)
    # plt.imshow(y_predicted/3)
    # plt.show()
    # np.save("./y_predicted.npy", y_predicted)