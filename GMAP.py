import pathlib
import math
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import rasterio as rio
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.profiles import DefaultGTiffProfile

from complex_kernel import Load_dataset

USE_CHECKPOINT = False
SAVE_CHECKPOINT = False
PDDP_FLAG = False
DATASET = 'Flevoland'#'San Francisco', 'Flevoland'

def load_image(path):
    """Load a GeoTiff image into a numpy array."""
    with rio.open(path, 'r') as dataset:
        raster = dataset.read()
        
    if np.ndim(raster) == 2:
        return raster
    elif np.ndim(raster) == 3 and np.shape(raster)[0] == 1:
        return np.squeeze(raster)
    else:
        return reshape_as_image(raster)
    
def save_image(path, image):
    """Write an image to file in GeoTiff format."""
    path.parent.mkdir(parents=True, exist_ok=True) 
    n_dim = np.ndim(image)
    n_rows = np.shape(image)[0]
    n_cols = np.shape(image)[1]
    if n_dim == 2:
        image_rio = np.reshape(image, (1,n_rows,n_cols))
        n_bands = 1
    elif n_dim == 3:
        image_rio = reshape_as_raster(image)
        n_bands = np.shape(image)[2]
    else:
        raise ValueError("Expected array dimension to be 2 or 3. Received {}.".format(n_dim))
    profile_out = DefaultGTiffProfile()
    profile_out["count"] = n_bands
    profile_out["height"] = n_rows
    profile_out["width"] = n_cols
    profile_out["transform"] = rio.transform.IDENTITY
    with rio.open(path, 'w', **profile_out) as dst:
        dst.write(image_rio)

def extract_labelled_samples(image, gt):
    """Return all the labelled samples. In gt, the value 0 means no label."""
    if np.ndim(gt) != 2:
        raise ValueError("The GT must be a 2D image.")
    if np.shape(image)[0:2] != np.shape(gt):
        raise ValueError("Array and GT must have the same size")
    
    labelled_samples = {}
    labels = np.unique(gt)
    for label in labels:
        if label != 0:
            samples = image[gt==label]
            labelled_samples[label] = samples
    
    return labelled_samples

def estimate_gaussian_models(training_set, no_prior=False):
    """Return all the models. Mean and Covariance matrices"""
    models = {}
    tot_samples = 0
    for label, samples in training_set.items():
        if np.ndim(samples) == 1:
            samples = np.reshape(samples, [-1,1])
        num_samples, num_features = np.shape(samples)
        param = {}
        param["mean"] = np.mean(samples, axis=0)
        param["cov"] = np.cov(samples, rowvar=False)
        param["num_samples"] = num_samples
        param["num_features"] = num_features
        models[label] = param
        tot_samples += num_samples
        
    for label in models.keys():
        if no_prior:
            models[label]["prior_prob"] = 1    
        models[label]["prior_prob"] = models[label]["num_samples"] / tot_samples
        
    return models

def multivariate_classify(image, models):
    """Return the classification map in case of multivariate data."""
    index_to_label = {idx:label for idx, label in enumerate(models.keys())}
    
    discriminant_function_list = []    
    for label, model in models.items():
        mean = model["mean"]
        cov = model["cov"]
        add = np.eye(np.shape(cov)[0])*0.001
        cov = cov + add
        p_i = model["prior_prob"]
        g = multivariate_normal.logpdf(image, mean=mean, cov=cov)
        discriminant_function_list.append(g+math.log(p_i))
    discriminant_function = np.stack(discriminant_function_list, axis=0) 
    
    c_map_with_index = np.argmax(discriminant_function, axis=0)
    classification_map = np.zeros_like(c_map_with_index)
    for idx, lab in index_to_label.items():
        classification_map[c_map_with_index==idx] = lab
    return classification_map

def univariate_classify(image, models):
    """Return the classification map in case of univariate data."""
    index_to_label = {idx:label for idx, label in enumerate(models.keys())}
    
    discriminant_function_list = []    
    for label, model in models.items():
        mean = float(model["mean"])
        var = float(model["cov"])
        p_i = float(model["prior_prob"])
        g = norm.logpdf(image, loc=mean, scale=var)
        discriminant_function_list.append(g+math.log(p_i))
    discriminant_function = np.stack(discriminant_function_list, axis=0) 
    
    c_map_with_index = np.argmax(discriminant_function, axis=0)
    classification_map = np.zeros_like(c_map_with_index)
    for idx, lab in index_to_label.items():
        classification_map[c_map_with_index==idx] = lab
    return classification_map

def classify_image(image, models):
    """Return the classification map in case of univariate or multivariate data."""
    if np.ndim(image) == 2:
        return univariate_classify(image, models)
    elif np.ndim(image) == 3:
        if np.shape(image)[2] == 1:
            return univariate_classify(image[:,:,0], models)
        else:
            return multivariate_classify(image, models)
    else:
        raise ValueError("Image dimensions must be either 2 or 3.")   
    
def split_test_train(input_gt, split_ratio):
    train_set = np.zeros_like(input_gt)
    test_set = np.zeros_like(input_gt)
    labels = np.unique(input_gt)
    for label in labels:
        if label == 0:
            continue
        indexes = np.array(np.where(input_gt == label))
        X_train, X_test= train_test_split(indexes.T, test_size=split_ratio)
        test_set[X_test[:,0],X_test[:,1]] = label
        train_set[X_train[:,0],X_train[:,1]] = label

    return train_set, test_set

def compute_confusion_matrix(c_map, gt):
    """Return the confusion matrix given a classification map and a ground truth."""
    gt = gt.ravel()
    c_map = c_map.ravel()
    y_pred = c_map[gt != 0]
    y_true = gt[gt != 0]
    return confusion_matrix(y_true, y_pred)    
    
def compute_classification_report(c_map, gt):
    """Return a classification report given a classification map and a ground truth."""
    gt = gt.ravel()
    c_map = c_map.ravel()
    y_pred = c_map[gt != 0]
    y_true = gt[gt != 0]
    return classification_report(y_true, y_pred, output_dict=True)

def export_to_xlsx(path_str, data):
    """Export some data to an Excel file."""
    path = pathlib.Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True) 
    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path_str, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

if __name__=='__main__':

    complex_image, train_map, n_classes = Load_dataset(DATASET, USE_CHECKPOINT, SAVE_CHECKPOINT, PDDP_FLAG)

    bands_image = np.reshape(complex_image,(complex_image.shape[0],complex_image.shape[1],-1))

    training_set = extract_labelled_samples(bands_image,train_map.squeeze())

    models = estimate_gaussian_models(training_set)

    classification_map = classify_image(bands_image, models)

    np.save("./Output/Flevoland/complex_MAP_complete.npy", classification_map)
    plt.imshow(classification_map)
    plt.show()
