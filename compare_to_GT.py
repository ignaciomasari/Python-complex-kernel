import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

prediction_path = './Output/Baltrum_Island_L_S_reduced/complex_sym_powellFalse_pddp1000_mrf_GC3_lambTrue.npy'
ground_truth_path = './Input/Baltrum_Island_L_S_reduced/BI_label_test_corrected.npy'

# Convert the images to numpy arrays
prediction_array = np.load(prediction_path)
ground_truth_array = np.load(ground_truth_path)

# Flatten the arrays to 1D arrays
prediction_flat = prediction_array.flatten()
ground_truth_flat = ground_truth_array.flatten()

# Get the confusion matrix
confusion_mat = confusion_matrix(ground_truth_flat, prediction_flat)

confusion_mat = confusion_mat[1:, 1:]

# calculate the following indices: User's accuracy and Producer's accuracy by class. Average accuracy, Overall accuracy, Kappa coefficient for the whole image
# User's accuracy = TP / (TP + FP[row_total])
UA = np.diag(confusion_mat) / np.sum(confusion_mat, axis=0) * 100

# Producer's accuracy = TP / (TP + FN[column_total])
PA = np.diag(confusion_mat) / np.sum(confusion_mat, axis=1) * 100

AA = np.mean(UA)

OA = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat) * 100

# Kappa coefficient
N = np.sum(confusion_mat)
p0 = OA / 100
pe = np.sum(np.sum(confusion_mat, axis=0) * np.sum(confusion_mat, axis=1)) / (N * N)
kappa = (p0 - pe) / (1 - pe)


# Print the results with only two decimals points
print('User\'s accuracy: ', np.round(UA, 2))
print('Producer\'s accuracy: ', np.round(PA, 2))
print('Average accuracy: ', np.round(AA, 2))
print('Overall accuracy: ', np.round(OA, 2))
print('Kappa coefficient: ', np.round(kappa, 2))

if False:
    # Plot both images with the same colorbar
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(prediction_array)
    ax[0].set_title('Prediction')
    ax[1].imshow(ground_truth_array)
    ax[1].set_title('Ground truth')
    plt.show()
