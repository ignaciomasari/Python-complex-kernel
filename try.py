import numpy as np
import pandas as pd

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

    return Lambda

def HoKashyap2(matrix, max_iterations):
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
    
    Lambda_final = np.ones(col)
    if Lambda.min() > 0:
        for i, idx in enumerate(Lambda):
            Lambda_final[i] = int(10 * min(idx / Lambda.min(), 10))
        

    return Lambda_final

if __name__== "__main__" :

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('./Input/try_mat.csv', header=None)

    # Convert the DataFrame to a numpy array
    matrix = df.values

    lambda_ = HoKashyap2(matrix.T, max_iterations=1000000)

    print(lambda_)

    a = 0
