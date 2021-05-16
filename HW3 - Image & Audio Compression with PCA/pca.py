import numpy as np 

def compress(X, k):
    h, w = X.shape
    # Step 1: Find mean vector
    x_mean = np.mean(X, axis=0) 
    
    # Step 2: Subtract mean
    X_hat = X - x_mean
    
    # Step 3: Compute Covariance matrix
    cov_mat = np.cov(X_hat, rowvar=0)
    
    # Step 4: Compute eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    sorted_eig_vals = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_eig_vals]
    eig_vecs = eig_vecs[:, sorted_eig_vals]
    
    # Step 5: Choose k
    #eig_vals_filter = sorted_eig_vals[:k]
    U_k = eig_vecs[:, :k]  
    
    # Step 6: Compute compressed matrix
    Z = X_hat @ U_k 
    return Z, U_k, x_mean

def decode(Z, U_k, x_mean):
    # Reconstruct
    Xapprox = (Z @ U_k.T) + x_mean
    return Xapprox.real


