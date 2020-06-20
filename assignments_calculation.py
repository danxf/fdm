"""
This file contains method for topic assignment calculation on data.
Use calculate_assignments() for small data. (this method will return D x T dense matrix where D is the number of documents and T is the number of topics)
Use calculate_assignments_sparse() for large data. This will return D x T _sparse_ matrix which i s an approximation of the matrix that is returned from calculate_assignments(). The approximation results from converting negligeble values to 0. Note that this might lead for a better approximation of the documents since the NMF solver from scikit is numerical and will not reach 0 even if needed.

"""


import numpy as np
import pickle as pkl
from FDM import FDM

from sklearn.decomposition import non_negative_factorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

def _get_empirical_dist(data, voc_size):
    cv = CountVectorizer(token_pattern='[0-9]+', vocabulary = map(lambda x: str(x), range(voc_size)) ) 
    csr_counts = cv.fit_transform([' '.join(map(lambda x:str(x), doc)) for doc in data])
    return normalize(csr_counts, norm= 'l1', axis=1)


def calculate_assignments(topics, data, voc_size, iterations = 1000):
    """
    Parameters:
        data : list of lists of int
        iterations : int
            maximum number of iteration for NMF (optional)
            
    Returns:
        assignments : np array
            assignments[i,:] is the proportions of each topic in document i
    """
    nmf_solver = non_negative_factorization(X = _get_empirical_dist(data, voc_size),
                                H = topics,
                                max_iter = iterations,
                                solver = 'mu',
                                beta_loss = 'kullback-leibler',
                                init = 'custom',
                                update_H = False,
                                n_components = topics.shape[0])
    return nmf_solver[0]
    


def _csr_vappend(a,b):
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a

from time import time
def calculate_assignments_sparse(topics, data, voc_size, iterations = 1000, threshold = 1e-4):
    """
    This function should be used when theres a lot of documents and the vocabulary size is large
    Parameters:
        data : list of lists
        iterations : int
            maximum number of iteration for NMF (optional)
            
    Returns:
        assignments : CSR matrix
            assignments[i,:] is the proportions of each topic in document i where topics with low probability are removed
    """        
    #calulate block size
    Ndocs_batch = (50000*10000) // voc_size #fits in 4GB of memory
    
    Nbatches = len(data) // Ndocs_batch
    if Nbatches*Ndocs_batch < len(data):
        Nbatches += 1
    
    start_time = time()
    for i in range(Nbatches):
        
        
        partial_assignments = calculate_assignments(topics, data[i*Ndocs_batch:(i+1)*Ndocs_batch], voc_size, iterations)
        partial_assignments[partial_assignments < threshold] = 0 
        #re-normalize
        partial_assignments /= partial_assignments.sum(axis=1)[:,np.newaxis]
        
        if i==0:
            sparse_assignments = csr_matrix(partial_assignments)
        else: 
            sparse_assignments = _csr_vappend(sparse_assignments, csr_matrix(partial_assignments))
    

        print('Done batch {} out of {}. Elapsed {:.2f} min.'.format(i,Nbatches,   (time()-start_time)/60  ))
    
    return sparse_assignments
        

