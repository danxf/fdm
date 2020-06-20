import pickle as pkl
from FDM import FDM
import numpy as np


import matplotlib.pylab as pl

from scipy.spatial.distance import cdist 
from scipy.optimize import linear_sum_assignment    


if __name__ == '__main__':
        
    Ntopics = 20
    voc_size = 250
    doc_size = 30
    beta_prior = 10.0 / voc_size * np.ones(voc_size)
    alpha_prior = 1.0 / Ntopics * np.ones(Ntopics) 
    
    
    print('Generating synthetic topics...')
    gt_topics = np.random.dirichlet(beta_prior, size = Ntopics)
    
    
    print('Generating synthetic corpus')
    
    Ndocs = 50000
    
    thetas = np.random.dirichlet(alpha_prior, size = Ndocs)
    document_mixtures = np.dot(thetas,gt_topics)

    synthetic_corp = [list(np.random.choice(voc_size, p=document_mixtures[i,:], size = doc_size)) for i in range(Ndocs)]
    
        
    #creating second moment matrix from the synthetic corpus
    print('Building second moment matrix')
    data_matrix = FDM.build_data_matrix(synthetic_corp,voc_size)
    #data_matrix = FDM.build_data_matrix(synthetic_corp,voc_size, method = FDM.MOMENT_COMPUTE_SPARSE_C)
    
    #creating FDM model    
    fdm = FDM(Ntopics,data_matrix,learning_rate=1e-2)
    
    #fitting FDM model
    init_topics = FDM.topic_data_init(synthetic_corp, Ntopics, voc_size)
    
    
    
    fdm.fit(50000, init_topics = init_topics)
    
    inferred_topics = fdm.get_topics()
    
    
    
    dist_matrix = cdist(gt_topics,inferred_topics, 'cityblock')
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    minimizing_inidces = col_ind[np.argsort(row_ind)]
    
    reconstruction_errors = dist_matrix[np.arange(Ntopics),minimizing_inidces]
    average_error = reconstruction_errors.mean()
    print ('average reconstruction error: {:.3f}'.format(average_error) )
    
    
    #plotting a random ground truth topic with the closest (in L1 metric) inferred topic
    pl.figure()    
    pl.plot(gt_topics[0,:],'-o', label = 'Ground Topic {}'.format(0))
    pl.plot(inferred_topics[minimizing_inidces[0],:],'-o', 
            label = 'Inferred Topic {}, dist = {:.3f}'.format(minimizing_inidces[0], reconstruction_errors[0])
           )
               
    pl.legend()
    pl.show()
        
    
    
    
    
    
    
