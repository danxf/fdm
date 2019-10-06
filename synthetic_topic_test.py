import cPickle as pkl
from FDM import FDM
import numpy as np


import matplotlib.pylab as pl


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
    
    #creating FDM model    
    fdm = FDM(50,data_matrix)
    #fitting FDM model
    fdm.fit(50000)
    inferred_topics = fdm.get_topics()
    
    #measuring reconstruction L1 error    
    reconstruction_errors = np.array(
        [np.abs( inferred_topics - gt_topics[gt_idx,:][np.newaxis,:] ).sum(axis = 1).min() for gt_idx in range(Ntopics)]
    )
    average_error = reconstruction_errors.mean()    

    print ('average reconstruction error: {:.3f}'.format(average_error) )
    
    
    
    minimizing_inidces = [np.argmin(np.abs( inferred_topics - gt_topics[gt_idx,:][np.newaxis,:] ).sum(axis = 1)) 
                            for gt_idx in range(Ntopics)
                         ]

    #plotting a random ground truth topic with the closest (in L1 metric) inferred topic
    pl.figure()    
    pl.plot(gt_topics[0,:],'-o', label = 'Ground Topic {}'.format(0))
    pl.plot(inferred_topics[minimizing_inidces[0],:],'-o', 
            label = 'Inferred Topic {}, dist = {:.3f}'.format(minimizing_inidces[0], reconstruction_errors[0])
           )
               
    pl.legend()
    pl.show()
        
    
    
    
    
    
    
