"""
Communities on Political Blogs graph. 
"""

import numpy as np
import tensorflow as tf


import networkx as nx
import matplotlib.pylab as pl

from FDM import FDM



def ent(p):
    return - ( p*np.log(p) + (1.-p)*np.log(1.-p))


if  __name__ == '__main__':

    GRAPH_PATH = 'example_data/polblogs.gml'
    
    BPG = nx.read_gml(GRAPH_PATH,label = 'id')
    
    voc_size = len(BPG.nodes)
        
    assert (np.array(BPG.nodes) == np.arange(1,voc_size+1)).sum() == voc_size

    graph_labels = np.array([BPG.node[u]['value'] for u  in BPG.nodes ])
    
    
    """
    The transition from a directed graph to a topic model is given as follows:
    For every node, we consider the set of its neighbours (outgoing edges) to be a document. 
    This has the interpretation that a user has a mixed memebrship \theta in a number of communities, and the 
    neighbours are inpenedent samples from the \theta mixture of the communities. 
    Given the set of documents, we proceed as in a regular topic model. 
    """    
    doc_lst = []
    for u in BPG.nodes:
        neighb = [x for x in BPG.neighbors(u)] 
        neighb = [x-1 for x in neighb]
                
        if len(neighb) == 0:
            continue
        
        doc_lst.append(neighb)                
    
    
    adjacency = FDM.build_data_matrix(doc_lst,voc_size)

    

    print('Training model...')
    #training an FDM model
    Ntopics = 10
    fdm = FDM(Ntopics, adjacency)

    fdm.fit(num_iterations = 10000)
    
    res_topics,res_weights = (fdm.get_topics(), fdm.get_weights())
    
    total_topic_weights = res_weights.sum(axis = 0)
    
    #probability that label == 1 given topic i is   (res_topics[i,:] * graph_labels).sum()
    #that probability is expected to be  close to 0 or 1. 
    
    conditional_label_entropy = (
            ent( (res_topics*graph_labels[np.newaxis,:]).sum(axis = 1) ) 
                * total_topic_weights
        ).sum()
    
    print('Conditional label entropy = {:.3f} , ent(0.9) = {:.3f}'.format(conditional_label_entropy,ent(0.9)))
    
    
    #plot two topics with most weight
    w_ord = np.argsort(total_topic_weights)[::-1]
    pl.figure()    
    pl.plot(res_topics[w_ord[0],:],'-o', label = 'Topic {}'.format(w_ord[0]))
    pl.plot(res_topics[w_ord[1],:],'-o', label = 'Topic {}'.format(w_ord[1]))    
    pl.plot((1./20)*graph_labels,'--',linewidth= 5, label = 'Label')    
    pl.legend()
    pl.show()

    






