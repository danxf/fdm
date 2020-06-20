"""
Simple Topic Model example with 20newsgroups dataset. 

Similar to 
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
"""

import numpy as np
import pickle as pkl
from FDM import FDM


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':

    voc_size = 1000
    print('Fetching data...')
    newsgroups_data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes')).data
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features = voc_size,
                                    stop_words='english')
    features = tf_vectorizer.fit_transform(newsgroups_data)
    #dictionary = dict([(j,i) for (i,j) in tf_vectorizer.vocabulary_.iteritems()])
    dictionary = dict([(j,i) for (i,j) in tf_vectorizer.vocabulary_.items()])
    
    
    print('Dictionary size: {}, n_samples: {}'.format(len(dictionary),features.shape[0]))
    

    Ndocs_orig = len(newsgroups_data)
    print('Processing data...')
    #convert sparse matrix count vector represenation to list of lists    
    corpus = [ [i for i in features.getrow(d).nonzero()[1] for _ in range(features[d,i])] for d in range(Ndocs_orig)]
    
    #removing 0-length documents
    corpus = [d for d in corpus if len(d)>0]


    


    print('Building Matrix...')    
    #use the static method to convert from list of lists to the second moment matrix of the corpus
    #data_matrix = FDM.build_data_matrix(corpus, voc_size)
    data_matrix = FDM.build_data_matrix(synthetic_corp,voc_size, method = FDM.MOMENT_COMPUTE_SPARSE_C)
    



    Ntopics = 20

    print('Initializing topics....')
    init_topics = FDM.topic_data_init(corpus, Ntopics, len(dictionary))

    
    print('Training model...')
    #training an FDM model

    fdm = FDM(Ntopics, data_matrix)
    fdm.fit(num_iterations = 30000, init_topics = init_topics)
    #fdm.fit(num_iterations = 30000, init_topics = init_topics,uniform_is_coef = .5)
    
    
    fdm.print_all_topics(dictionary)



