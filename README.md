# FDM
This code is an implementation of the Full Dependence Mixture Topic Model (FDM) model described in:
Topic Modeling via Full Dependence Mixtures,  Fisher D., Kozdoba M., Mannor S. , ICML 2020
([arxiv](https://arxiv.org/abs/1906.06181))

## Examples:
* `synthetic_topic_test.py`: Topic reconstruction for random Dirichlet distributed topics and documents. 
* `newsgroup_test.py`: Topic Model example with 20newsgroups dataset. 
* `graph_communities_test.py`: Communities on Political Blogs graph.

## Requirements:
This version of FDM was tested with Python 3.6 and TensorFlow 2.0.0

newsgroup_test.py requires sklearn to load and parse 20newsgroups
graph_communities_test.py requires networkx to parse the graph. 


Moment matrices are by default computed via a python implementation. 
To use the faster Cython implementation: 

#Install cython:
pip install cython 
#compile _fast_moment.pyx 
python3 fast_moments_build.py build_ext --inplace

Then use FDM.build_data_matrix with  parameter method=FDM.MOMENT_COMPUTE_SPARSE_C. 





