# FDM
This code is an implementation of the Full Dependence Mixture Topic Model (FDM) model described in Topic Modeling via Full Dependence Mixtures,  Fisher D., Kozdoba M., Mannor S. , 2019 ([arxiv](https://arxiv.org/abs/1906.06181))

## Examples:
* `synthetic_topic_test.py`: Topic reconstruction for random Dirichlet distributed topics and documents. 
* `newsgroup_test.py`: Topic Model example with 20newsgroups dataset. 
* `graph_communities_test.py`: Communities on Political Blogs graph.

## Requirements:
This version of FDM was tested with Python 2.7 and TensorFlow 1.12.0 

graph_communities_test.py requires networkx to parse the graph. 
newsgroup_test.py requires sklearn. 




