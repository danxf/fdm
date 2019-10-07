import numpy as np
import tensorflow as tf
from itertools import product
import cPickle as pkl


class FDM:
    def __init__(self,T,data,learning_rate=0.01,regularizer_coef=0.0):
        """
        Parameters
        ----------
        T : int
            Number of topics to fit
        data : Numpy 2-d array
            Numpy array that sums to 1 representing token-token co-occurence in the data.
            Can be constructed using FDM.build_data_matrix()
        learning_rate : float, optional
            The learning rate for the Adam optimizer (default is 0.01)
        regularizer_coef : float, optional
            Coefficient to determine how much to penalize the co-occurence of *different* topics in the same document,
            negative values will *reward* co-occurence. (default is 0 which disregrads the weights distribution)
        """
        self.T = T
        self.learning_rate = learning_rate
        self.regularizer_coef = regularizer_coef
        self.data_to_fit = data
        self.voc_size = self.data_to_fit.shape[0]
        self.data_reshaped = self.data_to_fit.reshape((self.voc_size * self.voc_size)) #flattened data matrix used for SGD sampling

        self._tf_session = None
        self._loss_arr = []
        self.graph = None 
        self.t_tfinit = None 
        self.t_loss = None 
        self.t_topics = None 
        self.t_train_op = None 
        self.t_weights = None 
        self.chosen_index_1 = None 
        self.chosen_index_2 = None 
        self.tf_saver = None 
        self.global_iteration_counter = 0
        self.batch_size = None
        
        self.topics = None
        self.weights = None    
    
    def get_topics(self):
        return self.topics
    def get_weights(self):
        return self.weights
    
    def fit(self, num_iterations=100000, batch_size = 512, batches_to_sample = 8000):
        """
        Parameters
        ----------
        num_iterations : int (optional)
            Number of SGD steps to train the model (default is 100K)
        batch_size : int (optional)
            number of variables to optimize in each SGD step
        batches_to_sample : int (optional)
            the number of batches to sample before performing the SGD steps, low values result in inefficient GPU utilization
            this value can be optimized by observing GPU usage during training (default is 8K). See also Sampler class below.
        """
        
        
        self.topics = np.zeros((self.T,self.voc_size))
        self.weights = np.zeros((self.T,self.T))
        self.batch_size = batch_size
        #build graph
        (self.graph,self.t_tfinit,self.t_loss,self.t_topics,self.t_train_op,
                self.t_weights,self.chosen_index_1,self.chosen_index_2,
                self.tf_saver) = FDM._build_graph(self.T,self.voc_size,batch_size,self.regularizer_coef,self.learning_rate)
        
        #start session
        self._tf_session = tf.Session(graph = self.graph) 
        self._tf_session.run(self.t_tfinit)
        
        #train model
        self._do_iterations(num_iterations,batch_size,batches_to_sample)
        
        
        
    def continue_model_training(self,num_iterations=100000,batch_size = 512, batches_to_sample = 8000):
        #check if model was initialized
        if self._tf_session is None:
            print ('Cannot continue uninitialized model.\nUse FDM.fit() first, or load an initialized FDM model.')
            return
        
        #continue training
        self._do_iterations(num_iterations,batch_size,batches_to_sample)


    """
    This class is used to pool random samples. 
    It is more efficient to call np.random.choice with a large size parameter, versus calling np.random.choice 
    size times with size=1.  
    This class implements such pooling. 
    """
    class Sampler:
        def __init__(self, sampling_dist, batch_size, batches_to_sample):
            self.batch_size = batch_size
            self.batches_to_sample = batches_to_sample
            self.sampling_dist = sampling_dist
            self.samples = np.random.choice(range(sampling_dist.shape[0]), p = self.sampling_dist, size = (self.batches_to_sample,self.batch_size))
            self.fetch_sample_idx = 0
        
        def get_sample(self):
            if self.fetch_sample_idx < self.batches_to_sample:
                self.fetch_sample_idx += 1 
                return self.samples[self.fetch_sample_idx - 1,:]
            else:
                self.samples = np.random.choice(range(self.sampling_dist.shape[0]), p = self.sampling_dist, size = (self.batches_to_sample,self.batch_size))
                self.fetch_sample_idx = 1
                return self.samples[self.fetch_sample_idx - 1,:]
                
        
    def _do_iterations(self,num_iterations=100000,batch_size = 512, batches_to_sample = 8000):
        local_iteration_counter = 0
        sampler = FDM.Sampler(self.data_reshaped, batch_size, batches_to_sample)
        
        while local_iteration_counter < num_iterations:
            #sample batch
            batch = sampler.get_sample()
            #prepare batch
            pair_0 = batch / self.voc_size
            pair_1 = batch % self.voc_size
            
            #do SGD steps
            res = self._tf_session.run([self.t_train_op,self.t_loss],
                                        feed_dict = {self.chosen_index_1:pair_0,
                                                    self.chosen_index_2:pair_1 }
                                        )
            
            self.global_iteration_counter += 1    
            local_iteration_counter += 1
            
            #update topics and weights data members
            if local_iteration_counter%1000 == 0:
                print ('local iteration no. {} / {}'.format(local_iteration_counter,num_iterations))
                self._loss_arr.append(res[1])
                self.topics = self._tf_session.run(self.t_topics)
                self.weights = self._tf_session.run(self.t_weights)
        
        #at the end of the training we update to the latest value
        self.topics = self._tf_session.run(self.t_topics)
        self.weights = self._tf_session.run(self.t_weights)        

    
    def print_topic(self, topicNumber, dictionary = None, nwords = 20, topic_threshold=1e-5):
        """
        Parameters
        ----------
        topicNumber : int
            the topic index to be printed (must be in the range [0,..,T-1])
        dictionary : (token_index --> token_string) (optional)
            a mapping between token indices and the tokens themselves,
            if no map is provided the indices themselves will be printed
        nwords : int (optional)
            the maximum (see below) number of top words to be printed. top words are the most probable words in the topic.
        topic_threshold : float (optional)
            the minimum probability needed to print the token. tokens with probability lower than
            topic_threshold will not be printed even though they are in the top nwords.
        """
        
        #using default mapping  x (int) --- > "x" (str) if none is provided
        if dictionary is None:
            dictionary = dict(zip(range(self.voc_size),map(str,range(self.voc_size)) ))
        
        Ndict = len(dictionary)
        topic_vec = self.topics[topicNumber,:]
        topic_vec = sorted( zip(xrange(Ndict),topic_vec), key = lambda k: -k[1] )

        topic_vec = [ (dictionary[i],w) for i,w in topic_vec if w>topic_threshold]

        for a in topic_vec[:nwords]:
            print ('{:14}{:.4f}'.format(a[0],a[1]) )
        print ('')
            
    def print_all_topics(self,dictionary = None, nwords = 20, topic_threshold=1e-5):
        for t in range(self.T):
            print ('Topic number {}:'.format(t))
            self.print_topic(t, dictionary, nwords, topic_threshold)
    
    def save_model(self,model_name, save_dir = ''):
        """
        Parameters
        ----------
        model_name : str
            file name prefix
        save_dir : str (optional)
            save directory for the model (default is current directory)
            
        """
        
        #save tf session using tf.saver
        self.tf_saver.save(self._tf_session,save_dir + '/'+model_name+'fdm_tf_session')
        #save model parameters
        with open(save_dir + '/'+model_name+'fdm_state.txt','w') as config_fd:
            config_fd.write('{}\n'.format(self.T))
            config_fd.write('{}\n'.format(self.learning_rate))
            config_fd.write('{}\n'.format(self.regularizer_coef))
            config_fd.write('{}\n'.format(self.global_iteration_counter))
            config_fd.write('{}\n'.format(self.batch_size))
        #save likelihood over time
        with open(save_dir + '/' + model_name + 'fdm_loss.txt','w') as loss_fd:
            loss_fd.write('\n'.join(map(str,self._loss_arr)))
        #save data matrix 
        with open(save_dir + '/' + model_name + 'data_matrix.pkl','wb') as pklfd:
            pkl.dump(self.data_to_fit, pklfd, -1)
            
        
        
    @staticmethod
    def load_model_params(model_name, load_dir = ''):
        #load parameters
        params = []
        with open(load_dir + '/'+model_name+'fdm_state.txt','r') as config_fd:
            for line in config_fd:
                params.append(line[:-1])
        (T,learning_rate,regularizer_coef,global_iteration_counter,batch_size) = (
            int(params[0]),float(params[1]),float(params[2]),int(params[3]),int(params[4]))
        #load likelihoods over time
        loss_arr = []
        with open(load_dir + '/'+model_name+'fdm_loss.txt','r') as loss_fd:
            for line in loss_fd:
                loss_arr.append(float(line[:-1]))
        #load data_matrix
        with open(load_dir + '/' + model_name + 'data_matrix.pkl', 'rb') as pklfd:
            data_matrix = pkl.load(pklfd)
        
        return (T, learning_rate, regularizer_coef, global_iteration_counter, batch_size, loss_arr, data_matrix)
    

    @classmethod
    def load_model_from_file(cls, model_name, load_dir = ''):
        """
        Returns:
        --------
        An FDM object withthe same state as if it was created with the same data and ran
        with the same number of iterations using FDM.fit()
        """
        
        
        (T, learning_rate, regularizer_coef, global_iteration_counter, batch_size, loss_arr, data_matrix) = FDM.load_model_params(model_name, load_dir)
        loaded_fdm = cls(T,data_matrix,learning_rate=learning_rate,regularizer_coef=regularizer_coef)
        loaded_fdm._loss_arr = loss_arr
        loaded_fdm.global_iteration_counter = global_iteration_counter
        loaded_fdm.batch_size = batch_size
        #load session build graph
        
        
        #build graph
        (loaded_fdm.graph, loaded_fdm.t_tfinit, loaded_fdm.t_loss, loaded_fdm.t_topics, loaded_fdm.t_train_op,  
                loaded_fdm.t_weights, loaded_fdm.chosen_index_1, loaded_fdm.chosen_index_2, 
                loaded_fdm.tf_saver
                )=FDM._build_graph(loaded_fdm.T,loaded_fdm.voc_size,loaded_fdm.batch_size,loaded_fdm.regularizer_coef,loaded_fdm.learning_rate)
        
        #start session
        loaded_fdm._tf_session = tf.Session(graph = loaded_fdm.graph) 
        loaded_fdm.tf_saver.restore(loaded_fdm._tf_session, load_dir + '/' + model_name + 'fdm_tf_session')
        
        
        loaded_fdm.topics = loaded_fdm._tf_session.run(loaded_fdm.t_topics)
        loaded_fdm.weights = loaded_fdm._tf_session.run(loaded_fdm.t_weights)
        
        return loaded_fdm
        
            
        
        
        
    @staticmethod
    def build_data_matrix(corpus,voc_size):
        """
        Parameters
        ----------
        corpus : list of lists of int
            A list represeting the corpus where each list corpus[i] is a list of ints representing tokens in the range {0,...,voc_size-1}
        voc_size : int 
            The number of unique tokens in the corpus
            
        Returns
        -------
        second_moment_matrix : Numpy array of shape (voc_size,voc_size)
            Empricial FDM matrix. The (i,j) entry represents the probability of observing the tokens i and j in the same document, where the documents are sampled uniformly at random from the corpus.
        """
        
        FDM_M_matrix = np.zeros((voc_size,voc_size))
        
        for i,d in enumerate(corpus):
            d_emp_dist = np.zeros(voc_size)
            d_len=np.float(len(d))
            
            if d_len == 1:
                continue
            
            non_zero_idxs = set()
            for w in d:
                d_emp_dist[w]+=1
                non_zero_idxs.add(w)
            d_emp_dist /= d_len
            
            #creating FDM empirical matrix for each document d
            d_emp_prod = np.zeros((voc_size,voc_size))
            for (w,v) in product(non_zero_idxs,repeat=2):
                if v!=w:
                    FDM_M_matrix[w,v] += d_emp_dist[w]*d_emp_dist[v]* (d_len/(d_len-1))
                else:
                    FDM_M_matrix[w,w] += (d_len/(d_len-1))*(d_emp_dist[w]*d_emp_dist[w] - (1/d_len)*d_emp_dist[w])
                    
        #The empirical FDM matrix of the corpus is the average of the empirical FDM of each of the documents
        FDM_M_matrix /= FDM_M_matrix.sum()   
        return FDM_M_matrix         
        
    
    
    
    @staticmethod
    def _build_graph(Npartitions,
                    voc_size,
                    batch_size,
                    gamma_regularizer,
                    learning_rate):
        
        graph=tf.Graph()
        with graph.as_default():
            
            chosen_index_1 = tf.placeholder(dtype=tf.int32,shape=(batch_size))
            chosen_index_2 = tf.placeholder(dtype=tf.int32,shape=(batch_size))
            
            t_weights_free = tf.Variable(tf.truncated_normal([Npartitions,Npartitions],mean=1.0,stddev=1.0),dtype=tf.float32)
            t_weights_free_sym = t_weights_free + tf.transpose(t_weights_free)
            t_weights = tf.reshape( tf.nn.softmax(tf.reshape(t_weights_free_sym,[Npartitions*Npartitions])) , [Npartitions,Npartitions])
            
            t_topics_free = tf.Variable(tf.truncated_normal([Npartitions,voc_size],mean=1.0,stddev=1.0),dtype=tf.float32)
            t_topics = tf.nn.softmax(t_topics_free) #default axis is (-1)

            
            
            t_gamma = gamma_regularizer
            
            pre_target =  tf.log((
                    tf.reduce_sum ((
                        tf.matmul(
                            tf.expand_dims(tf.transpose(tf.gather(t_topics,chosen_index_1,axis=1)),-1),
                            tf.expand_dims(tf.transpose(tf.gather(t_topics,chosen_index_2,axis=1)),1)
                            ) 
                    * t_weights),axis=[1,2])
                    )) 
            target = tf.reduce_mean( 
                        tf.where(tf.is_nan(pre_target), tf.zeros_like(pre_target),pre_target )
                    ) + t_gamma * tf.reduce_sum(tf.diag_part(t_weights))
            
            
            
            t_global_step = tf.Variable(0, name='global_step', trainable=False,dtype=tf.int32)
                
            #now optimizer
            t_loss = -target 
            
            
            t_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            t_train_op = t_optimizer.minimize(t_loss, global_step=t_global_step,var_list=[t_topics_free,t_weights_free]) 
            
            t_tfinit = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=2)
            
            return (graph,t_tfinit,
                    t_loss,t_topics,t_train_op,t_weights,chosen_index_1,chosen_index_2,
                    saver
                    )
