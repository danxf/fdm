import numpy as np
from itertools import product
from collections import Counter

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from datetime import datetime

import gc

class FDM:

    def __init__(self,T,data,learning_rate=0.01,regularizer_coef=0.0,reg2 = 0.0, init_std_dev = .05,    
                 verbose_period = 1000,
                 log_period = 100,
                 optimizer_type = 'adam'
    ):
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
        
        optimizer_type : 'adam' or 'rmsprop'
        """
        self.T = T
        self.learning_rate = learning_rate
        self.regularizer_coef = regularizer_coef
        self.reg2 = reg2
        self.data_to_fit = data
        self.voc_size = self.data_to_fit.shape[0]
        #self.data_reshaped = self.data_to_fit.reshape((self.voc_size * self.voc_size)) #flattened data matrix used for SGD sampling

        
        self.verbose_period = verbose_period        
        self.optimizer_type = optimizer_type


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
        self.init_std_dev = init_std_dev
        
        self.log_period = log_period
        
        self.topics = None
        self.weights = None    
        
        self.callback = None
    
    def get_topics(self):
        #return self.topics 
        return self._tf_session.run(self.t_topics)
        

    def get_weights(self):
        #return self.weights
        return self._tf_session.run(self.t_weights)        
    

    def get_topics_free(self):
        #return self.topics 
        return self._tf_session.run(self.t_topics_free)
        

    def get_weights_free(self):
        #return self.weights
        return self._tf_session.run(self.t_weights_free)        

    def get_state(self):
        
        opt_var_vals = self._tf_session.run(self.t_opt_vars)
        
        return self.get_weights_free(),self.get_topics_free(), opt_var_vals
    
    def set_state(self,state):
        weights_f, topics_f, opt_var_vals = state
        
        self._tf_session.run( self.t_weights_free_assign_op, feed_dict = {self.t_weights_free_pl:weights_f} ) 
        self._tf_session.run( self.t_topics_free_assign_op, feed_dict = {self.t_topics_free_pl:topics_f} ) 

        self._tf_session.run(self.t_opt_vars_assigns, feed_dict = dict(zip(self.t_opt_vars_pls,opt_var_vals)))

    
    def init_topics(self, topics):
        
        lt = np.log(topics) 
        lt_m = lt.max(axis = 1)[:,np.newaxis]
        lt -= lt_m
        
        self._tf_session.run( self.t_topics_free_assign_op, feed_dict = {self.t_topics_free_pl:lt} ) 
        
    def init_weights(self, weights):
        
        lw = np.log(weights) 
        assert np.where(np.abs(lw - lw.T)>1e-10)[0].shape[0] == 0 , 'Weights must be symmetric'
        
                
        lw_m = lw.max()
        lw -= lw_m
        
        lw /= 2
        
        self._tf_session.run( self.t_weights_free_assign_op, feed_dict = {self.t_weights_free_pl:lw} ) 
    


    def fit(self, num_iterations=100000, batch_size = 512, batches_to_sample = 8000, init_topics = None, init_weights = None, uniform_is_coef = None, optimizer_param = {}):
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
        self.optimizer_param = optimizer_param
        
        
        #build graph
        (self.graph,self.t_tfinit,self.t_loss,self.t_topics,self.t_train_op,
                self.t_weights,
                self.chosen_index_1,self.chosen_index_2,self.is_corrections_pl,
                self.tf_saver, 
                self.t_topics_free_pl,
                self.t_weights_free_pl,
                self.t_weights_free_assign_op,
                self.t_topics_free_assign_op,
                self.t_pre_target,
                self.t_learning_rate_pl,
                self.t_weights_free,
                self.t_topics_free,
                self.t_opt_vars,
                self.t_opt_vars_pls,
                self.t_opt_vars_assigns                
        ) = FDM._build_graph(self.T,self.voc_size,batch_size,self.regularizer_coef,self.reg2,
                             optimizer_type = self.optimizer_type,
                             optimizer_param = self.optimizer_param,
                             init_std_dev = self.init_std_dev
        )
        
        #start session
        self._tf_session = tf.Session(graph = self.graph) 
        self._tf_session.run(self.t_tfinit)
        
        
        if init_topics is not None:
            self.init_topics(init_topics)
        if init_weights is not None:
            self.init_weights(init_weights)

        print('Building sampler...')                        
        self.sampler = FDM.Sampler(self.data_to_fit, batch_size, 
                                batches_to_sample, voc_size = self.voc_size,
                                uniform_is_coef = uniform_is_coef
                                )
        
        #train model
        #print('Starting iterations...')                
        self._do_iterations(num_iterations)
        
        
        
    def continue_model_training(self,num_iterations=100000):
        #check if model was initialized
        if self._tf_session is None:
            print ('Cannot continue uninitialized model.\nUse FDM.fit() first, or load an initialized FDM model.')
            return
        
        #continue training
        self._do_iterations(num_iterations)


    """
    This class is used to pool random samples. 
    It is more efficient to call np.random.choice with a large size parameter, versus calling np.random.choice 
    size times with size=1.  
    This class implements such pooling. 
    
    The np.random.Generator.choice  method allocates a new array of size sampling_dist , which doubles memory consumption during sampling and takes time. Can be avoided in principle. 
    """
    class Sampler:
        def __init__(self, sampling_dist, batch_size, batches_to_sample, voc_size=None, do_reshape = True,
                     uniform_is_coef = None,
                     is_threshold = 1e-20
                     ):
            self.batch_size = batch_size
            self.batches_to_sample = batches_to_sample            
            
            self.voc_size = voc_size
            if do_reshape:
                self.sampling_dist = sampling_dist.reshape((voc_size * voc_size)) #flattened data matrix                
            else:
                self.sampling_dist = sampling_dist
                        
            self.do_is = not (uniform_is_coef is None)
            
            if self.do_is:                

                self.sampling_dist_ground = self.sampling_dist
                self.sampling_dist = self.sampling_dist.copy()
                self.sampling_dist *= (1. - uniform_is_coef)
                
                uidx = self.sampling_dist_ground > is_threshold
                self.sampling_dist[uidx] += uniform_is_coef/ np.sum(uidx)
                
                print('Sampler: Unifom added, on {:.2f}%.'.format( (np.sum(uidx)/uidx.shape[0])*100 ))
            
                
            
            self.rng = np.random.default_rng()
            
            self.age = 0
            #print('Resetting...')
            self.reset()
            
        def reset(self):
            
            print('{}:Sampling... '.format(datetime.today()))
            
            
            self.samples = self.rng.choice(self.sampling_dist.shape[0], p = self.sampling_dist, size = (self.batches_to_sample,self.batch_size))
            
            
            print('{}:Done sampling.'.format(datetime.today()))
            
            self.fetch_sample_idx = 0
            
            self.age += 1
            
        
        def get_sample(self):
            
            if self.fetch_sample_idx >= self.batches_to_sample:
                self.reset()
            
            self.fetch_sample_idx += 1 
            return self.samples[self.fetch_sample_idx - 1,:]
            

        def get_sampled_pairs(self):
            
            batch = self.get_sample()
                                                            
            pair_0 = batch // self.voc_size   
            pair_1 = batch % self.voc_size
            
            
            if not self.do_is:            
                return pair_0, pair_1
            
            is_corrections = self.sampling_dist_ground[batch] / self.sampling_dist[batch]
            return pair_0, pair_1, is_corrections




                
        
    def _do_iterations(self,num_iterations=100000):
        local_iteration_counter = 0        
        
        while local_iteration_counter < num_iterations:

            if self.callback is not None:
                self.callback.cb(self, self.global_iteration_counter)
                                                
            if not self.sampler.do_is:
                pair_0, pair_1 = self.sampler.get_sampled_pairs()

                feed_dict = {self.chosen_index_1:pair_0,
                             self.chosen_index_2:pair_1,
                             self.t_learning_rate_pl: self.learning_rate
                        }
            else:
                pair_0, pair_1, is_corrections = self.sampler.get_sampled_pairs()

                feed_dict = {self.chosen_index_1:pair_0,
                             self.chosen_index_2:pair_1,
                             self.t_learning_rate_pl: self.learning_rate,
                             self.is_corrections_pl: is_corrections
                        }
                
            
            
            #do SGD steps
            res = self._tf_session.run([self.t_train_op,self.t_loss], feed_dict = feed_dict)
            
            self.global_iteration_counter += 1    
            local_iteration_counter += 1
            
            #update topics and weights data members
            if local_iteration_counter%self.log_period == 0:
                
                if not self.sampler.do_is:
                    empirical_ent = np.log(self.data_to_fit[pair_0,pair_1]).mean()
                else:
                    empirical_ent = (is_corrections*np.log(self.data_to_fit[pair_0,pair_1])).mean()
                
                empirical_kl = empirical_ent + res[1]
                
                self._loss_arr.append((res[1],empirical_kl))
                
                self.topics = self._tf_session.run(self.t_topics)
                self.weights = self._tf_session.run(self.t_weights)
                
                if local_iteration_counter%self.verbose_period == 0:
                    print ('local iteration no. {} / {}'.format(local_iteration_counter,num_iterations))
                    print ('kl: {}, loss: {}'.format(empirical_kl,res[1]))

        
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
        topic_vec = sorted( list(zip(range(Ndict),topic_vec)), key = lambda k: -k[1] )

        topic_vec = [ (dictionary[i],w) for i,w in topic_vec if w>topic_threshold]

        for a in topic_vec[:nwords]:
            print ('{:14}{:.4f}'.format(a[0],a[1]) )
        print ('')
            
    def print_all_topics(self,dictionary = None, nwords = 20, topic_threshold=1e-5):
        for t in range(self.T):
            print ('Topic number {}:'.format(t))
            self.print_topic(t, dictionary, nwords, topic_threshold)
    

        
    MOMENT_COMPUTE_DIRECT = 0
    MOMENT_COMPUTE_SPARSE = 1
    MOMENT_COMPUTE_SPARSE_C = 2
    MOMENT_COMPUTE_SPARSE_C_COUNTER = 3
    MOMENT_COMPUTE_SPARSE_C_FROM_COUNTER = 4
    
    @staticmethod
    def build_data_matrix(corpus,voc_size, verbose_cnt = None, method = MOMENT_COMPUTE_SPARSE):
        """
        Parameters
        ----------
        corpus : list of lists of int
            A list represeting the corpus where each list corpus[i] is a list of ints representing tokens in the range {0,...,voc_size-1}
        voc_size : int 
            The number of unique tokens in the corpus

        method : The method by which to compute the moment. 
            FDM.MOMENT_COMPUTE_SPARSE is usually sufficient.
            FDM.MOMENT_COMPUTE_SPARSE_C is a faster cython implementation. Requires compilation. See the README file.
            
        Returns
        -------
        second_moment_matrix : Numpy array of shape (voc_size,voc_size)
            Empricial FDM matrix. The (i,j) entry represents the probability of observing the tokens i and j in the same document, where the documents are sampled uniformly at random from the corpus.
        """
        
        FDM_M_matrix = np.zeros((voc_size,voc_size), dtype = np.float)
        
                
        if method == FDM.MOMENT_COMPUTE_SPARSE_C:
            
            from _fast_moment import _fast_add_moment    

            for i,d in enumerate(corpus):
                if len(d) == 1:
                    continue

                d_arr = np.array(d, dtype = np.int)
                _fast_add_moment(FDM_M_matrix,d_arr)
                
                if (verbose_cnt is not None) and ((i % verbose_cnt) == 0):
                    print(i)
            
        elif method == FDM.MOMENT_COMPUTE_SPARSE:

            for i,d in enumerate(corpus):
                d_emp_dist = np.zeros(voc_size, dtype = np.float)
                d_len=np.float(len(d))
                
                if d_len <= 1:
                    continue
                
                non_zero_idxs = set()
                for w in d:
                    d_emp_dist[w]+=1
                    non_zero_idxs.add(w)
                d_emp_dist /= d_len
                
                #creating FDM empirical matrix for each document d
                for (w,v) in product(non_zero_idxs,repeat=2):
                    if v!=w:
                        FDM_M_matrix[w,v] += d_emp_dist[w]*d_emp_dist[v]* (d_len/(d_len-1))
                    else:
                        FDM_M_matrix[w,w] += (d_len/(d_len-1))*(d_emp_dist[w]*d_emp_dist[w] - (1/d_len)*d_emp_dist[w])

                if (verbose_cnt is not None) and ((i % verbose_cnt) == 0):
                    print(i)            
        
        elif method == FDM.MOMENT_COMPUTE_DIRECT:
            
            #the "direct" computation which multiplies matrices, typically much slower

            dummy  = np.zeros((voc_size,voc_size),dtype = np.float)
            rng_voc = np.arange(voc_size)                                
            
            for i,d in enumerate(corpus):
                d_emp_dist = np.zeros(voc_size)
                d_len=np.float(len(d))
                
                if d_len <= 1:
                    continue
                
                non_zero_idxs = set()
                for w in d:
                    d_emp_dist[w]+=1
                    non_zero_idxs.add(w)
                d_emp_dist /= d_len
                
                
                np.dot(((d_len)/(d_len-1))*d_emp_dist[:,np.newaxis],d_emp_dist[np.newaxis,:], out = dummy) 
                d_emp_dist /= (d_len-1)
                dummy[rng_voc,rng_voc] -= d_emp_dist
                            
                FDM_M_matrix += dummy
                
                dummy[:,:] = 0
                
                if (verbose_cnt is not None) and ((i % verbose_cnt) == 0):
                    print(i)            

        elif method == FDM.MOMENT_COMPUTE_SPARSE_C_COUNTER:
            #this is C iteration, but which computes the empirical measure first, and then 
            #iterates only over non-zero entries, similarly to FDM.MOMENT_COMPUTE_SPARSE. 
            raise Exception('Moment computation method not implemented.')

        elif method == FDM.MOMENT_COMPUTE_SPARSE_C_FROM_COUNTER:
            #This assumes a different fromat of the corpus.
            #each document is an array of tokens, and array of counts 
            from _fast_moment import _fast_add_moment_from_counter    

            for i,(toks,cnts) in enumerate(corpus):
                if (len(toks) == 0) or ( (len(toks) == 1) and (cnts[0]<=1) ):
                    continue

                toks = toks.astype(np.int)
                cnts = cnts.astype(np.int)                
                _fast_add_moment_from_counter(FDM_M_matrix,toks,cnts)
                
                if (verbose_cnt is not None) and ((i % verbose_cnt) == 0):
                    print(i)
            
            
        else:
            raise Exception('Unknown moment compute method.')
                                        
        
        #could happen due to numeric issues
        FDM_M_matrix[FDM_M_matrix<0] = 0
        
        #The empirical FDM matrix of the corpus is the average of the empirical FDM of each of the documents                
        FDM_M_matrix /= FDM_M_matrix.sum()   
        
        return FDM_M_matrix         
        
    
    

    
    @staticmethod
    def _build_graph(Npartitions,
                    voc_size,
                    batch_size,
                    gamma_regularizer,
                    reg2,
                    optimizer_param,
                    optimizer_type,
                    init_std_dev = .05,
                    ):
        
        
        
        graph=tf.Graph()
        with graph.as_default():
            
            chosen_index_1 = tf.placeholder(dtype=tf.int32,shape=(batch_size))
            chosen_index_2 = tf.placeholder(dtype=tf.int32,shape=(batch_size))
            is_corrections_pl = tf.placeholder_with_default(tf.ones(batch_size,dtype=tf.float32),                                                         
                                                            shape=(batch_size))
            
            learning_rate_pl = tf.placeholder(dtype=tf.float32)

            
            
            
            t_weights_free = tf.Variable(tf.truncated_normal([Npartitions,Npartitions],mean=0.,stddev=init_std_dev),dtype=tf.float32)
            t_weights_free_sym = t_weights_free + tf.transpose(t_weights_free)
            t_weights = tf.reshape( tf.nn.softmax(tf.reshape(t_weights_free_sym,[Npartitions*Npartitions])) , [Npartitions,Npartitions])
            
            t_topics_free = tf.Variable(tf.truncated_normal([Npartitions,voc_size],mean=0.,stddev=init_std_dev),dtype=tf.float32)
            t_topics = tf.nn.softmax(t_topics_free) #default axis is (-1)


            t_topics_free_pl = tf.placeholder(tf.float32,shape = [Npartitions,voc_size])
            t_weights_free_pl = tf.placeholder(tf.float32,shape = [Npartitions,Npartitions])

            t_weights_free_assign_op = tf.assign(t_weights_free,t_weights_free_pl)
            t_topics_free_assign_op = tf.assign(t_topics_free,t_topics_free_pl)



            
            
            t_gamma = gamma_regularizer
            t_gamma2 = reg2
            
            pre_target =  tf.log((
                    tf.reduce_sum ((
                        tf.matmul(
                            tf.expand_dims(tf.transpose(tf.gather(t_topics,chosen_index_1,axis=1)),-1),
                            tf.expand_dims(tf.transpose(tf.gather(t_topics,chosen_index_2,axis=1)),1)
                            ) 
                    * t_weights),axis=[1,2])
                    )) 
            target = tf.reduce_mean( 
                        is_corrections_pl*tf.where(tf.is_nan(pre_target), tf.zeros_like(pre_target),pre_target )
                    ) + t_gamma * tf.reduce_sum(tf.diag_part(t_weights)) + t_gamma2 * tf.reduce_sum( tf.diag_part(t_weights) / tf.reduce_sum(t_weights,axis=1))
            
            
                        
                
            #now optimizer
            t_loss = -target 
                        
            
            #t_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            #t_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = .9)
            #t_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            

            if optimizer_type == 'adam':
                t_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl,
                                                 **optimizer_param
                                                 )
            elif optimizer_type == 'rmsprop':
                t_optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate_pl, 
                                                        **optimizer_param
                )
            else:
                raise ValueError('Unknown optimizer')
            

            opt_vars = t_optimizer.variables()
            opt_vars_pls = [tf.placeholder(dtype=v.dtype,shape=v.shape) for v in opt_vars]            
            opt_vars_assigns = [tf.assign(v,pl) for v,pl in zip(opt_vars,opt_vars_pls)]
            
            
            
            t_train_op = t_optimizer.minimize(t_loss) 
            
            t_tfinit = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=2)
            
            t_loss_to_display = -(target - (t_gamma * tf.reduce_sum(tf.diag_part(t_weights)) + t_gamma2 * tf.reduce_sum( tf.diag_part(t_weights) / tf.reduce_sum(t_weights,axis=1))))
            
            return (graph,t_tfinit,
                    t_loss_to_display,t_topics,t_train_op,t_weights,
                    chosen_index_1,chosen_index_2, is_corrections_pl,
                    saver,
                    t_topics_free_pl,
                    t_weights_free_pl,
                    t_weights_free_assign_op,
                    t_topics_free_assign_op,
                    pre_target,
                    learning_rate_pl,
                    t_weights_free,
                    t_topics_free,
                    opt_vars,
                    opt_vars_pls,
                    opt_vars_assigns                    
                    )



    
    @staticmethod
    def topic_data_init(corpus, Ntopics, Ndict, beta_smoother= 1e-2, star_alpha = .9):
        """
        Initialize topics as empirical distributions of random subsamples of the corpus. 
        This is similar to the initizlization in LDA Gibbs samplers. 
        """
        
        clen = len(corpus)
        res = np.zeros((Ntopics,Ndict))
        init_size = clen // Ntopics
        
        permutation = np.random.permutation(clen)
        
        cpos = 0
        for i in range(Ntopics):
            
            cnt = Counter()
            for didx in permutation[cpos:cpos+init_size]:
                cnt.update(corpus[didx])
            
            for k,v in cnt.items():
                res[i,k] = v
            
            cpos += init_size
        
        
        res += beta_smoother 
        res /= res.sum(axis = 1)[:,np.newaxis]
        
        mean = res.mean(axis = 0)[np.newaxis,:]
        res = star_alpha*res + (1.-star_alpha)* mean
        
        
        return res


    @staticmethod
    def topic_data_init_csr(csr, Ntopics, Ndict, beta_smoother= 1e-2, star_alpha = .9):
        """
        The input is the corpus as a csr matrix 
        Initialize topics as empirical distributions of random subsamples of the corpus. 
        This is similar to the initizlization in LDA Gibbs samplers. 
        """
        
        clen = csr.shape[0]
        res = np.zeros((Ntopics,Ndict))
        init_size = clen // Ntopics
        
        permutation = np.random.permutation(clen)
        
        cpos = 0
        for i in range(Ntopics):
            
            
            res[i,:] = csr[permutation[cpos:cpos+init_size]].sum(axis = 0).A1
            
            cpos += init_size
        
        
        res += beta_smoother 
        res /= res.sum(axis = 1)[:,np.newaxis]
        
        mean = res.mean(axis = 0)[np.newaxis,:]
        res = star_alpha*res + (1.-star_alpha)* mean
        
        
        return res




    @staticmethod
    def topic_data_init_iter(corpus_iter, Ntopics, Ndict, beta_smoother= 1e-2, star_alpha = .9):
        """
        Initialize topics as empirical distributions of random subsamples of the corpus. 
        This is similar to the initizlization in LDA Gibbs samplers. 
        """
                
        res = np.zeros((Ntopics,Ndict))
        
        i_topic = 0
        perm = np.random.permutation(Ntopics)
        for doc in corpus_iter:
            
            c_topic = perm[i_topic]
            cnt = Counter(doc)            
            for k,v in cnt.items():
                res[c_topic,k] += v
            
            i_topic += 1
            if i_topic == Ntopics:
                i_topic = 0
                perm = np.random.permutation(Ntopics)

        
        res += beta_smoother 
        res /= res.sum(axis = 1)[:,np.newaxis]
        
        mean = res.mean(axis = 0)[np.newaxis,:]
        res = star_alpha*res + (1.-star_alpha)* mean
        
        
        return res




