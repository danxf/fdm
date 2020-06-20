#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True


import numpy
cimport numpy

ctypedef numpy.int_t DINT_t
ctypedef numpy.float_t DFLOAT_t



def _fast_add_moment(numpy.ndarray[DFLOAT_t, ndim=2] M, 
                     numpy.ndarray[DINT_t, ndim=1] doc):
    
    cdef int i,j, tok_i    
    cdef int d_len = doc.shape[0]
    
    cdef DFLOAT_t v = 1 
    v /=  (d_len*(d_len-1))
            
    for i in range(d_len):
        tok_i = doc[i]
        for j in range(d_len):
            M[tok_i,doc[j]] += v
    
    
    #and complete the diagonal elems too
    for i in range(d_len):
        tok_i = doc[i]
        M[tok_i,tok_i] -= v
    
    
def _fast_add_moment_from_counter(numpy.ndarray[DFLOAT_t, ndim=2] M, 
                     numpy.ndarray[DINT_t, ndim=1] toks,
                     numpy.ndarray[DINT_t, ndim=1] cnts
                    ):
    
    cdef int i,j, tok_i, cnt_i
    cdef int n_tok = toks.shape[0]
    cdef int d_len = 0 
    
    
    for i in range(n_tok):
        d_len += cnts[i]
    
    
    cdef DFLOAT_t v = 1 
    v /=  (d_len*(d_len-1))

            
    for i in range(n_tok):
        tok_i = toks[i]
        cnt_i = cnts[i]
        
        for j in range(n_tok):            
            M[tok_i,toks[j]] += v*( (cnt_i*cnts[j]) ) 
            
    
    
    #and complete the diagonal elems too
    for i in range(n_tok):
        tok_i = toks[i]
        M[tok_i,tok_i] -= v*cnts[i]
    
    





