#Implementation of pLSA as described in Hoffman paper
#Notation as follows : 
# w:word, d:document, z:topic
# V: number of words, D: number of docs, Z: number of topics


import numpy as np

def normalize_1d(a, out=None):
    if out is None: out = np.empty_like(a)
    s = np.sum(a)
    if s != 0.0 and len(a) != 1:
        np.divide(a, s, out)
    return out

def normalize(M, axis=0, out=None):
    if len(M.shape) == 1: return normalize_1d(M, out)
    if out is None: out = np.empty_like(M)
    if axis == 0:
        M = M.T #M.swapaxes(0,1)
        out = out.T

    for i in range(len(M)):
        normalize_1d(M[i], out[i])

    if axis == 0: out = out.T

    return out


def loglikelihood(td, p_z, p_w_z, p_d_z):
    """
    Compute the log-likelihood that the model generated the data.
    """
    V, D = td.shape
    L = 0.0
    for w,d in zip(*td.nonzero()):
        p_d_w = np.sum(p_z * p_w_z[w,:] * p_d_z[d,:])
        if p_d_w > 0: L += td[w,d] * np.log(p_d_w)
    return L



def train(td,
          p_z, p_w_z, p_d_z,
          p_z_old, p_w_z_old, p_d_z_old,
          maxiter, eps):

    R = td.sum() # total number of word counts

    lik = loglikelihood(td, p_z, p_w_z, p_d_z)

    for iteration in range(1, maxiter+1):
        # Swap old and new
        p_d_z_old, p_d_z = (p_d_z, p_d_z_old)
        p_w_z_old, p_w_z = (p_w_z, p_w_z_old)
        p_z_old, p_z = (p_z, p_z_old)

        # Set to 0.0 
        p_d_z *= 0.0
        p_w_z *= 0.0
        p_z *= 0.0

        for w,d in zip(*td.nonzero()):
            # E-step
            p_z_d_w = p_z_old * p_d_z_old[d,:] * p_w_z_old[w,:]
            normalize(p_z_d_w, out=p_z_d_w)

            # M-step
            s = td[w,d] *  p_z_d_w
            p_d_z[d,:] += s
            p_w_z[w,:] += s
            p_z += s

        # normalize
        normalize(p_d_z, axis=0, out=p_d_z)
        normalize(p_w_z, axis=0, out=p_w_z)
        p_z /= R

        lik_new = loglikelihood(td, p_z, p_w_z, p_d_z)
        lik_diff = lik_new - lik
        assert(lik_diff >= -1e-10)
        lik = lik_new

        if lik_diff < eps:
            print "No more progress, stopping EM at iteration", iteration
            break

        print "Iteration", iteration

        print "Parameter change"
        print "P(z): ", np.abs(p_z - p_z_old).sum()
        print "P(w|z): ", np.abs(p_w_z - p_w_z_old).sum()
        print "P(d|z): ", np.abs(p_d_z - p_d_z_old).sum()
        print "L += %f" % lik_diff




class pLSA(object):

    def __init__(self, model=None):
        """
        model: a plsa model returned by train
        """
        self.p_z = None
        self.p_w_z = None
        self.p_d_z = None
        if model is not None: self.set_model(model)
      
		
    def random_init(self, Z, V, D):
        """
        Z: the number of topics desired.
        V: vocabulary size.
        D: number of documents.
        """
        # np.random.seed(0) # uncomment for deterministic init
        if self.p_z is None:
            self.p_z = normalize(np.random.random(Z))
        if self.p_w_z is None:
            self.p_w_z = normalize(np.random.random((V,Z)), axis=0)
        if self.p_d_z is None:
            self.p_d_z = normalize(np.random.random((D,Z)), axis=0)


    def train(self, td, Z, maxiter=500, eps=0.01):
        """
        Train the model.

        td: a V x D term-document matrix of term-counts.
        Z: number of topics desired.
        """
        V, D = td.shape

        self.random_init(Z, V, D)

        p_d_z_old = np.zeros_like(self.p_d_z)
        p_w_z_old = np.zeros_like(self.p_w_z)
        p_z_old = np.zeros_like(self.p_z)


        train(td.astype(np.uint32),
                   self.p_z, self.p_w_z, self.p_d_z,
                   p_z_old, p_w_z_old, p_d_z_old,
                   maxiter, eps)

        return self.get_model()

    def topic_labels(self, inv_vocab, N=10):
        """
        For each topic z, find the N words with highest probability.

        inv_vocab: a term-index => term-string dictionary

        Return: Z lists of N words.
        """
        Z = len(self.p_z)
        ret = []
        for z in range(Z):
            ind = np.argsort(self.p_w_z[:,z])[-N:][::-1]
            ret.append([inv_vocab[i] for i in ind])
        return ret

    def get_model(self):
        return (self.p_z, self.p_w_z, self.p_d_z)

    def set_model(self, model):
        self.p_z, self.p_w_z, self.p_d_z = model

    
		
