#Code to form Term-Document matrix from all text
import numpy as np
import scipy.sparse as sp
from math import log

def tokenize(text):
    return text.split()

def tc(dataset, tokenizer=tokenize):
    vocab = {}
    docs = []

    for doc in dataset:
        d = {} # token => count

        for term in tokenizer(doc):
				    vocab[term] = 1
				    d[term] = d.get(term, 0) + 1

        docs.append(d)

    sorted_terms = sorted(vocab.keys())
    vocab = dict([(t, i) for i, t in enumerate(sorted_terms)])

    return docs, vocab



def to_sparse_matrix(tfidf_dict, vocab):
    tfm = sp.lil_matrix((len(vocab), len(tfidf_dict)), dtype=np.double)

    for j, doc in enumerate(tfidf_dict):
        for term in doc:
            try:
                i = vocab[term]
                tfm[i,j] = doc[term]
            except KeyError:
                pass

    return tfm


def to_vector(idf_dict, vocab):
    ret = np.zeros(len(idf_dict))
    for term, idx in vocab.items():
        ret[idx] = idf_dict[term]
    return ret

def idc_from_tc(term_counts):
    t = {}
    for doc in term_counts:
        for term in doc:
            t[term] = t.get(term, 0) + 1 

    return t

def idf_from_tc(term_counts):
    n_docs = len(term_counts)
    idf = {}
    idc = idc_from_tc(term_counts)
    for term in idc:
        idf[term] = log(n_docs*1.0/(idc[term]))

    return idf

def inverse_vocab(vocab):
    """
    Converts a vocab dictionary term => index to index => term
    """
    return dict((i,t) for t,i in vocab.items())


def tf_mul_idf(tf, idf):
		"""
		Multiplies TF matrix with IDF to give weighted frequency
		"""
		for i in range(len(idf)):
				for j in range(len(tf[0])):
						tf[i,j] = tf[i,j] * idf[i]

		return tf
		
if __name__=='__main__':
		sample_input = open('clean_review.txt','r')
		datasets = []
		for line in sample_input:
				datasets.append(line)
		td_dict, vocab = tc(datasets)
		td = to_sparse_matrix(td_dict, vocab).toarray()
		idf = to_vector(idf_from_tc(td_dict), vocab)
		print "term-document matrix size", td.shape
		print vocab
		print vocab['film']
		ind = vocab['film']
		print td[ind,:]
		print idf[ind]
		print tf_mul_idf(td, idf)[ind,:]
