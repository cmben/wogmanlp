#Main accessor program to interact with user. 
#This code uses plsa and other utiliy functions and outputs
# N topics with most important words belonging to it.

import clean 
import tfidf
from plsa import pLSA
import plsa

if __name__=='__main__':
		print 'This is PLSA program'
		path = raw_input('Please enter location of dir where data is stored :')
		
		#get clean reviews in a list
		clean_review_text = clean.getAllTextInput(path)
		assert(len(clean_review_text) > 0)
		print 'Cleaning and Processing Complete....'
	
		#get tf, idf matrix
		td_dict, vocab = tfidf.tc(clean_review_text[:40])
		td = tfidf.to_sparse_matrix(td_dict, vocab).toarray()
		idf = tfidf.to_vector(tfidf.idf_from_tc(td_dict), vocab)
		tf_by_idf = tfidf.tf_mul_idf(td, idf)
		print "term-document matrix size", tf_by_idf.shape
		print 'Feature Extraction complete....'

		#Train the PLSA model with 500 max iterations
		num_topics = input('Please enter number of desired topics:')
		myplsa = pLSA()
		model = myplsa.train(tf_by_idf, num_topics, 500)
		print 'Training data phase complete....'

		#get the topic labels
		topic_words = input('Please enter number of words for each topic:')
		inv_vocab = tfidf.inverse_vocab(vocab)
		label_plsa = pLSA(model)
		print label_plsa.topic_labels(inv_vocab, topic_words)
