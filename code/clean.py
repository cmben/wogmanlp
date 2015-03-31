from bs4 import BeautifulSoup
import re
import nltk 
import os
#nltk.download()
from nltk.corpus import stopwords

def textToWords(raw_review):
		#Function to clean the raw review and output the formatted words

		#Clean HTML Markups
		review_text = BeautifulSoup(raw_review).get_text()
		
		#Remove punctuation marks
		letters_only = re.sub('[^a-zA-Z]+', ' ', review_text)
		
		#convert to lower case, and split into words
		words = letters_only.lower().split()
		
		#remove stopwords
		stop_words = set(stopwords.words('english'))
		meaningful_words = [w for w in words if not w in stop_words]
		
		#remove words with less than 3 characters
		long_words = [w for w in meaningful_words if len(w) >= 3]

		#return the words
		return(' '.join(long_words))


def getAllTextInput(dir_name):
		clean_train_reviews = []
		i=1
		for filename in os.listdir(dir_name):
				current=os.path.join(dir_name, filename)
				if(os.path.isfile(current)):
						inFile=open(current, 'r')
						raw_review = inFile.read()
						clean_train_reviews.append(textToWords(raw_review))
						print 'File '+str(i)+' proccessed\n'
						i=i+1
		return clean_train_reviews

if __name__=='__main__':
		#inFile = open('/home/chaitanya/instamojoEx/data/review_2.txt', 'r')
		#raw_review = inFile.read()
		#words = textToWords(raw_review)
		#print words
		path = '/home/chaitanya/instamojoEx/data/'
		getAllTextInput(path)
