# Topic Modelling on Wogma Reviews text.

##Introduction
Wogma is a Movie review site with reviews available over 1000 movies in Hollywood as well as Bollywood. The website is http://wogma.com

This code attempts to find out number of topics which comprises these reviews texts and most important words belonging to that topic. The code is implemented in an unsupervised manner using technique known as PLSA (Probablilistic Latent Semantic Analysis). Read more about it here http://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis

##Step By Step Guide 
1. Scrape the list of all urls which contains reviews. Please see urllist.py. The sample is given is listofurls.txt
2. For each url, extract review text from html pages. Refer data.py
3. Clean the raw review text. Refer clean.py
4. Get Term Frequency and IDF from text
5. Form Bag Of words by multiplying TF with IDF. Refer tdidf.py
6. Train PLSA model with BoW , providing number of desired topics. Refer plsa.py
7. Get List of Topics with most important words for it. 
8. The main program is main.py

##Future work 
1. Unit tests to be added

##Limitations
The code seems to work fine with small data (100 reviews or so). But is extremely slow for full text (1000 reviews). Also, with large data, the EM step does not converge even after 500 iterations.
