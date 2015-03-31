from lxml import html
import requests
import os.path
import codecs
from HTMLParser import HTMLParser
import urllib2

i=1
prefix='review_'
save_path='/home/data/'

def getallurls():
   f=open('listOfUrls.txt', 'wb')
   begin_url = 'http://wogma.com'
   page=requests.get('http://wogma.com/movies/basic/')
   tree=html.fromstring(page.text)
   reviews=tree.xpath('//div[@class="button related_pages review "]/a/@href')
   for review in reviews:
   f.write(begin_url+str(review)+'\n')
   f.close()

def saveReviewData(reviews):
   global i
   fullname = os.path.join(save_path, prefix+str(i)+'.txt')
   i=i+1         
   file_handle = codecs.open(fullname, encoding='utf8', mode='wb')
   for review in reviews:
      file_handle.write(review+'\n')
   file_handle.close()

def getReviewData(url):
   page=requests.get(url)
   tree=html.fromstring(page.text)
   reviews=tree.xpath('//div[@class="review large-first-letter"]/p/text()')
   saveReviewData(reviews)

def getReviewForListOfUrls():
   f=open('listOfUrls.txt', 'r')
   for line in f:
      getReviewData(line)
   f.close()


if __name__=='__main__':
   path = raw_input('Please input path for storing data:')
   save_path = path
   getallurls()
   getReviewForListOfUrls()

