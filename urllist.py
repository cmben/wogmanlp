from HTMLParser import HTMLParser
import urllib2
from lxml import html
import requests


def getallurls():

   f=open('listOfUrls.txt', 'wb')
   
   begin_url = 'http://wogma.com'
   
   page=requests.get('http://wogma.com/movies/basic/')
   tree=html.fromstring(page.text)
   reviews=tree.xpath('//div[@class="button related_pages review "]/a/@href')
   for review in reviews:
   f.write(begin_url+str(review)+'\n')

   f.close()


