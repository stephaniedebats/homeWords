from __future__ import unicode_literals
import pyzillow
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults
from pyzillow.pyzillow import ZillowWrapper, GetUpdatedPropertyDetails
from bs4 import BeautifulSoup
from urllib import urlopen
import re
import pandas as pd
import datetime as dt
import sys
import cPickle	
import numpy as np
from scipy.spatial.distance import cosine
from six.moves.html_parser import HTMLParser

spacy = sys.modules['spacy']
gensim = sys.modules['gensim']
from homeWords import nlp
from homeWords import lda
from homeWords import ref_df
from homeWords import estimator


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

h = HTMLParser()


def define_coverage(row):
	home_size = float(row.home_size)
	property_size = float(row.property_size)
	
	cov1 = home_size/property_size
	if cov1 > 0.45:
		num_floors = 2
		cov1 = home_size/num_floors/property_size
		if cov1 > 0.45:
			num_floors = 3
			cov1 = home_size/num_floors/property_size
	return cov1

def clean(doc, stoplist, d, nlp):
	clean = doc.lower()
	clean = re.sub("(\d+)(th)", '', clean)
	clean = re.sub("(\d+)(rd)", '', clean)
	clean = re.sub("(\d+)(nd)", '', clean)
	clean = re.sub("(\d+)(\s)?(s(\.)?f(\.)?)", r"\1 square feet", clean)
	clean = re.sub("(\+\/\-)(\s)?(s(\.)?f(\.)?)", r"\1 square feet", clean)
	clean = re.sub("(\+)(\s)?(s(\.)?f(\.)?)", r"\1 square feet", clean)
	clean = re.sub("(\s)(\s)?(s(\.)?f(\.)?)", r"\1 san francisco", clean)
	clean = re.sub(r"http?\S+", "", clean)
	clean = re.sub(r"www.\S+", "", clean)
	clean = re.sub(re.compile('<.*?>'),'', clean)
	clean = re.sub('\.(?![a-zA-Z]{2})', '', clean)
	clean = re.sub('\/(?![a-zA-Z]{2})', '', clean)
	clean = [token for token in gensim.utils.simple_preprocess(clean, deacc=True) if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stoplist]
	clean = " ".join(clean)
	pattern = re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')
	clean = pattern.sub(lambda x: d[x.group()], clean)        
	clean = nlp(clean)
	#clean = [token.lemma_ for token in clean if token != 'living' else token]
	clean = [token.lemma_ if token != 'living' else token for token in clean]
	return clean



def call_zillow_api(address, zipcode):
	#YOUR_ZILLOW_API_KEY = 'X1-ZWz1ffghcdgzkb_a80zd' #stephaniedebats@gmail.com
	YOUR_ZILLOW_API_KEY = 'X1-ZWz19ktnpd1lor_67ceq' #forsalebyowner1234@gmail.com
	zillow_data = ZillowWrapper(YOUR_ZILLOW_API_KEY)
	
	try:
		deep_search_response = zillow_data.get_deep_search_results(address, zipcode)
		result = GetDeepSearchResults(deep_search_response)
		print 'User input zillow_id = {}'.format(result.zillow_id)
	except:
		result = -9999
		
	return result



def getEstimate(result, address, zipcode):

	zestimate = "{:,.0f}".format(int(result.zestimate_amount))

	# Use Zillow ID to scrape home description and photos
	webpage = urlopen('http://www.zillow.com/homedetails/' + str(result.zillow_id) + '_zpid/').read()
	soup = BeautifulSoup(webpage, 'lxml')

	#<meta property="og:zillow_fb:address" content="549 Prospect St, San Carlos, CA 94070"/>

	full_address = soup.find("meta",  property="og:zillow_fb:address")
	full_address = full_address['content']

	address = full_address.split(',')[0]
	zipcode = full_address.split(',')[2].split('CA')[1]

	description = soup.find('meta',  {'name':'twitter:description'})
	description = strip_tags(h.unescape(description['content']))

	front_photo = soup.find('meta', {'name':'twitter:image'})
	front_photo = front_photo['content']
	front_photo = str(front_photo)

	interior_photos = soup.findAll(href=re.compile('http://photos.zillowstatic.com/p_c/'))
	interior_photos = [i.get('href') for i in interior_photos]

	# Create pandas dataframe for regression model
	columns = (['bedrooms', 'bathrooms', 'latitude', 'longitude', 'property_size',
		'home_size', 'year_built', 'last_sold_year', 'last_sold_month',
		'date_of_purchase', 'lot_coverage', 'lda_k5_0', 'lda_k5_1', 
		'lda_k5_2', 'lda_k5_3', 'lda_k5_4', 94002.0, 94010.0, 94025.0,
		94030.0, 94062.0, 94063.0, 94065.0, 94070.0, 94401.0, 94402.0,
		94403.0, 94404.0])

	df = pd.DataFrame(0, index=range(1), columns=columns)

	# Fill in df with Zillow stats
	for c in df.columns[0:7]:
		df[c] = result.get_attr(c)

	# Fill in zipcode one-hot variable
	df[float(zipcode)] = 1

	# Fill in df with date info
	mydate = dt.date.today()
	df['last_sold_year'] = mydate.year
	df['last_sold_month'] = mydate.month
	df['date_of_purchase'] = (mydate - dt.date(2013,1,1)).days

	# Fill in df with lot coverage
	df['lot_coverage'] = df.apply(lambda row: define_coverage(row), axis=1)

        
	#LDA
	stoplist = (['house', 'home', 'seller', 'buyer',
		'broker', 'agent', 'properties', 'listing',
		'am', 'a.m.', 'pm', 'p.m.',
		'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
		'mon', 'tue', 'tues', 'weds', 'thur', 'thurs', 'fri', 'sat', 'sun',
		'january', 'february', 'march', 'april', 'may', 'june', 'july',
		'august', 'september', 'october', 'november', 'december',
		'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
		'noon', 'online',
		'belmont', 'burlingame', 'hillsborough', 'menlo', 'park',
		'east', 'palo', 'alto', 'palo', 'alto', 'atherton', 'redwood',
		'city', 'millbrae', 'emerald', 'hills', 'woodside', 'san', 'carlos',
		'redwood', 'shores', 'san', 'mateo', 'foster', 'city'])

	d = {
	    'virtual tour' : '',
	    'walk tour' : '',
	    'video' : '',
	    'open house' : '',  
	    'rm' : 'room',
	    'rms' : 'rooms',
	    'br': 'bedroom',
	    'brs' : 'bedroom',
	    'bed' : 'bedroom',
	    'beds' : 'bedroom',
	    'bath' : 'bathroom',
	    'baths': 'bathroom',
	    'ba' : 'bathroom',
	    'bas' : 'bathroom',
	    'bd' : 'bedroom',
	    'bds' : 'bedroom',
	    'mbr' : 'master bedroom',
	    'lr' : 'living room',
	    'dr' : 'dining room',   ### what about drive?
	    'fr' : 'family room',
	    'bsmt' : 'basement',
	    'ss' : 'stainless steel',
	    'kit' : 'kitchen',
	    'flr' : 'floor',
	    'flrs' : 'floor',
	    'hw' : 'hardwood',
	    'hwf' : 'hardwood floor',
	    'hdwd' : 'hardwood',
	    'fp' : 'fireplace',
	    'fps' : 'fireplace',
	    'fplc' : 'fireplace',
	    'fplcs' : 'fireplace',
	    'lg' : 'large',
	    'lrg' : 'large',
	    'bkyd' : 'backyard',
	    'yr' : 'year',
	    'yrs' : 'years',
	    'sqft' : 'square feet',
	    'sq ft' : 'square feet',
	    'sq-ft' : 'square feet',
	    'mkt' : 'market', 
	}

	tokens = clean(description, stoplist, d, nlp)
	tokens = lda.id2word.doc2bow(tokens)
	topics = lda[tokens]
	print 'lda topics = {}'.format(topics)

	table = np.zeros([1, 5])

	for t in topics:
		table[0, t[0]] = t[1]

	for i in range(table.shape[1]):
	    df['lda_k5_' + str(i)] = pd.Series(table[:, i])
	   
	print table[0]

	####### DATAFRAME IS READY FOR REGRESSION MODEL!!!!!
	
#	with open('/Users/srd/Projects/homeWords_website/homeWords/my_gbrt.pkl', 'rb') as fid:
#		estimator = cPickle.load(fid)

	estimate = estimator.predict(df)
	estimate0 = estimate[0]
	estimate = "{:,.0f}".format(estimate0)
	print "estimate = {}".format(estimate)
	print type(estimate)

	print front_photo
	print type(front_photo)


	############# Find matches in ref_df ##################

	#ref_df = pd.read_csv('/Users/srd/Projects/homeWords_website/homeWords/Final_Database_for_website.csv', parse_dates=['last_sold_date'])
	
	matches = (ref_df[
		(ref_df.last_sold_year.isin([2016, 2015])) &
		(ref_df.zip == float(zipcode)) &
		(
			(ref_df.home_size > float(df.home_size)*0.8) &
			(ref_df.home_size < float(df.home_size)*1.2)
			) &
		((ref_df.last_sold_price > float(estimate0)*0.7) & 
			(ref_df.last_sold_price < float(estimate0)*1.3))
		])

	print matches.dtypes
	print 'matches zid'
	print matches.zillow_id
	print 'result zid'
	print result.zillow_id
	print type(result.zillow_id)
	
	matches = matches[matches.zillow_id != int(result.zillow_id)]
	#matches = matches[matches.address != address]

	def cosine_sim_scores(row, topics):
		check_topics = np.array([row['lda_k5_0'], row['lda_k5_1'], row['lda_k5_2'], row['lda_k5_3'], row['lda_k5_4']])
		return cosine(topics, check_topics)

	matches['cosine_distance'] = matches.apply(lambda row: cosine_sim_scores(row, table[0]),axis=1)
	matches = matches.sort_values(by='cosine_distance', ascending=True)
	zids = matches.head(2)['zillow_id'].tolist()
	matches_addresses = matches.head(2)['address'].tolist()
	matches_dates = matches.head(2)['last_sold_date'].tolist()
	matches_prices = matches.head(2)['last_sold_price'].tolist()
	matches_prices = ["{:,.0f}".format(price) for price in matches_prices]

	comps = {}
	counter = 0
	for zid in zids:
		print 'zid = {}'.format(zid)
		comps[counter] = {}
		comps[counter]['address'] = matches_addresses[counter]
		comps[counter]['date'] = matches_dates[counter]
		comps[counter]['price'] = matches_prices[counter]
		# Use Zillow ID to scrape home description and photos
		webpage = urlopen('http://www.zillow.com/homedetails/' + str(zid) + '_zpid/').read()
		soup = BeautifulSoup(webpage, 'lxml')

		#description = soup.find('meta',  property='og:description')
		#print(description['content'] if description else 'No meta description given')

                                
		description0 = soup.find('meta',  {'name':'twitter:description'})
		comps[counter]['description'] = strip_tags(h.unescape(description0['content']))

		front_photo0 = soup.find('meta', {'name':'twitter:image'})
		front_photo0 = front_photo0['content']
		comps[counter]['front_photo'] = str(front_photo0)

		interior_photos0 = soup.findAll(href=re.compile('http://photos.zillowstatic.com/p_c/'))
		comps[counter]['interior_photos'] = [i.get('href') for i in interior_photos0]
		counter += 1

	print 'ending here'
	print comps







	#################

	return full_address, estimate, zestimate, description, front_photo, interior_photos, comps
