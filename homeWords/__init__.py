import pandas as pd
import spacy
import gensim

import cPickle
import os

current_path = os.path.abspath(os.getcwd())


ref_df = pd.read_csv(current_path + '/homeWords/' + 'Final_Database_for_website.csv')


nlp = spacy.en.English(parser=False)

lda = gensim.models.LdaModel.load(current_path + '/homeWords/' + 'my_lda_model_k5.lda')



with open(current_path + '/homeWords/' + 'my_gbrt.pkl', 'rb') as fid:
    estimator = cPickle.load(fid)





from flask import Flask
app = Flask(__name__)
from homeWords import views

#print 'finished __init__.py'
