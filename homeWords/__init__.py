import spacy
import gensim
import pandas as pd
ref_df = pd.read_csv('/Users/srd/Projects/homeWords_website/homeWords/Final_Database_for_website.csv')
from flask import Flask
app = Flask(__name__)
from homeWords import views
