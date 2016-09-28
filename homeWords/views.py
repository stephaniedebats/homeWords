from my_model import getEstimate
from my_model import call_zillow_api
from flask import request
from flask import render_template
from homeWords import app
import pandas as pd


@app.route('/', methods=['GET', 'POST'])
def homepage():
	zipcodes = ['94002', '94010', '94025', '94063', '94030', '94062',
	'94065', '94070', '94401', '94402', '94404', '94403']
	return render_template("homepage.html", 
		zipcodes = zipcodes)

@app.route('/output')
def output():
  address = request.args.get('address')
  zipcode = request.args.get('zipcode')

  result = call_zillow_api(address, zipcode)

  if result == -9999:
    return render_template("error.html")

  else:
    estimate, zestimate, description, front_photo, interior_photos, comps = getEstimate(result, address, zipcode)
    
    return render_template("output.html",
      address = address, zipcode = zipcode, the_estimate = estimate, zestimate = zestimate,
      description = description, front_photo = front_photo, interior_photos=interior_photos,
      comps = comps
      )

