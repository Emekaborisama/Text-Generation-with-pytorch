from flask import Flask,jsonify,request,render_template, make_response
from flask_cors import CORS, cross_origin
from app.predict_model import text_generator
import os 
import sys



app = Flask(__name__)
cors = CORS(app)



@app.route("/")
def index():
    return("welcome to love letter generation pytorch model")


@app.route('/lovelettergen', methods = ['POST'])
def lovel():
    text_g = request.form['content']
    result = text_generator(text_g, 40, temperature=3)
    return result


