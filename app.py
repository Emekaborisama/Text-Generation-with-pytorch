from flask import Flask,jsonify,request,render_template, make_response
from flask_cors import CORS, cross_origin
from predict_model import text_generator as tg
import os 



app = Flask(__name__)
cors = CORS(app)



@app.route("/")
def index():
    return("welcome to love letter generation pytorch model")


@app.route('/lovelettergen', methods = ['POST'])
def lovel():
    text = requests.get_json()
    text_g = text['text']
    result = tg(text_g, 40, temperature=3)
    return result

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port = 8080, debug=True)

