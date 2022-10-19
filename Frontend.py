import os
# Flask
from flask import Flask, request, render_template
import requests

# Declare a flask app
app = Flask('Front-end')

SERVER_ADDRESS = os.environ.get('SERVER_ADDRESS')
SERVER_ADDRESS_FRONT = os.environ.get('SERVER_ADRESS_FRONT')
PORT = os.environ.get('PORT')

# Récupère la quantité des images
def get_json():
    r = requests.get(SERVER_ADDRESS+"/quantite_image")
    json = r.json()
    return json

@app.route('/', methods=['GET', 'POST'])
def json_id():
    json = get_json()
    return render_template("index.html", json=json)

@app.route('/mask')
def mask():
    return render_template("mask.html")

@app.route('/mask', methods=['GET', 'POST'])
def id_valide():
    if request.method == 'POST':
        select = request.form.get('comp_select')
        r = requests.post(SERVER_ADDRESS+"/result_prediction", json=select)
    
    link_image = SERVER_ADDRESS+"/static/images/"+select+".png"
    
    return  render_template("mask.html", link_image=link_image, select=select)

@app.route('/result')
def result():
    return render_template("result.html")

@app.route('/result', methods=['GET', 'POST'])
def resultat(): 
    if request.method == 'POST':
        r = requests.get(SERVER_ADDRESS+"/get_id_image")
        select = request.form.get('fname')
        
    link_image = SERVER_ADDRESS+"/static/images/"+select+".png"
    link_mask = SERVER_ADDRESS+"/static/images/mask/"+select+".png"
    link_pred = SERVER_ADDRESS+"/static/images/prediction/prediction.png"
    return render_template("result.html", link_image=link_image, link_mask=link_mask, link_pred=link_pred)

if __name__ == '__main__':
    app.run(port=PORT)