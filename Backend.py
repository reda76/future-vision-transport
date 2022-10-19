import os, glob
import numpy as np

# Flask
from flask import Flask, jsonify, request

#tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt

IMG_WIDTH_HEIGHT = (256, 256)

SERVER_ADDRESS = os.environ.get('SERVER_ADDRESS')
PORT = os.environ.get('PORT')

# Declare a flask app
app = Flask('API')

# Contiens les images et les masks
def images_file():
    val_input_dir = 'static/images'
    val_target_dir = 'static/images/mask'

    val_input_path = sorted(glob.glob(val_input_dir + '/*.png'))
    val_target_path = sorted(glob.glob(val_target_dir +'/*.png'))
    
    return val_input_path, val_target_path

# Récupère la quantité d'images pour afficher la liste déroulante et l'envoie à l'adresse /quantite_image
@app.route('/quantite_image')
def image_quantites():
    images, masks = images_file()
    images_quantite = len(images)
    
    return jsonify(images_quantite)

def model_predict(img):
    unet = keras.models.load_model('model/Unet_adam_mse_aug_final.h5')
    img = img_to_array(load_img(f'{img}', target_size=(IMG_WIDTH_HEIGHT)))/255
    img = np.expand_dims(img, 0)
    preds = unet.predict(img)

    return preds

# Récupère la prédiction et l'envoie à l'adresse /result_prediction
@app.route('/result_prediction', methods=['POST'])
def prediction():  
    images, masks = images_file()
    json = request.json

    print("CECI EST LE PRINT DU JSON FINAL")
    print(json)
    preds = model_predict(images[int(json)])
    
    pred_mask = np.argmax(preds, axis=-1)
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    pred_mask = np.squeeze(pred_mask)
    
    if os.path.exists("static/images/prediction") == False:
        os.mkdir("static/images/prediction")
    
    plt.imsave('static/images/prediction/prediction.png', pred_mask, cmap='nipy_spectral')
    return json

if __name__ == '__main__':
    app.run(port=PORT)
