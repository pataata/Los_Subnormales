"""
This file contains the services of our server in order to predict the
result of the inter-dimensional travel.


Authors:
    - Luis Ignacio Ferro Salinas
    - Rubén Sánchez Mayén


Last update:
    september 14th, 2022
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load
#from werkzeug.utils import secure_filename
import os

#Cargar el modelo
app_model = load("modelo_preliminar.joblib")

#Generar el servidor (Back-end)
servidorWeb = Flask(__name__)

@servidorWeb.route("/sendData",methods=['POST'])
def sendData():
    #Procesar datos de entrada 
    contenido = request.json
    datosEntrada = np.array( \
            [float(contenido["CryoSleep"])
            float(contenido["RoomService"]),
            float(contenido["Spa"]),
            float(contenido["VRDeck"]),
            float(contenido["VIP"]),
            float(contenido["FoodCourt"]),
            float(contenido["Age"]),
            float(contenido["C"]),
            float(contenido["E"]),
            float(contenido["D"]),
            float(contenido["B"]),
            float(contenido["T"]),
            float(contenido["HomePlanet_Europa"]),
            float(contenido["Age"]),
            float(contenido["TotalBill"]),
            float(contenido["S"]),
            float(contenido["ShoppingMall"]),
            float(contenido["Destination_55Cancrie"]),
            float(contenido["Destination_PSOJ318.5-22"]),
            float(contenido["Destination_TRAPPIST-1e"]),
            float(contenido["A"]),
            float(contenido["08"]),
            float(contenido["constant"])])


    return datosEntrada

    #Utilizar el modelo
    resultado = app_model.predict(datosEntrada.reshape(1,-1))

    #Regresar la salida del modelo
    #print(resultado)
    return jsonify({"resultado":str(resultado[0])})

if __name__ == '__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8080')
