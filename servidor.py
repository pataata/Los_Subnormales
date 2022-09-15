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
    X_to_pred = np.array( 
            [[float(contenido["CryoSleep"]),
            float(contenido["RoomService"]),
            float(contenido["Spa"]),
            float(contenido["VRDeck"]),
            float(contenido["VIP"]),
            float(contenido["FoodCourt"]),
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
            float(contenido["HomePlanet_Earth"]),
            float(contenido["HomePlanet_Mars"]),
            float(contenido["01"]),
            float(contenido["08"]),
            float(contenido["constant"])]])

    # Predict with highest score model.
    y_pred = app_model.predict(X_to_pred.reshape(1,-1))

    #Regresar la salida del modelo
    #print(resultado)
    return jsonify({"resultado":str(y_pred[0])})

@servidorWeb.route("/test",methods=['POST'])
def processData():
    #Procesar datos de entrada 
    contenido = request.json
    X_to_pred = np.array( 
            [[float(contenido["CryoSleep"]),
            float(contenido["RoomService"]),
            float(contenido["Spa"]),
            float(contenido["VRDeck"]),
            float(contenido["VIP"]),
            float(contenido["FoodCourt"]),
            contenido["Cabin"],
            contenido["HomePlanet"],
            float(contenido["Age"]),
            #float(contenido["TotalBill"]),
            contenido["Side"],
            float(contenido["ShoppingMall"]),
            contenido["Destination"],
            float(contenido["GroupID"]),
            #float(contenido["constant"])
            ]])
    print(X_to_pred)

    # Predict with highest score model.
    #y_pred = app_model.predict(X_to_pred.reshape(1,-1))

    #Regresar la salida del modelo
    #print(resultado)
    return jsonify({"resultado":'JSON Recibido'})

if __name__ == '__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8080')
