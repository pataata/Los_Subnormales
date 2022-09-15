"""
This file contains the services of our server in order to predict the
result of the inter-dimensional travel.


Authors:
    - Luis Ignacio Ferro Salinas
    - Rubén Sánchez Mayén


Last update:
    september 15th, 2022
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

#Decoder
def encoder(val,options):
    """
    val: categoria. String
    options: categorias. Array de strings
    """
    for i in range(len(options)):
        if(options[i] == val):
            options[i] = 1.0
        else:
            options[i] = 0.0
    return options

@servidorWeb.route("/sendData",methods=['POST'])
def processData():
    
    contenido = request.json
    print(contenido)
    #Procesar datos de entrada 

    # Codificar datos
    encoded_cabins = encoder(contenido["Cabin"],['C','E','D','B','T','A'])
    encoded_planets = encoder(contenido['HomePlanet'],['Europa','Earth','Mars'])
    encoded_side = encoder(contenido['Side'],['Starboard','Port'])
    encoded_destination = encoder(contenido['Destination'],['55 Cancrie','PSOJ318.5-22','TRAPPIST-1e'])
    encoded_group = encoder(contenido['GroupID'],['01','08'])
    totalBill = float(contenido['RoomService']) + float(contenido['Spa']) + float(contenido['VRDeck']) + float(contenido['FoodCourt']) + float(contenido['ShoppingMall'])

    # Cargar los datos a un numpy array
    X_to_pred = np.array([[
        float(contenido["CryoSleep"]), #CryoSleep
        float(contenido["RoomService"]), #RoomService
        float(contenido["Spa"]), #Spa
        float(contenido["VRDeck"]), #VRDeck
        float(contenido["VIP"]), #VIP
        float(contenido["FoodCourt"]), #FoodCourt
        encoded_cabins[0], #C
        encoded_cabins[1], #E
        encoded_cabins[2], #D
        encoded_cabins[3], #B
        encoded_cabins[4], #T
        encoded_planets[0], #HomePlanet_Europa
        float(contenido["Age"]), #Age
        totalBill, #TotalBill
        encoded_side[0], #S
        float(contenido["ShoppingMall"]), #ShoppingMall
        encoded_destination[0], #Destination_55Cancrie
        encoded_destination[1], #Destination_PSOJ318.5-22
        encoded_destination[2], #Destination_TRAPPIST-1e
        encoded_cabins[5], #A
        encoded_planets[1], #HomePlanet_Earth
        encoded_planets[2], #HomePlanet_Mars
        encoded_group[0], #01
        encoded_group[1], #08
        1.0 #constant
    ]])

    # Predict with highest score model.
    y_pred = app_model.predict(X_to_pred.reshape(1,-1))

    #Regresar la salida del modelo
    print(y_pred[0])
    return jsonify({"resultado":str(y_pred[0])})

if __name__ == '__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8080')
