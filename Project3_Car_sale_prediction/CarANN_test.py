'''
Author: Bright NZ 109766140@qq.com
Date: 2022-09-11 02:26:38
LastEditors: Bright NZ 109766140@qq.com
LastEditTime: 2022-09-12 13:30:18
FilePath: \project3\PROJECT3\project3\CarANN,py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from pickle import APPEND
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.python.keras
from multiprocessing.dummy import Array
from turtle import register_shape
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from tensorflow import keras
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import *


htl=25

def Predict(hl1):
    # Import data
    data = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')
    # Create input dataset from data
    inputs = data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
    # print("Input data Shape=",inputs.shape)

    # Create output dataset from data
    output = data['Car Purchase Amount']
    # Transform Output
    output = output.values.reshape(-1,1)
    # Show Output Transformed Shape
    # print("Output Data Shape=",output.shape)

    # Scale input
    scaler_in = MinMaxScaler()
    input_scaled = scaler_in.fit_transform(inputs)
    # print(input_scaled)

    # Scale output
    scaler_out = MinMaxScaler()
    output_scaled = scaler_out.fit_transform(output)
    # print(output_scaled)

    #split data
    input_train, input_test, output_train, output_test=train_test_split(input_scaled, output_scaled, test_size=0.2)

    # Create model
    model = Sequential()
    model.add(Dense(25, input_dim=5, activation='relu'))
    model.add(Dense(hl1, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))#linear is the default parameter
    print(model.summary())
    #train model 
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["accuracy"] )
    epochs_hist = model.fit(input_train, output_train, epochs=100, batch_size=30, verbose=1, validation_data=(input_test, output_test))
    # print(epochs_hist.history.keys()) #print dictionary keys
    # input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])

    input_test_sample = np.array([[1,47,39814.522,5958.460188,326373.1812]])# purchase amount（47434.98265）
    output_test_sample = np.array([[28925.70549]])
    # Scale input test sample data

    input_test_sample_scaled = scaler_in.transform(input_test_sample)
    output_test_sample_scaled =scaler_out.transform(output_test_sample)
    #print (input_test_sample_scaled)
    #print (output_test_sample_scaled)
    # Predict output
    output_predict_sample_scaled = model.predict(input_test_sample_scaled)

    # Print predicted output
    #print('Predicted Output (Scaled) =', output_predict_sample_scaled )

    # Unscaled output
    output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
    print (output_predict_sample, output_test_sample)
    Predicted_accuracy =abs(output_predict_sample-output_test_sample)/output_test_sample
    print(' Predict accuracy(Predicted Output/ Purchase Amount ) is ' , Predicted_accuracy)
    #MSE=mean_squared_error(output_test_sample_scaled, output_predict_sample_scaled )
    #return MSE


    #loss, acc=model.evaluate(input_test_sample_scaled, output_test_sample_scaled, verbose=1)# verbose=1 show record verbose=o not
    #print ("test accuracy: %.3f"% acc)
    ERROR_RATE=(abs(output_predict_sample-output_test_sample)/output_test_sample)*100   
    #print ( " MES is:")
    #print ( MSE )
   
    #print ( "Error Percentage(Error%) is:",ERROR_RATE)
    #print (ERROR_RATE)


    a = 0
    for num in ERROR_RATE:
        a = a *10 + num
        print(a)

    b = 0
    for num in a:
        b = b *10 + num
     
    Error_rate = b
    print("Error: % is", Error_rate)
    
    #return Error_rate
if __name__=="__main__":#running condition
     app.run(debug=True)   
    