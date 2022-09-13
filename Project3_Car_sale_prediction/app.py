'''
Author: Bright NZ 109766140@qq.com
Date: 2022-09-13 13:06:32
LastEditors: Bright NZ 109766140@qq.com
LastEditTime: 2022-09-13 19:40:45
FilePath: \project3\app.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from multiprocessing.dummy import Array
from tensorflow import keras
from turtle import register_shape
from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow.python.keras
from multiprocessing.dummy import Array
from turtle import register_shape
from sklearn.model_selection import train_test_split


data = pd.read_csv('Car_Purchasing_Data.csv')
inputs = data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
   
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
model.add(Dense(25, activation='relu'))
    # model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))#linear is the default parameter
print(model.summary())
    #train model 
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["accuracy"])
epochs_hist = model.fit(input_train, output_train, epochs=100, batch_size=30, verbose=1,validation_data=(input_test, output_test))
    # print(epochs_hist.history.keys()) #print dictionary keys
    # input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])    

model.save('model_car_purchase_app.hdf5')
#model.summary
print(model) 
Pdata=[]
#def predict(Pdata):    
    #model=keras.models.load_model('model_car_purchase_app.hdf5')
    #model.summary
    #print(model) 
predict_data=[[0]] 
    #Pdata=pd.DataFrame(columns=['Gender,Age,Annual Salary,Credit Card Debt,Net Worth,Car Purchase Amount'],
                              #data=np.array([Pdata]).reshape(1, 5)) 
    #Pdata["Gender"].unique()
input_data=np.array([[1,45,89065,35678.45,1234567]])
    #Scale input data

input_data_scaled = scaler_in.transform(input_data)
predict_data_scaled=scaler_out.transform(predict_data)
     #Predict output
predict_data_scaled = model.predict(input_data_scaled)
     # Scale output data
prediction=scaler_out.inverse_transform(predict_data_scaled)
print(prediction)
a = 0
for num in prediction:
    a = a *10 + num
    print(a)
    b = 0
for num in a:
    b = b *10 + num
    print(b)
PREDICTION=str(np.round(b,2))

print("the result is :", PREDICTION)

#return PREDICTION
