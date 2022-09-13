from pickle import APPEND
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.python.keras
from multiprocessing.dummy import Array
from turtle import register_shape
from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
from ann_visualizer.visualize import ann_viz
from sklearn.preprocessing import MinMaxScaler
from graphviz import Graph
import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin'
from keras.models import Sequential,load_model

from multiprocessing.dummy import Array
from tensorflow import keras
from turtle import register_shape
from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np

from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow.python.keras
from multiprocessing.dummy import Array
from turtle import register_shape
from sklearn.model_selection import train_test_split
from tensorflow import *
#import ann_visualizer
#from  ann_visualizer.visualize import ann_viz
   

app=Flask(__name__)
cors=CORS(app)


# Import data
data = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')
# print(data)

# Plot data
# sns.pairplot(data)
# plt.show(block=True)

# Create input dataset from data
inputs = data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
# Show Input Data
# print(inputs)
# Show Input Shape
# print("Input data Shape=",inputs.shape)

# Create output dataset from data
output = data['Car Purchase Amount']
# Show Output Data
# print(output)
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
ann_viz(model, view="True", filename="network7.gv", title="Car Purchase Prediction")

# Train model
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["accuracy"] )
epochs_hist = model.fit(input_train, output_train, epochs=100, batch_size=30, verbose=1, validation_data=(input_test, output_test))
# print(epochs_hist.history.keys()) #print dictionary keys

model.save("model_car_purchase.hdf5")
model.summary
print(model)

#scores = model.evaluate(input_test, scaler_out.inverse_transform(output_test.reshape(-1, 1)), verbose=0)
#print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
#用训练好的模型衡量测试数据精确度
results = model.evaluate(input_test, output_test)
print("训练好的模型衡量测试数据精确度：", results )

                      
# Plot the training graph to see how quickly the model learns
#plt.plot(epochs_hist.history['loss'])
#plt.plot(epochs_hist.history['val_loss'])

#plt.title('Model Loss Progression During Training/Validation')
#plt.ylabel('Training and Validation Losses')
#plt.xlabel('Epoch Number')
#plt.legend(['Training Loss', 'Validation Loss'])
#plt.show(block=True)

# Evaluate model
# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 
# ***(Note that input data must be normalized)***

# input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])
# input_test_sample2 = np.array([[1, 46.73, 61370.67, 9391.34, 462946.49]])
# (1,43,53798.55112,11160.35506,638467.1773,42925.70921)
# input_test_sample = np.array([[1,50,51752.23445,10985.69656,629312.4041]])# purchase amount（47434.98265）
# output_test_sample = np.array([[47434.98205]])
#4. 1,43,53798.55112,11160.35506,638467.1773,42925.70921
#5. 1,58,79370.03798,14426.16485,548599.0524,67422.36313
#6. 1,57,59729.1513,5358.712177,560304.0671,55915.46248
#7. 1,57,68499.85162,14179.47244,428485.3604,56611.99784
#8. 1,47,39814.522,5958.460188,326373.1812,28925.70549
input_test_sample = np.array([[ 1,57,68499.85162,14179.47244,428485.3604]])# purchase amount（47434.98265）
output_test_sample = np.array([[56611.99784]])
# Scale input test sample data

input_test_sample_scaled = scaler_in.transform(input_test_sample)
output_test_sample_scaled =scaler_out.transform(output_test_sample)
print (input_test_sample_scaled)
print (output_test_sample_scaled)
# Predict output
output_predict_sample_scaled = model.predict(input_test_sample_scaled)

# Print predicted output
print('Predicted Output (Scaled) =', output_predict_sample_scaled )

# Unscaled output
output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
print ( output_predict_sample, output_test_sample)
Predicted_accuracy =1-abs(output_predict_sample-output_test_sample)/output_test_sample
print(' Predict accuracy(Predicted Output/ Purchase Amount ) is ' , Predicted_accuracy)
#loss, acc=model.evaluate(input_test_sample_scaled, output_test_sample_scaled, verbose=1)# verbose=1 show record verbose=o not
#print ("test accuracy: %.3f"% acc)
MSE=mean_squared_error(output_test_sample_scaled, output_predict_sample_scaled)
#mean_squared_error / mse 均方误差，常用的目标函数，公式为((y_pred-y_true)**2).mean()
ERROR_RATE=(abs(output_predict_sample-output_test_sample)/output_test_sample)*100
#acc = epochs_hist.history['accuracy']
#loss = epochs_hist.history['loss']
print ( " MES is:")
print ( MSE )
print ( "Error Percentage(Error%) is:",ERROR_RATE)
#print (ERROR_RATE)


a = 0
for num in ERROR_RATE:
    a = a *10 + num
#print(a)

b = 0
for num in a:
    b = b *10 + num
    print(b)
 
    
#print('%.2f' % b)

#print(" ".join('%s' %id for id in a ))
#print (acc)
#print (loss)    
# 计算误差
result = np.mean(abs(output_predict_sample_scaled - output_test_sample_scaled))
print("The mean error of linear regression:")
print(result)

model= keras.models.load_model("model_car_purchase.hdf5")#打印保存的模型，查看结构
for index in range(3):
    layer=model.get_layer(index=index)  
    print(model)
    model.summary

if __name__=="__main__":#running condition
     app.run(debug=True)