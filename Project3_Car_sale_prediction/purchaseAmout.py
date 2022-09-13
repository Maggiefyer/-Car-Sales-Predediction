'''
Author: Bright NZ 109766140@qq.com
Date: 2022-09-13 19:42:09
LastEditors: Bright NZ 109766140@qq.com
LastEditTime: 2022-09-13 23:50:10
FilePath: \project3\purchaseAmout.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import PySimpleGUI as sg
from multiprocessing.dummy import Array
from app import *



sg.theme('LightGreen1')

layout = [[sg.Text("Enter your gender:")],   
           
          [sg.Text('Gender', size =(15, 1)),
    	  sg.Radio('Male', 'group 1', key='-MALE-'),
    	  sg.Radio('Female', 'group 1', key='-FEMALE-')],
          [sg.Text("Enter your age:"), sg.Input(key='-AGE-',  size=(30, 1))],
          [sg.Text("Enter your annual salary:"), sg.Input(key='-SALARY-',  size=(22, 1))],          
          [sg.Text("Enter your credit card debt:"), sg.Input(key='-DEBT-',  size=(20, 1))],
          [sg.Text("Enter your net worth:"), sg.Input(key='-WORTH-',  size=(25, 1))],
          [sg.Text('click submit to predict amount',font='Lucida', size=(40,1))],
          [sg.Button("Submit"), sg.Button("cancel"), sg.Exit()]
]          
          
         
window = sg.Window('Car Sales Predictor', layout)
#NEED TO ADD VALIDATION
while True:
    event, values = window.read()
    if event == "Cancel":
        break

    if event == "Submit":
        if values["-MALE-"]==True:
            gender = 1
        elif values["-FEMALE-"]==True:
            gender = 0
        input_test_sample = np.array([[gender, values["-AGE-"], values["-SALARY-"], values["-DEBT-"], values["-WORTH-"]]])
        print (input_test_sample)
        input_test_sample_scaled = scaler_in.transform(input_test_sample)
        output_predict_sample_scaled = model.predict(input_test_sample_scaled)
        #print('Predicted Output (Scaled) =', output_predict_sample_scaled)
        output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
        #print('Predicted Output / Purchase Amount ', output_predict_sample)
        a = 0
        for num in output_predict_sample:
            a = a *10 + num
            print(a)
        b = 0
        for num in a:
            b = b *10 + num
        print(b)
        
        output_predict_sample=str(np.round(b,2))

        print("the result is :", output_predict_sample)
       
        sg.popup('The purchase amount is:' , output_predict_sample)
    elif event == sg.WINDOW_CLOSED:
        break
window.close()
