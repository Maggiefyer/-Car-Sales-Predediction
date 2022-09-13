'''
Author: Bright NZ 109766140@qq.com
Date: 2022-09-04 00:47:27
LastEditors: Bright NZ 109766140@qq.com
LastEditTime: 2022-09-14 00:31:29
FilePath: \project3\exampleGUI.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from doctest import OutputChecker
from multiprocessing.sharedctypes import Value
from select import select
from unittest import result
import PySimpleGUI as sg
#from app import predict 
from app import *
#from CarANN_test import predict

sg.theme('LightGreen1')


layout = [# "RADIO" makes the radio buttons part of the same group, so when you click one, the other will be unchecked
          [sg.Text("Enter your gender:"), sg.Radio("Male", "RADIO", key='-MALE-'), sg.Radio("Female", "RADIO", key='-FEMALE-')],
          [sg.Text("Enter your age:"), sg.Input(key='-AGE-', do_not_clear=True, size=(20, 1))],
          [sg.Text("Enter your annual salary:"), sg.Input(key='-SALARY-', do_not_clear=True, size=(12, 1))],          
          [sg.Text("Enter your credit card debt:"), sg.Input(key='-DEBT-', do_not_clear=True, size=(10, 1))],
          [sg.Text("Enter your net worth:"), sg.Input(key='-WORTH-', do_not_clear=True, size=(15, 1))],
          [sg.Text('click submit to predict amount',font='Lucida',justification='left', size=(30,1))],
          [sg.Button("Submit"), sg.Button("cancel"), sg.Exit(),sg.In( key='-PREDICTION-',font='Lucida',justification='right',tooltip="Purchase amount prediction", size=(10,1))]
]

#window = sg.Window('Prediction ', layout)
window = sg.Window('Prediction',
                   layout,
                   alpha_channel=0.8, # 设置透明度
                   no_titlebar=True, # 去除顶部状态栏
                   grab_anywhere=True # 允许随意拖动窗口
                   )

def validate(values):
         is_valid = True
         values_invalid = []
         if not values['-MALE-'] and not values['-FEMALE-']:
                     values_invalid.append('Gender')
                     is_valid = False  
         if len(values['-AGE-']) == 0: 
                  values_invalid.append('Age')
                  is_valid = False
         if len(values['-SALARY-']) == 0:
                  values_invalid.append('Salary')
                  is_valid = False
         if len(values['-DEBT-']) == 0:
                  values_invalid.append('Debt')
                  is_valid = False
         if len(values['-WORTH-']) == 0:
                    values_invalid.append('Worth')
                    is_valid = False    
         
       
         result = [values_invalid]
         print(result) 
         print(values)
      
         
         return result, 

#def predict(values_invalid):
     
     #predict(result)
     
     
def generate_error_message(values_invalid):
            error_message = ''
            for value_invalid in values_invalid:
                 error_message += ('\nInvalid' + ':' + value_invalid)
            return error_message   

while True:    
       event, values = window.read( )
       print (event)
       print (values)
       key='_PREDICTION_' 
       if  event == None:
            break        
       if event in (sg.WIN_CLOSED, 'Exit'):
        break
       elif event == 'Submit':            
                    
            if values["-MALE-"]==True:
                      gender = 1
            elif values["-FEMALE-"]==True:
                      gender = 0
            input_test_sample = np.array([[gender, values["-AGE-"], values["-SALARY-"], values["-DEBT-"], values["-WORTH-"]]])
            input_test_sample_scaled = scaler_in.transform(input_test_sample)
            output_predict_sample_scaled = model.predict(input_test_sample_scaled)
            print('Predicted Output (Scaled) =', output_predict_sample_scaled)
            output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
            print('Predicted Output / Purchase Amount ', output_predict_sample)
            prediction=output_predict_sample
            a = 0
            for num in prediction:
                a = a *10 + num
                print(a)
            b = 0
            for num in a:
                b = b *10 + num
                print(b)
        
            prediction=str(np.round(b,2)) 

            print("the result is :", prediction)
            window["-PREDICTION-"].update(prediction, select=True, text_color="red", background_color="gold")
            window["-PREDICTION-"].set_focus()
            sg.popup('The purchase amount is:', prediction)
            
       else:                 
            sg.popup("input is wrong")
            
window.close()