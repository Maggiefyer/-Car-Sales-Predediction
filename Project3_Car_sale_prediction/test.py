'''
Author: Bright NZ 109766140@qq.com
Date: 2022-08-30 20:40:49
LastEditors: Bright NZ 109766140@qq.com
LastEditTime: 2022-09-11 21:34:53
FilePath: \project3\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# unit testing cases for testing the accuracy when changing of the number of the neurons in the hidden layer

from __future__ import print_function
from re import A
import pytest
from CarANN_test import Predict
#from CarANN1_test import Predict

Bestcase=0.00011302865902765955
def test_minimum_neurons():
          A = Predict(10)     
          assert A<=Bestcase           
          
def test_medium_neurons():
          B =Predict(15)       
          assert B<=Bestcase          
      
def test_maximum_neurons():
          C= Predict(25)        
          assert C<=Bestcase
   

                 
