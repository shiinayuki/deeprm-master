# -*- coding: utf-8 -*-
"""
@Copyright (C) 2022 mewhaku . All Rights Reserved 
@Time ： 2022/1/23 20:25
@Author ： mewhaku
@File ：test.py
@IDE ：PyCharm
"""
#import theano

#print (theano.config.blas.ldflags)

import theano.tensor as T
from theano import function
x=T.dscalar('x')
y=T.dscalar('y')
z=x+y
f=function([x,y],z)
print (f(2,3))