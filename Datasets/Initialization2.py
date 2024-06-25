import numpy as np
import math
import sys
sys.path.append('../')
import random
import time
from keras.datasets import cifar10
from copy import deepcopy


def Separate(input,output):
	TIX=[[[[float(input[l,j,i,k]/255) for i in range(len(input[0,0]))] for j in range(len(input[0]))] for k in range(len(input[0,0,0]))] for l in range(len(input))]
	TOX=[output[i,0] for i in range(len(input))]
	return TIX,TOX

def Separation(input,output):
	TI=[[[[input[l][k][j][i] for i in range(len(input[0][0][0]))] for j in range(len(input[0][0]))] for k in range(len(input[0]))] for l in range(400)]
	TO=[output[i] for i in range(400)]
	VI=[[[[input[l][k][j][i] for i in range(len(input[0][0][0]))] for j in range(len(input[0][0]))] for k in range(len(input[0]))] for l in range(400,500)]
	VO=[output[i] for i in range(400,500)]
	return TI,TO,VI,VO

def weight_init_conv(f,other):
	X=float((f[1]*f[0]*f[2])/(other[0]*other[1]))															#calculates the normalization factor for the weights
	weights=[[[[float(random.gauss(0,1))*math.sqrt(2/X) for i in range(f[0])] for j in range(f[1])] for k in range(f[3])] for mux in range(f[2])]	#weight initialization
	bias=[random.gauss(0,1) for mux in range(f[2])]												#bias initialization
	return weights, bias


def weight_init(out_size,in_size):
	weights=[[float(random.gauss(0,1))*math.sqrt(1/in_size) for i in range(in_size)] for j in range(out_size)]
	bias=[random.gauss(0,1) for i in range(out_size)]
	return weights, bias


#Top function used to call functions that read the img, labels, initialize weights and biases

def Initialization(f1,m,l,se):
	random.seed(se)
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	TIX,TOX=Separate(x_train,y_train)
	train_input,train_output,validation_input,validation_output=Separation(TIX,TOX)
	test_input,test_output=Separate(x_test,y_test)

	y1,y2=len(train_input[0][0]),len(train_input[0][0][0])
	cw,cb,lw,lb=[],[],[],[]

	for i in range(len(f1)):
		c1w,c1b=weight_init_conv(f1[i],m[i])
		cw.append(c1w)
		cb.append(c1b)


	for i in range(len(l)):
		l1w,l1b=weight_init(l[i][1],l[i][0])
		lw.append(l1w)
		lb.append(l1b)

	return train_input,train_output,validation_input,validation_output,test_input,test_output,cw,cb,lw,lb


def BN(f):
	gamma,beta=[],[]
	RunMean,RunVar=[],[]
	for i in range(len(f)):
		gamma.append([random.gauss(0,1) for j in range(f[i])])
		beta.append([random.gauss(0,1) for j in range(f[i])])

		RunMean.append([0 for j in range(f[i])])
		RunVar.append([0 for j in range(f[i])])

	return gamma,beta,RunMean,RunVar