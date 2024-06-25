
import numpy as np
import math
import sys
import random
import time
#import ImgShuffle
import joblib
from keras.datasets import cifar10
import ImgShuffle_CIFAR10
import FP_Dense
import BP_Dense
import cProfile

#Extracting data from the MNIST database - training and validation data. 
#This takes an input with 4 parameters - img pixel data, img label data, num of train img, num of validation img.

def extract(a):
	img = open(a[0],"rb"); lblT= open(a[1],"rb") 					#opens the pixel and label information
	img.read(16) 									#reads the first 16 bits of info from the img file
	lblT.read(8) 									#reads the first 8 bits of info from the img file
	w,h,s1,s2=28,a[2]+a[3],a[2],a[3]						#size of the img, num of imgs to be read (train+valid), num of train imgs, num of valid imgs
	TrainInput=[[[[0 for x in range(w)] for xc in range(w)] for ux in range(1)] for y in range(s1)] 	#declares an pixel array of size num of train imgs*img size*img size
	TrainOutput=[0 for x in range(s1)]						#declares an label array of size num of train imgs
	ValidInput=[[[[0 for x in range(w)] for xc in range(w)] for ux in range(1)] for y in range(s2)]	#declares an pixel array of size num of valid imgs*img size*img size 
	ValidOutput=[0 for x in range(s2)]						#declares an label array of size num of valid imgs
	for i in range(h):								#loops thru the num of imgs to be read and stores the data into two different data sets - train and valid data based on the value of h. 
		lbl=ord(lblT.read(1))
		#print(i,h)
		if(i<a[2]):
			TrainOutput[i]=int(lbl)
			for j in range(w):
				for k in range(w):
					val=ord(img.read(1))
					TrainInput[i][0][j][k]=float(float(val)/255)
		if(i>=a[2]):
			ValidOutput[i-a[2]]=int(lbl)
			for j in range(w):
				for k in range(w):
					val=ord(img.read(1))
					ValidInput[i-a[2]][0][j][k]=float(float(val)/255)

	img.close();lblT.close()
	#print(TrainInput)
	return TrainInput,TrainOutput,ValidInput,ValidOutput


#Extracting data from the MNIST database - test data. 
#This takes an input with 3 parameters - img pixel data, img label data, num of test img
def extract2(a):
	img = open(a[0],"rb"); lblT= open(a[1],"rb")
	img.read(16)
	lblT.read(8)
	w,h=28,a[2]
	input=[[[[0 for x in range(w)] for xc in range(w)] for ux in range(1)] for y in range(h)] 
	output=[0 for x in range(h)]
	for i in range(h):
		lbl=ord(lblT.read(1))
		output[i]=int(lbl)
		for j in range(w):
			for k in range(w):
				val=ord(img.read(1))
				input[i][0][j][k]=float(float(val)/255)
	img.close();lblT.close()
	return input,output


def weight_init_conv(f,other):
	X=float((f[1]*f[0]*f[2])/(other[0]*other[1]))															#calculates the normalization factor for the weights
	weights=[[[[float(random.gauss(0,1))*math.sqrt(1/X) for i in range(f[0])] for j in range(f[1])] for k in range(f[3])] for mux in range(f[2])]	#weight initialization
	bias=[random.gauss(0,1) for mux in range(f[2])]												#bias initialization
	return weights, bias


def weight_init(out_size,in_size):
	weights=[[float(random.gauss(0,1))*math.sqrt(1/in_size) for i in range(in_size)] for j in range(out_size)]
	bias=[random.gauss(0,1) for i in range(out_size)]
	return weights, bias


#Top function used to call functions that read the img, labels, initialize weights and biases
def Initialization(a,t,f1,m,l):
	train_input,train_output,validation_input,validation_output=extract(a)
	test_input,test_output=extract2(t)
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
		