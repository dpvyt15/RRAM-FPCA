
import numpy as np
import math
import sys
import random
import time
#import ImgShuffle
import joblib
from keras.datasets import cifar10
import ImgShuffle_CIFAR10
import FP_NN2
import BP_NN2
import cProfile


#devices = tf.config.list_physical_devices()
#tf.debugging.set_log_device_placement(True)


"""5 layer CNN for MNIST database using CPU/GPU. This is the top module that calls the different scripts for different tasks."""

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
		y1,y2=int((y1)/m[i][0]),int((y2)/m[i][1])

	length=int(y1*y2*f1[len(f1)-1][2])

	for i in range(len(l)):
		l1w,l1b=weight_init(l[i],length)
		lw.append(l1w)
		lb.append(l1b)
		length=l[i]

	return train_input,train_output,validation_input,validation_output,test_input,test_output,cw,cb,lw,lb


def add(in1,in2):
	u=[(in1[i]+in2[i]) for i in range(len(in1))]
	return u


def accu(input,output):
	u=1 if input[output]==max(input) else 0
	return u

def main():
	start_time = time.time()
	xA,yA=10,10
	RRAMRes=3
	PulseRes=4
	Split=1
	Ni1=5
	Ni2=3
	TotalTrain=400
	TotalValid=100
	TotalTest=100
	RGB=1
	L1O=6
	L2O=16
	
	f1=[]
	poolsize=[]
	f1.append([Ni1,Ni1,6,RGB])
	poolsize.append([2,2])
	f1.append([Ni2,Ni2,16,6])
	poolsize.append([2,2])
	
	start_time1=time.time()
	TI,TO,VI,VO,ti,to,cw,cb,lw,lb=Initialization(("train-images-idx3-ubyte","train-labels-idx1-ubyte",TotalTrain,TotalValid),("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte",TotalTest),f1,poolsize,(120,84,10))
	minibatch,eta,lmbda,epochs=10,float(0.08),float(5),10
	M1,M2=float(eta)/minibatch,0.99995
	TrainBatches=int(TotalTrain/minibatch)

	end_time1=time.time()

	print("Unpacking time:")
	print(end_time1-start_time1)
	
	for Epo in range(epochs):
		start_time2= time.time()
		count=0
		for k in range(TrainBatches):
			print(k)
			Scb,Slb=[[0 for i in j] for j in cb],[[0 for i in j] for j in lb]
			Scw,Slw=[[[[[0 for i in j] for j in kk] for kk in l] for l in p] for p in cw], [[[0 for i in j] for j in l] for l in lw]
			for start in range(minibatch*k,minibatch*(k+1)):
				XXX,x2,y1,o1,a1=FP_NN2.TopInference(TI,start,cw,cb,lw,lb,poolsize,RRAMRes,PulseRes)
				Cw,Cb,Lw,D=BP_NN2.TopBackPropNew(a1,o1,lw,y1,x2,XXX,cw,TI[start],TO[start],poolsize)
				Slb=[[Slb[i][j]+D[i][j] for j in range(len(lb[i]))] for i in range(len(lb))]
				Slw=[[[Slw[i][j][k]+Lw[i][j][k] for k in range(len(lw[i][j]))] for j in range(len(lw[i]))] for i in range(len(lw))]
				Scw=[[[[[Scw[i][j][k][l][m]+Cw[i][j][k][l][m] for m in range(len(cw[i][j][k][l]))] for l in range(len(cw[i][j][k]))] for k in range(len(cw[i][j]))] for j in range(len(cw[i]))] for i in range(len(cw))] 
				Scb=[[Scb[i][j]+Cb[i][j] for j in range(len(cb[i]))] for i in range(len(cb))]
			lb=[[(lb[i][j]-(Slb[i][j]*M1)) for j in range(len(lb[i]))] for i in range(len(lb))]
			lw=[[[(lw[i][j][k]*M2)-(Slw[i][j][k]*M1) for k in range(len(lw[i][j]))] for j in range(len(lw[i]))] for i in range(len(lw))]
			cw=[[[[[(cw[i][j][k][l][m]*M2)-(Scw[i][j][k][l][m]*M1) for m in range(len(cw[i][j][k][l]))] for l in range(len(cw[i][j][k]))] for k in range(len(cw[i][j]))] for j in range(len(cw[i]))] for i in range(len(cw))] 
			cb=[[cb[i][j]-(Scb[i][j]*M1) for j in range(len(cb[i]))] for i in range(len(cb))]
			
		Accuracy=0
		for ick in range(TotalValid):
			XXX,x2,y1,o1,a1=FP_NN.TopInference(VI,ick,cw,cb,lw,lb,poolsize,RRAMRes,PulseRes)
			Accuracy=Accuracy+accu(a1[len(a1)-1],VO[ick])
		print("Epoch {0}: Validation accuracy {1}".format(Epo, Accuracy))
		#print(Accuracy)
		TI,TO=ImgShuffle_CIFAR10.Shuffle(TI,TO)
		VI,VO=ImgShuffle_CIFAR10.Shuffle(VI,VO)
		elapsed_time_secs2 = time.time() - start_time2
		print("Epoch time:")
		print(elapsed_time_secs2)

	Accuracy=0
	for i in range(TotalTest):
		XXX,x2,y1,o1,a1=FP_NN.TopInference(ti,i,cw,cb,lw,lb,poolsize,RRAMRes,PulseRes)
		Accuracy=Accuracy+accu(a1[len(a1)-1],to[i])
	print("Test accuracy {0}".format(Accuracy))

	elapsed_time_secs = time.time() - start_time
	print("Total Time:")
	print(elapsed_time_secs)

if __name__ =="__main__":
	random.seed(2)
	#main()
	cProfile.run('main()')
