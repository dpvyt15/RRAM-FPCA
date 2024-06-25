
import numpy as np
import math
import sys
import random
import time
#import ImgShuffle
import joblib
from keras.datasets import cifar10
import ImgShuffle_CIFAR10
import Python_CIFAR10_FP
import Python_CIFAR10_BP1


#devices = tf.config.list_physical_devices()
#tf.debugging.set_log_device_placement(True)


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
def Initialization(a,t,f1,f2,m,l):
	train_input,train_output,validation_input,validation_output=extract(a)
	test_input,test_output=extract2(t)

	u1,u2=int(len(train_input[0][0])),int(len(train_input[0][0][0]))
	y1,y2=int((u1-f1[0]+1)/m[0]),int((u2-f1[1]+1)/m[1])
	y11,y22=int((y1-f2[0]+1)/m[0]),int((y2-f2[1]+1)/m[1])
	length=int(y11*y22*f2[2])
	c1w,c1b=weight_init_conv(f1,m)
	c2w,c2b=weight_init_conv(f2,m)
	l1w,l1b=weight_init(l[0],length)
	l2w,l2b=weight_init(l[1],l[0]) 
	l3w,l3b=weight_init(l[2],l[1])
	return train_input,train_output,validation_input,validation_output,test_input,test_output,c1w,c1b,c2w,c2b,l1w,l1b,l2w,l2b,l3w,l3b


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
	TotalTrain=40000
	TotalValid=10000
	TotalTest=10000
	RGB=1
	L1O=6
	L2O=16
	
	start_time1=time.time()
	TI,TO,VI,VO,ti,to,c1w,c1b,c2w,c2b,l1w,l1b,l2w,l2b,l3w,l3b=Initialization(a=("train-images-idx3-ubyte","train-labels-idx1-ubyte",TotalTrain,TotalValid),t=("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte",TotalTest),f1=(Ni1,Ni1,L1O,RGB),f2=(Ni2,Ni2,L2O,L1O),m=(2,2),l=(120,84,10))
	minibatch,eta,lmbda,epochs,poolsize=10,float(0.05),float(5),10,2
	M1,M2=float(eta)/minibatch,0.99995
	TrainBatches=int(TotalTrain/minibatch)

	end_time1=time.time()

	print("Unpacking time:")
	print(end_time1-start_time1)
	
	for Epo in range(epochs):
		start_time2= time.time()
		count=0
		for k in range(TrainBatches):
			Sc1b,Sc2b,Sl1b,Sl2b,Sl3b=[0 for i in c1b],[0 for i in c2b],[0 for i in l1b],[0 for i in l2b],[0 for i in l3b]
			Sc1w,Sc2w,Sl1w,Sl2w,Sl3w=[[[[0 for i in j] for j in kk] for kk in l] for l in c1w],[[[[0 for i in j] for j in kk] for kk in l] for l in c2w],[[0 for i in j] for j in l1w],[[0 for i in j] for j in l2w],[[0 for i in j] for j in l3w]	
			for start in range(minibatch*k,minibatch*(k+1)):
				print(start)
				x1,x2,y1,x11,x21,y11,t1,o1,a1,o2,a2,o3,a3=Python_CIFAR10_FP.TopInference(TI,start,c1w,c1b,c2w,c2b,l1w,l1b,l2w,l2b,l3w,l3b,RRAMRes,PulseRes)
				C1w,C1b,C2w,C2b,L1w,D1,L2w,D2,L3w,D3= Python_CIFAR10_BP1.TopBackPropNew(a3,o3,l3w,a2,o2,l2w,a1,o1,l1w,t1,y11,x21,x11,c2w,y1,x2,x1,c1w,TI[start],TO[start])
				Sl1b=add(Sl1b,D1); Sl2b=add(Sl2b,D2); Sl3b=add(Sl3b,D3); Sl1w=np.add(Sl1w,L1w); Sl2w=np.add(Sl2w,L2w); Sl3w=np.add(Sl3w,L3w); Sc1w=np.add(Sc1w,C1w); Sc2w=np.add(Sc2w,C2w); Sc1b=add(Sc1b,C1b); Sc2b=add(Sc2b,C2b)
			Sl1b=[i*M1 for i in Sl1b]; l1b=[(l1b[i]-Sl1b[i]) for i in range(len(l1b))]
			Sl2b=[i*M1 for i in Sl2b]; l2b=[(l2b[i]-Sl2b[i]) for i in range(len(l2b))]
			Sl3b=[i*M1 for i in Sl3b]; l3b=[(l3b[i]-Sl3b[i]) for i in range(len(l3b))]
			Sc1b=[i*M1 for i in Sc1b]; c1b=[(c1b[i]-Sc1b[i]) for i in range(len(c1b))]
			Sc2b=[i*M1 for i in Sc2b]; c2b=[(c2b[i]-Sc2b[i]) for i in range(len(c2b))]
			Sl1w=[[i*M1 for i in j] for j in Sl1w]; l1w=[[i*M2 for i in j] for j in l1w]
			Sl2w=[[i*M1 for i in j] for j in Sl2w]; l2w=[[i*M2 for i in j] for j in l2w]
			Sl3w=[[i*M1 for i in j] for j in Sl3w]; l3w=[[i*M2 for i in j] for j in l3w]
			Sc1w=[[[[i*M1 for i in j] for j in kk] for kk in l] for l in Sc1w]; c1w=[[[[i*M2 for i in j] for j in kk] for kk in l] for l in c1w]
			Sc2w=[[[[i*M1 for i in j] for j in kk] for kk in l] for l in Sc2w]; c2w=[[[[i*M2 for i in j] for j in kk] for kk in l] for l in c2w]
			l1w=np.subtract(l1w,Sl1w); l2w=np.subtract(l2w,Sl2w); l3w=np.subtract(l3w,Sl3w); c1w=np.subtract(c1w,Sc1w); c2w=np.subtract(c2w,Sc2w); 

		Accuracy=0
		for ick in range(TotalValid):
			x1,x2,y1,x11,x21,y11,t1,o1,a1,o2,a2,o3,a3=Python_CIFAR10_FP.TopInference(VI,ick,c1w,c1b,c2w,c2b,l1w,l1b,l2w,l2b,l3w,l3b,RRAMRes,PulseRes)
			Accuracy=Accuracy+accu(a3,VO[ick])
		print("Epoch {0}: Validation accuracy {1}".format(Epo, Accuracy))
		#print(Accuracy)
		TI,TO=ImgShuffle_CIFAR10.Shuffle(TI,TO)
		VI,VO=ImgShuffle_CIFAR10.Shuffle(VI,VO)
		elapsed_time_secs2 = time.time() - start_time2
		print("Epoch time:")
		print(elapsed_time_secs2)

	Accuracy=0
	for i in range(TotalTest):
		x1,x2,y1,x11,x21,y11,t1,o1,a1,o2,a2,o3,a3=Python_CIFAR10_FP.TopInference(ti,i,c1w,c1b,c2w,c2b,l1w,l1b,l2w,l2b,l3w,l3b,RRAMRes,PulseRes)
		Accuracy=Accuracy+accu(a3,to[i])
	print("Test accuracy {0}".format(Accuracy))

	elapsed_time_secs = time.time() - start_time
	print("Total Time:")
	print(elapsed_time_secs)

if __name__ =="__main__":
	main()
