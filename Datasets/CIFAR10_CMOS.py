
import numpy as np
import math
import sys
sys.path.append('../')
import random
import time
#import ImgShuffle
import joblib
from keras.datasets import cifar10
import ImgShuffle_CIFAR10
import FP_NN
import BP_NN
import VMMNew.RRAM_Programming
import ConvolutionCheck.Programming
from copy import deepcopy



#devices = tf.config.list_physical_devices()
#tf.debugging.set_log_device_placement(True)


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
	weights=[[[[float(random.gauss(0,1))*math.sqrt(1/X) for i in range(f[0])] for j in range(f[1])] for k in range(f[3])] for mux in range(f[2])]	#weight initialization
	bias=[random.gauss(0,1) for mux in range(f[2])]												#bias initialization
	return weights, bias


def weight_init(out_size,in_size):
	weights=[[float(random.gauss(0,1))*math.sqrt(1/in_size) for i in range(in_size)] for j in range(out_size)]
	bias=[random.gauss(0,1) for i in range(out_size)]
	return weights, bias


#Top function used to call functions that read the img, labels, initialize weights and biases
def Initialization(f1,m,l):
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
		y1,y2=int((y1-f1[i][0]+1)/m[i][0]),int((y2-f1[i][1]+1)/m[i][1])

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
	RRAMRes,PulseRes,Split=4,3,1
	Ni1,Ni2,Ni3=5,3,3
	TotalTrain=400
	TotalValid=100
	TotalTest=100
	RGB=3
	L1O=6
	L2O=16
	L30=120
	
	poolsize=[]
	f1=[]
	f1.append([Ni1,Ni1,6,RGB])
	poolsize.append([2,2])
	f1.append([Ni2,Ni2,16,6])
	poolsize.append([2,2])
	f1.append([Ni3,Ni3,120,16])
	poolsize.append([1,1])
	
	start_time1=time.time()
	TI,TO,VI,VO,ti,to,cw,cb,lw,lb=Initialization(f1,poolsize,(84,10))
	minibatch,eta,lmbda,epochs=10,float(0.08),float(5),5
	M1,M2=float(eta)/minibatch,0.99995
	TrainBatches=int(TotalTrain/minibatch)

	end_time1=time.time()
	print("Unpacking time: {0}".format(end_time1-start_time))

	
	for Epo in range(epochs):
		start_time2= time.time()
		count=0

		Scb,Slb=[[0 for i in j] for j in cb],[[0 for i in j] for j in lb]
		Scw,Slw=[[[[[0 for i in j] for j in kk] for kk in l] for l in p] for p in cw], [[[0 for i in j] for j in l] for l in lw]

		for k in range(TrainBatches):
			print("Epoch: {0}, TrainBatch: {1}".format(Epo,k))

			for start in range(minibatch*k,minibatch*(k+1)):
				XXX,x2,y1,o1,a1=FP_NN.TopInference(TI,start,cw,cb,lw,lb,poolsize,RRAMRes,PulseRes)
				max1=max(a1[len(a1)-1])
				next=deepcopy(a1[len(a1)-1])
				next[next.index(max1)]=0
				max2=max(next)
				Ax=accu(a1[len(a1)-1],TO[start])
				if ((max1-max2<0.5) or (Ax==0)):
					Cw,Cb,Lw,D=BP_NN.TopBackPropNew(a1,o1,lw,y1,x2,XXX,cw,TI[start],TO[start],poolsize)
					Slb=[[Slb[i][j]+D[i][j] for j in range(len(lb[i]))] for i in range(len(lb))]
					Slw=[[[Slw[i][j][k]+Lw[i][j][k] for k in range(len(lw[i][j]))] for j in range(len(lw[i]))] for i in range(len(lw))]
					Scw=[[[[[Scw[i][j][k][l][m]+Cw[i][j][k][l][m] for m in range(len(cw[i][j][k][l]))] for l in range(len(cw[i][j][k]))] for k in range(len(cw[i][j]))] for j in range(len(cw[i]))] for i in range(len(cw))] 
					Scb=[[Scb[i][j]+Cb[i][j] for j in range(len(cb[i]))] for i in range(len(cb))]
					count=count+1

				if(count==minibatch):
					lb=[[(lb[i][j]-(Slb[i][j]*M1)) for j in range(len(lb[i]))] for i in range(len(lb))]
					lw=[[[(lw[i][j][k]*M2)-(Slw[i][j][k]*M1) for k in range(len(lw[i][j]))] for j in range(len(lw[i]))] for i in range(len(lw))]
					cw=[[[[[(cw[i][j][k][l][m]*M2)-(Scw[i][j][k][l][m]*M1) for m in range(len(cw[i][j][k][l]))] for l in range(len(cw[i][j][k]))] for k in range(len(cw[i][j]))] for j in range(len(cw[i]))] for i in range(len(cw))] 
					cb=[[cb[i][j]-(Scb[i][j]*M1) for j in range(len(cb[i]))] for i in range(len(cb))]
					count=0
					Scb,Slb=[[0 for i in j] for j in cb],[[0 for i in j] for j in lb]
					Scw,Slw=[[[[[0 for i in j] for j in kk] for kk in l] for l in p] for p in cw], [[[0 for i in j] for j in l] for l in lw]
			
		Accuracy=0
		print("Checking validation accuracy for {0} images for epoch {1}" .format(TotalValid,Epo))
		for ick in range(TotalValid):
			
			XXX,x2,y1,o1,a1=FP_NN.TopInference(VI,ick,cw,cb,lw,lb,poolsize,RRAMRes,PulseRes)
			Accuracy=Accuracy+accu(a1[len(a1)-1],VO[ick])
		print("Epoch {0}: Validation accuracy {1}".format(Epo, Accuracy))
		TI,TO=ImgShuffle_CIFAR10.Shuffle(TI,TO)
		VI,VO=ImgShuffle_CIFAR10.Shuffle(VI,VO)
		elapsed_time_secs2 = time.time() - start_time2
		print("Epoch time:")
		print(elapsed_time_secs2)

	Accuracy=0
	print("Checking test accuracy for {0} images" .format(TotalTest))
	for i in range(TotalTest):
		XXX,x2,y1,o1,a1=FP_NN.TopInference(ti,i,cw,cb,lw,lb,poolsize,RRAMRes,PulseRes)
		Accuracy=Accuracy+accu(a1[len(a1)-1],to[i])
	print("Test accuracy {0}".format(Accuracy))

	elapsed_time_secs = time.time() - start_time
	print("Total Time:")
	print(elapsed_time_secs)

if __name__ =="__main__":
	random.seed(4)
	main()
