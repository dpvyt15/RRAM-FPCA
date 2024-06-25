
import math
import sys
sys.path.append('../')
import random
import numpy as np
import time
import cmath
from joblib import Parallel, delayed
import ConvolutionCheck.ConvInMemP
import VMMNew.InMemVMMP
import ConvolutionCheck.GeneralFunctions
from numpy import array, exp


def biasAdd(Input,c2):
	Final=[[Input[i][j]+c2 for j in range(len(Input[0]))] for i in range(len(Input))]
	return Final


def maxpool(X,poolsize):
	Row,Col=len(X),len(X[0])
	poolX,poolY=poolsize[0],poolsize[1]
	T=[[max(X[poolX*i][poolY*j],X[poolX*i][(poolY*(j+1))-1],X[(poolX*(i+1))-1][poolY*j],X[(poolX*(i+1))-1][(poolY*(j+1))-1]) for j in range(int(Col/poolX))] for i in range(int(Row/poolY))]
	return T

def sigmoid(A):
	C=array(A)
	B=1/(1+exp(-C))
	return B

def CNFC(input):
	n_filters=len(input)
	ux=int(len(input)*len(input[0])*len(input[0][0]))
	FCIn=[0 for i in range(ux)]
	u=0
	for i in range(n_filters):
		rows=len(input[i])
		for j in range(rows):
			col=len(input[i][j])
			for k in range(col):
				FCIn[u]=input[i][j][k]
				u=u+1
	return FCIn

def ReLu(input):
	A=[[max(i,0) for i in j] for j in input]
	return A

def convolutionsTop(In,convb,poolsize):

	l5=int(len(convb))
	XXX=[[[In[k][i][j]+convb[k] for j in range(len(In[0][0]))] for i in range(len(In[0]))] for k in range(l5)]
	x2,y1=[],[]
	for i in range(l5):
		x2.append(maxpool(XXX[i],poolsize))
		
	y1=[[[max(i,0) for i in j] for j in k] for k in x2]
	return XXX,x2,y1

		
def TopInference(Input,start,cW,cW2,dcW,dcW2,mcW,convb,lW,lW2,dlW,dlW2,mlW,lb,poolsize,RRAMRes,PulseRes):

	#XXX: outputs of conv layers before Maxpool and ReLU, 0-> len(convw)-1. For n layer, XXX should be n
	#x2: conv layer outputs after Maxpool, before ReLU, 0-> len(convw)-1. For n layer, x2 should be n
	#y1: conv layer outputs after ReLU, consists of the input layer as well. For n layer, y1 should be n+1

	#start_time1=time.time()

	XXX,x2,y1=[],[],[]

	y1.append(Input[start])

	for lm in range(len(cW)):
		X1a=ConvolutionCheck.ConvInMemP.ConvolutionInMem(cW[lm],dcW[lm],cW2[lm],dcW2[lm],y1[lm],mcW[lm],PulseRes,RRAMRes,-1)
		X1,X2,Y1=convolutionsTop(X1a,convb[lm],poolsize[lm])
		XXX.append(X1)
		x2.append(X2)
		y1.append(Y1)

	#end_time1=time.time()

	#print("Convolution Time: {0}".format(end_time1-start_time1))

	#o1: outputs of FCC layers before ReLU, 0-> len(lw)-1. For n layer, o1 should be n
	#a1: FCC layer outputs after ReLU, consists of the input layer as well. For n layer, a1 should be n+1
		

	o1,a1=[],[]
	a1.append(CNFC(y1[len(cW)]))

	#print(len(a1[0]), len(lw), len(lb), len(lw[0]), len(lw[0][0]))

	#print(lb[0])

	#start_time2=time.time()
	
	for lm in range(len(lW)-1):
		O1=VMMNew.InMemVMMP.VMMMultiResSplit(lW[lm],dlW[lm],dlW2[lm],a1[lm],RRAMRes,PulseRes,mlW[lm],-1)
		O1=[O1[j]+lb[lm][j] for j in range(len(O1))]
		A1=[max(i,0) for i in O1]
		o1.append(O1)
		a1.append(A1)


	X=VMMNew.InMemVMMP.VMMMultiResSplit(lW[len(lW)-1],dlW[len(lW)-1],dlW2[len(lW)-1],a1[len(lW)-1],RRAMRes,PulseRes,mlW[len(lW)-1],-1)
	X=[X[j]+lb[len(lW)-1][j] for j in range(len(X))]

	o1.append(X)
	#maxX,minX=ConvolutionCheck.GeneralFunctions.sortingSingle(X)
	a4=[math.exp(i) for i in o1[len(lW)-1]]
	sumT=sum(a4)
	a3=[float(i)/sumT for i in a4]
	a1.append(a3)

	#end_time2=time.time()

	#print("FCC Time: {0}".format(end_time2-start_time2))

	return XXX,x2,y1,o1,a1	

