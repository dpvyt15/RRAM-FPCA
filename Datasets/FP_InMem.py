
import math
import sys
sys.path.append('../')
import random
import numpy as np
import time
import cmath
from joblib import Parallel, delayed
import Convolution.ForNNTop
import VMM.ForNNTop


def convolutions(Input,c2):
	Final=[[Input[i][j]+c2 for j in range(len(Input[0]))] for i in range(len(Input))]
	return Final


def maxpool(X):
	Row,Col=len(X),len(X[0])
	T=[[max(X[2*i][2*j],X[2*i][2*j+1],X[2*i+1][2*j],X[2*i+1][2*j+1]) for j in range(int(Col/2))] for i in range(int(Row/2))]
	return T

def sigmoid(A):
	B=[float(1/(1+math.exp(-1*i))) for i in A]
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

def convolutionsTop(In,convb):

	l5=int(len(convb))
	XXX=Parallel(n_jobs=-1)(delayed(convolutions)(In[i],convb[i]) for i in range(l5))
	x2=Parallel(n_jobs=-1)(delayed(maxpool)(XXX[i]) for i in range(l5))
	y1=Parallel(n_jobs=-1)(delayed(ReLu)(x2[i]) for i in range(l5))
	return XXX,x2,y1

		
def TopInference(Input,start,convw,convb,lw,lb,RRAMRes,PulseRes):

	#XXX: outputs of conv layers before Maxpool and ReLU, 0-> len(convw)-1. For n layer, XXX should be n
	#x2: conv layer outputs after Maxpool, before ReLU, 0-> len(convw)-1. For n layer, x2 should be n
	#y1: conv layer outputs after ReLU, consists of the input layer as well. For n layer, y1 should be n+1

	XXX,x2,y1=[],[],[]

	y1.append(Input[start])

	for lm in range(len(convw)):
		X1a=Convolution.ForNNTop.Top(convw[lm],y1[lm])
		X1,X2,Y1=convolutionsTop(X1a,convb[lm])
		#print(Y1)
		XXX.append(X1)
		x2.append(X2)
		y1.append(Y1)

	#o1: outputs of FCC layers before ReLU, 0-> len(lw)-1. For n layer, o1 should be n
	#a1: FCC layer outputs after ReLU, consists of the input layer as well. For n layer, a1 should be n+1
		

	o1,a1=[],[]
	a1.append(CNFC(y1[len(convw)]))

	#print(len(a1[0]), len(lw), len(lb), len(lw[0]), len(lw[0][0]))

	#print(lb[0])
	
	for lm in range(len(lw)-1):
		O1=VMM.ForNNTop.Top(lw[lm],a1[lm])
		O1=[O1[j]+lb[lm][j] for j in range(len(O1))]
		A1=[max(0,i) for i in O1]
		o1.append(O1)
		a1.append(A1)

	X=VMM.ForNNTop.Top(lw[len(lw)-1],a1[len(lw)-1])
	X=[X[j]+lb[len(lw)-1][j] for j in range(len(X))]

	o1.append(X)
	maxX,minX=Convolution.GeneralFunctions.sortingSingle(X)
	a4=[math.exp(i-maxX) for i in o1[len(lw)-1]]
	sumT=sum(a4)
	a3=[float(i)/sumT for i in a4]
	a1.append(a3)
	return XXX,x2,y1,o1,a1	

