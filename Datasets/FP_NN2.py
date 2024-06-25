
import math
import sys
import random
import numpy as np
import time
import cmath
from joblib import Parallel, delayed


def pad_with(vector, pad_width, iaxis, kwargs):
	pad_value = kwargs.get('padder', 10)
	vector[:pad_width[0]] = pad_value
	vector[-pad_width[1]:] = pad_value

def convolutions(Input,Weight,c2):
	Inputs=len(Input)
	x1,x2=len(Input[0])-len(Weight[0])+1,len(Input[0][0])-len(Weight[0][0])+1
	#print("In convolutions: {0},{1}".format(x1,x2))
	I=[[[sum(sum([[Input[um][i+a][j+b]*Weight[um][a][b] for b in range(len(Weight[0][0]))] for a in range(len(Weight[0]))],[])) for j in range(x2)] for i in range(x1)] for um in range(Inputs)]
	Final=[[sum([I[um][i][j] for um in range(Inputs)])+c2 for j in range(x2)] for i in range(x1)]
	return Final


def maxpool(X,poolsize):
	Row,Col=len(X),len(X[0])
	poolX,poolY=poolsize[0],poolsize[1]
	T=[[max(X[poolX*i][poolY*j],X[poolX*i][(poolY*(j+1))-1],X[(poolX*(i+1))-1][poolY*j],X[(poolX*(i+1))-1][(poolY*(j+1))-1]) for j in range(int(Col/poolX))] for i in range(int(Row/poolY))]
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

def convolutionsTop(In,convw,convb,poolsize):

	l5=int(len(convw))
	XXX,x2,y1=[],[],[]
	for i in range(l5):
		XXX.append(convolutions(In,convw[i],convb[i]))
		x2.append(maxpool(XXX[i],poolsize))
		y1.append(ReLu(x2[i]))
	return XXX,x2,y1


def PadIt(In,Ker):
	
	X,Y=int((len(Ker)-1)/2),int((len(Ker[0])-1)/2)
	return([[[(In[i][j-X][k-Y] if((j>=X and j<len(In[0])+X) and (k>=Y and k<len(In[0][0])+Y)) else 0) for k in range(len(In[0][0])+2*Y)] for j in range(len(In[0])+2*X)] for i in range(len(In))])
	
	

		
def TopInference(Input,start,convw,convb,lw,lb,poolsize,RRAMRes,PulseRes):

	#XXX: outputs of conv layers before Maxpool and ReLU, 0-> len(convw)-1. For n layer, XXX should be n
	#x2: conv layer outputs after Maxpool, before ReLU, 0-> len(convw)-1. For n layer, x2 should be n
	#y1: conv layer outputs after ReLU, consists of the input layer as well. For n layer, y1 should be n+1

	XXX,x2,y1=[],[],[]

	y1.append(Input[start])

	for lm in range(len(convw)):
		L1=PadIt(y1[lm],convw[lm][0][0])
		X1,X2,Y1=convolutionsTop(L1,convw[lm],convb[lm],poolsize[lm])
		#print(len(X1),len(X1[0]),len(X1[0][0]))
		#print(len(Y1),len(Y1[0]),len(Y1[0][0]))
		XXX.append(X1)
		x2.append(X2)
		y1.append(Y1)


	#print(len(y1),len(y1[0]),len(y1[0][0]))

	#o1: outputs of FCC layers before ReLU, 0-> len(lw)-1. For n layer, o1 should be n
	#a1: FCC layer outputs after ReLU, consists of the input layer as well. For n layer, a1 should be n+1
		

	o1,a1=[],[]
	a1.append(CNFC(y1[len(convw)]))

	#print(len(a1[0]), len(lw), len(lb), len(lw[0]), len(lw[0][0]))

	#print(lb[0])
	
	for lm in range(len(lw)-1):
		O1=[sum([lw[lm][i][j]*a1[lm][j] for j in range(len(a1[lm]))])+lb[lm][i] for i in range(len(lw[lm]))]
		A1=[max(0,i) for i in O1]
		o1.append(O1)
		a1.append(A1)

	X=[sum([lw[len(lw)-1][i][j]*a1[len(lw)-1][j] for j in range(len(a1[len(lw)-1]))])+lb[len(lw)-1][i] for i in range(len(lw[len(lw)-1]))]

	o1.append(X)
	a4=[math.exp(i) for i in o1[len(lw)-1]]
	sumT=sum(a4)
	a3=[float(i)/sumT for i in a4]
	a1.append(a3)
	return XXX,x2,y1,o1,a1	

