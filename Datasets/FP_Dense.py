
import math
import sys
import random
import numpy as np
import time
import cmath
from joblib import Parallel, delayed

def convolutions(Input,Weight,c2):
	Inputs=len(Input)
	x1,x2=len(Input[0])-len(Weight[0])+1,len(Input[0][0])-len(Weight[0][0])+1
	I=[[[sum(sum([[Input[um][i+a][j+b]*Weight[um][a][b] for b in range(len(Weight[0][0]))] for a in range(len(Weight[0]))],[])) for j in range(x2)] for i in range(x1)] for um in range(Inputs)]
	Final=[[sum([I[um][i][j] for um in range(Inputs)])+c2 for j in range(x2)] for i in range(x1)]
	return Final


def maxpool(X,poolsize):
	Row,Col=len(X),len(X[0])
	poolX,poolY=poolsize[0],poolsize[1]
	T=[[max(X[poolX*i][poolY*j],X[poolX*i][(poolY*(j+1))-1],X[(poolX*(i+1))-1][poolY*j],X[(poolX*(i+1))-1][(poolY*(j+1))-1]) for j in range(int(Col/poolX))] for i in range(int(Row/poolY))]
	return T

def sigmoid(A):
	return [float(1/(1+math.exp(-1*i))) for i in A]

def CNFC(input):
	C=np.ravel(input)
	return C.tolist()

def ReLu(input):
	return [[max(i,0) for i in j] for j in input]

def convolutionsTop(In,layer,convw,convb,poolsize):

	l3=int(len(In))
	l5=int(len(convw))
	XXX,x2,y1=[],[],[]
	for j in range(l3):
		for i in range(l5):
			XXX.append(convolutions(In[j][layer],convw[i],convb[i]))
			x2.append(maxpool(XXX[i],poolsize))
			y1.append(ReLu(x2[i]))

		In[j].append(XXX)
		x2[j].append(x2)
		y1[j].append(y1)
	return XXX,x2,y1


def PadIt(In,Ker):
	
	X,Y=int((len(Ker)-1)/2),int((len(Ker[0])-1)/2)
	return([[[(In[i][j-X][k-Y] if((j>=X and j<len(In[0])+X) and (k>=Y and k<len(In[0][0])+Y)) else 0) for k in range(len(In[0][0])+2*Y)] for j in range(len(In[0])+2*X)] for i in range(len(In))])
	
		
def TopInference(Input,convw,convb,lw,lb,poolsize,DL):

	#XXX: outputs of conv layers before Maxpool and ReLU, 0-> len(convw)-1. For n layer, XXX should be n
	#x2: conv layer outputs after Maxpool, before ReLU, 0-> len(convw)-1. For n layer, x2 should be n
	#y1: conv layer outputs after ReLU, consists of the input layer as well. For n layer, y1 should be n+1

	XXX,x2,y1={},{},{}

	l3=len(Input)

	for k in range(l3):
		XXX[k]=[]
		x2[k]=[]
		y1[k]=[]
		y1[k].append(Input[k])
		
	for lm in range(len(convw)):
		for j in range(l3):
			L1=PadIt(y1[j][lm],convw[lm][0][0])
			XXXa,x2a,y1a=[],[],[]
			for i in range(len(convw[lm])):
				XXXa.append(convolutions(L1,convw[lm][i],convb[lm][i]))
				x2a.append(maxpool(XXXa[i],poolsize[lm]))
				y1a.append(ReLu(x2a[i]))

				#X1,X2,Y1=convolutionsTop(L1,convw[lm],convb[lm],poolsize[lm])
			XXX[j].append(XXXa)
			x2[j].append(x2a)
			y1[j].append(y1a)


	#print(len(y1),len(y1[0]),len(y1[0][0]))

	#o1: outputs of FCC layers before ReLU, 0-> len(lw)-1. For n layer, o1 should be n
	#a1: FCC layer outputs after ReLU, consists of the input layer as well. For n layer, a1 should be n+1
		

	o1,a1={},{}
	for k in range(l3):
		o1[k]=[]
		a1[k]=[]
		a1[k].append(CNFC(y1[k][len(convw)]))

	#print(len(a1[0]), len(lw), len(lb), len(lw[0]), len(lw[0][0]))

	#print(lb[0])
	for k in range(l3):
		for lm in range(len(lw)-1):
			o1[k].append([sum([lw[lm][i][j]*a1[k][lm][j] for j in range(len(a1[k][lm]))])+lb[lm][i] for i in range(len(lw[lm]))])
			a1[k].append([max(0,i) for i in o1[k][lm]])

		o1[k].append([sum([lw[len(lw)-1][i][j]*a1[k][len(lw)-1][j] for j in range(len(a1[k][len(lw)-1]))])+lb[len(lw)-1][i] for i in range(len(lw[len(lw)-1]))])

		a4=[math.exp(i) for i in o1[k][len(lw)-1]]
		sumT=sum(a4)
		a3=[float(i)/sumT for i in a4]
		a1[k].append(a3)

	return XXX,x2,y1,o1,a1	

