
import math
import sys
import random
import numpy as np
import time
import cmath
import joblib
from joblib import Parallel, delayed
from numpy import array, exp

def pad_with(vector, pad_width, iaxis, kwargs):
	pad_value = kwargs.get('padder', 10)
	vector[:pad_width[0]] = pad_value
	vector[-pad_width[1]:] = pad_value

def transpose(A):
	B=[[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
	return B
	

def convolution(Input,Weight):
	x1=len(Input)-len(Weight)+1
	x2=len(Input[0])-len(Weight[0])+1
	#print(x1,x2)
	I=[[sum(sum([[Input[i+a][j+b]*Weight[a][b] for b in range(len(Weight[0]))] for a in range(len(Weight))],[])) for j in range(x2)] for i in range(x1)]
	return I

def sigmoidprime(A):
	C=array(A)
	B=exp(-C)/((1+exp(-C))*(1+exp(-C)))
	return B

def sigmoidP(A):
	B=exp(-A)/((1+exp(-A))*(1+exp(-A)))
	return B

def mult_Back(w,d,o,b):
	Y=transpose(w)
	temp=[(1 if i>0 else (0.5 if i==0 else 0)) for i in o]
	#temp=sigmoidprime(o)
	delta_new=[sum([Y[i][j]*d[j] for j in range(len(d))])*temp[i] for i in range(len(Y))]
	return delta_new

def mult2(in1,in2):
	u=[[(in1[i]*in2[j]) for j in range(len(in2))] for i in range(len(in1))]
	return u

def ReLuP(s):
	b=(1 if s>0 else (0.5 if s==0 else 0))
	return b
									

def ModifyThis(c1w):
	X=[[c1w[j][i] for j in range(len(c1w[0]))] for i in range(len(c1w))]
	Y=[[X[len(X)-1-i][len(X[0])-1-j] for j in range(len(X[0]))] for i in range(len(X))]
	return(Y)


def PadIt(In,Ker):
	X,Y=int((len(Ker)-1)/2),int((len(Ker[0])-1)/2)
	return([[(In[j-X][k-Y] if((j>=X and j<len(In)+X) and (k>=Y and k<len(In[0])+Y)) else 0) for k in range(len(In[0])+2*Y)] for j in range(len(In)+2*X)])
		

def partial(TrainIn,Mod,x1,x2,conv1w,uxx,u1,u2,u3,u4,u5,u6):

	Final=[[0 for i in range(u6*u2)] for j in range(u5*u1)]

	C1w=[[[0 for i in range(u4)] for j in range(u3)] for k in range(uxx)]
	FinalDeltaL=[[[0 for i in j] for j in k] for k in TrainIn]
	
	for amx in range(uxx):
		for i in range(u1):
			for j in range(u2):
				for o in range(u5):
					for p in range(u6):
						if(x1[(u5*i)+o][(u6*j)+p]==x2[i][j]): 
							Final[(u5*i)+o][(u6*j)+p]=Mod[i][j]*ReLuP(x2[i][j])

		L1=PadIt(TrainIn[amx],conv1w[amx])
		C1w[amx]=convolution(L1,Final)
		Modconv1w=ModifyThis(conv1w[amx])
		Padded=PadIt(Final,Modconv1w)
		Delta=convolution(Padded,Modconv1w)
		FinalDeltaL[amx]=[[Delta[len(Delta)-1-i][len(Delta[0])-1-j] for j in range(len(Delta[0]))] for i in range(len(Delta))]

	return C1w,FinalDeltaL		

def ConvBackProp(TrainIn,x2,x1,Mod,conv1w,poolsize):
	
	#TrainIn=y1 : previous layers output after relu and maxpool; x2=x21: current layers output after maxpool, before flattening or relu; 
	#x1=x11: current layers output before maxpool, flattening and relu
	#progression: TrainIn -> x11 -> x21
	#Mod at the delta of current layer; it is 3d matrix, with current layer output dimensions similar to x21

	#poolsize=2;
	u1,u2,u3,u4,u7,u5,u6,uxx=len(x2[0]),len(x2[0][0]),len(conv1w[0][0]),len(conv1w[0][0][0]),len(conv1w),poolsize[0],poolsize[1],len(conv1w[0])
	
	C1b=[sum(map(sum,Mod[i])) for i in range(u7)]
	C1w,FinalDeltaL=[],[]
	for x in range(u7):
		C1,F1=partial(TrainIn,Mod[x],x1[x],x2[x],conv1w[x],uxx,u1,u2,u3,u4,u5,u6)
		C1w.append(C1)
		FinalDeltaL.append(F1)
	FinalDelta=[[[sum([FinalDeltaL[x][amx][am][pm] for x in range(u7)]) for pm in range(len(TrainIn[0][0]))] for am in range(len(TrainIn[0]))] for amx in range(uxx)]
	return C1w,C1b,FinalDelta	


def Modify(A,u7,u1,u2):
	B=np.reshape(A,(u7,u1,u2))
	return (B.astype(float)).tolist()

def CNFC(input):
	C=np.ravel(input)
	return C.tolist()

def TopBackPropNew(a1,o1,lw,y1,x2,XXX,convw,TrainIn,TrainOut,poolsize):

	l3=len(TrainIn)

	#t1- after relu and flattening, y11- after flattening, x21- after maxpool, before flattening; x11- after conv, before maxpool
	#progression is: x11 -> x21 -> y11 -> t1

	DN,LwN={},{}
	Mod=[]

	for i in range(l3):

		DN[i],LwN[i]=[],[]
		D3=a1[i][len(a1[i])-1]; 
		D3[TrainOut[i]]=D3[TrainOut[i]]-1;
		L3=mult2(D3,a1[i][len(lw)-1])

		DN[i].append(D3)
		LwN[i].append(L3)

		for lm in range(len(lw)-1,0,-1):
			D3=mult_Back(lw[lm],DN[i][len(lw)-1-lm],o1[i][lm-1],a1[i][lm])
			L3W=mult2(D3,a1[i][lm-1])
			DN[i].append(D3)
			LwN[i].append(L3W)

		Dc=mult_Back(lw[0],DN[i][len(lw)-1],CNFC(x2[i][len(x2[i])-1]),a1[i][0])

		u1,u2,u7=len(x2[i][len(x2[i])-1][0]),len(x2[i][len(x2[i])-1][0][0]),len(convw[len(convw)-1])
		Mod.append(Modify(Dc,u7,u1,u2))

	PK,UK=[],[]
	for k in range(len(lw)):
		P=DN[0][k]
		U=LwN[0][k]
		for i in range(1,len(DN),1):
			P=[P[u]+DN[i][k][u] for u in range(len(P))]
			U=[[U[u][j]+LwN[i][k][u][j] for j in range(len(U[u]))] for u in range(len(U))]

		PK.append(P)
		UK.append(U)

	CbN,CwN={},{}
	for i in range(l3):
		CbN[i]=[]
		CwN[i]=[]

	for lm in range(len(convw)-1,-1,-1):
		for i in range(l3):
			C2w,C2b,Mod[i]=ConvBackProp(y1[i][lm],x2[i][lm],XXX[i][lm],Mod[i],convw[lm],poolsize[lm])
			CwN[i].append(C2w)
			CbN[i].append(C2b)


	CK,WK=[],[]
	for uk in range(len(convw)):
		C=CbN[0][uk]
		W=CwN[0][uk]
		for um in range(1,len(DN),1):
			W=[[[[W[j][k][l][m]+CwN[um][uk][j][k][l][m] for m in range(len(W[j][k][l]))] for l in range(len(W[j][k]))] for k in range(len(W[j]))] for j in range(len(W))] 
			C=[C[j]+CbN[um][uk][j] for j in range(len(C))]

		CK.append(C)
		WK.append(W)

	D,Lw,Cb,Cw=[],[],[],[]

	for i in range(len(PK)-1,-1,-1):
		D.append(PK[i])
		Lw.append(UK[i])

	for i in range(len(CK)-1,-1,-1):
		Cb.append(CK[i])
		Cw.append(WK[i])

	return Cw,Cb,Lw,D