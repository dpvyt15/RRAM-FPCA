
import math
import sys
sys.path.append('../')

import random
import numpy as np
import time
import cmath
import joblib
from joblib import Parallel, delayed
import VMMNew.ForNNTop
import ConvolutionCheck.ForNNTop

def transpose(A):
	B=[[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
	return B
	

def convolutionX(Input,Weight):
	x1=len(Input)-len(Weight)+1
	x2=len(Input[0])-len(Weight[0])+1
	#print(x1,x2)
	I=[[sum(sum([[Input[i+a][j+b]*Weight[a][b] for b in range(len(Weight[0]))] for a in range(len(Weight))],[])) for j in range(x2)] for i in range(x1)]
	return I

def convolution(Input,Weight,RRAMRes,PulseRes,Split,jobs):
	TBPW=[[[[i for i in j] for j in Weight] for k in range(1)] for l in range(1)]
	TBPI=[[[i for i in j] for j in Input] for k in range(1)]
	I=Convolution.ForNNTop.Top(TBPW,TBPI,RRAMRes,PulseRes,Split,jobs)
	IX=[[I[0][i][j] for j in range(len(I[0][0]))] for i in range(len(I[0]))]	
	return IX

def sigmoidprime(A):
	B=[i*(1-i) for i in A]
	return B

def mult_Back(w,d,o,b,RRAMRes,PulseRes,Split,jobs):
	Y=transpose(w)
	temp=[(1 if i>0 else (0.5 if i==0 else 0)) for i in o]
	X1a=VMM.ForNNTop.Top(Y,d,RRAMRes,PulseRes,Split,jobs)
	delta_new=[X1a[i]*temp[i] for i in range(len(X1a))]
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


def PadIt(In,ker):
	x,y=len(ker),len(ker[0])
	x1,y1=len(In),len(In[0])

	Out=[[0 for i in range(y1+2*y)] for j in range(x1+2*x)]

	for i in range(x,x1+x):
		for j in range(y, y1+y):
			Out[i][j]=In[i-x][j-y]
	return Out	

def partial(TrainIn,Mod,x1,x2,conv1w,uxx,u1,u2,u3,u4,u5,u6,RRAMRes,PulseRes,Split,jobs):

	#print("In the trouble")
	#print(u5,u6,len(Mod),len(Mod[0]),len(x1),len(x1[0]),len(x2),len(x2[0]))

	Final=[[0 for i in range(u6*u2)] for j in range(u5*u1)]

	#print(u6*u2,u5*u1)

	C1w=[[[0 for i in range(u4)] for j in range(u3)] for k in range(uxx)]
	#FinalDeltaL=[[[0 for i in j] for j in k] for k in TrainIn]
	
	for amx in range(uxx):
		for i in range(u1):
			for j in range(u2):
				for o in range(u5):
					for p in range(u6):
						if(x1[(u5*i)+o][(u6*j)+p]==x2[i][j]): 
							#print(i,j,u5*i+o,u6*j+p)
							Final[(u5*i)+o][(u6*j)+p]=Mod[i][j]*ReLuP(x2[i][j])
		C1w[amx]=convolution(TrainIn[amx],Final,RRAMRes,PulseRes,Split,jobs)

		#Modconv1w=ModifyThis(conv1w[amx])
		#Padded=PadIt(Final,Modconv1w)
		#Delta=convolution(Padded,Modconv1w)

		#FinalDeltaL[amx]=[[Delta[len(Delta)-1-i][len(Delta[0])-1-j] for j in range(len(Delta[0]))] for i in range(len(Delta))]

	return C1w	


def partial2(TrainIn,Mod,x1,x2,conv1w,uxx,u1,u2,u3,u4,u5,u6,RRAMRes,PulseRes,Split,jobs):

	Final=[[0 for i in range(u6*u2)] for j in range(u5*u1)]

	#C1w=[[[0 for i in range(u4)] for j in range(u3)] for k in range(uxx)]
	FinalDeltaL=[[[0 for i in j] for j in k] for k in TrainIn]
	
	for amx in range(uxx):
		for i in range(u1):
			for j in range(u2):
				for o in range(u5):
					for p in range(u6):
						if(x1[(u5*i)+o][(u6*j)+p]==x2[i][j]): 
							Final[(u5*i)+o][(u6*j)+p]=Mod[i][j]*ReLuP(x2[i][j])
		#C1w[amx]=convolution(TrainIn[amx],Final)

		Modconv1w=ModifyThis(conv1w[amx])
		Padded=PadIt(Final,Modconv1w)
		Delta=convolution(Padded,Modconv1w,RRAMRes,PulseRes,Split,jobs)

		FinalDeltaL[amx]=[[Delta[len(Delta)-1-i][len(Delta[0])-1-j] for j in range(len(Delta[0]))] for i in range(len(Delta))]

	return FinalDeltaL					
	

def ConvBackProp(TrainIn,x2,x1,Mod,conv1w,RRAMRes,PulseRes,Split,jobs):
	
	#TrainIn=y1 : previous layers output after relu and maxpool; x2=x21: current layers output after maxpool, before flattening or relu; 
	#x1=x11: current layers output before maxpool, flattening and relu
	#progression: TrainIn -> x11 -> x21
	#Mod at the delta of current layer; it is 3d matrix, with current layer output dimensions similar to x21

	poolsize=2;
	u1=len(x2[0]); 
	u2=len(x2[0][0]); 
	u3=len(conv1w[0][0]);   
	u4=len(conv1w[0][0][0]);   
	u7=len(conv1w);   
	u5=poolsize; 
	u6=poolsize; 
	uxx=len(conv1w[0])
	
	C1b=[sum(map(sum,Mod[i])) for i in range(u7)]
	C1w=Parallel(n_jobs=-1)(delayed(partial)(TrainIn,Mod[x],x1[x],x2[x],conv1w[x],uxx,u1,u2,u3,u4,u5,u6,RRAMRes,PulseRes,Split,jobs) for x in range(u7))
	FinalDeltaL=Parallel(n_jobs=-1)(delayed(partial2)(TrainIn,Mod[x],x1[x],x2[x],conv1w[x],uxx,u1,u2,u3,u4,u5,u6,RRAMRes,PulseRes,Split,jobs) for x in range(u7))

	FinalDelta=[[[sum([FinalDeltaL[x][amx][am][pm] for x in range(u7)]) for pm in range(len(TrainIn[0][0]))] for am in range(len(TrainIn[0]))] for amx in range(uxx)]
	return C1w,C1b,FinalDelta	


def Modify(A,u7,u1,u2):
	
	B=[[[0 for i in range(u2)] for j in range(u1)] for k in range(u7)]
	for i in range(len(A)):
		t1=int(i/(u1*u2))
		a1=int((i-(t1*u1*u2))/u2)
		b1=int(i-(t1*u1*u2)-(a1*u2))
		B[t1][a1][b1]=A[i]
	return(B)

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

def TopBackPropNew(a1,o1,lw,y1,x2,XXX,convw,TrainIn,TrainOut,RRAMRes,PulseRes,Split,jobs):

	#t1- after relu and flattening, y11- after flattening, x21- after maxpool, before flattening; x11- after conv, before maxpool
	#progression is: x11 -> x21 -> y11 -> t1

	DN=[]
	LwN=[]

	D3=a1[len(a1)-1]; 
	D3[TrainOut]=D3[TrainOut]-1;
	L3=mult2(D3,a1[len(lw)-1])

	DN.append(D3)
	LwN.append(L3)
	

	for lm in range(len(lw)-1,0,-1):
		D3=mult_Back(lw[lm],DN[len(lw)-1-lm],o1[lm-1],a1[lm],RRAMRes,PulseRes,Split,jobs)
		L3W=mult2(D3,a1[lm-1])
		DN.append(D3)
		LwN.append(L3W)
	
	Dc=mult_Back(lw[0],DN[len(lw)-1],CNFC(x2[len(x2)-1]),a1[0],RRAMRes,PulseRes,Split,jobs)

	u1=len(x2[len(x2)-1][0])
	u2=len(x2[len(x2)-1][0][0])
	u7=len(convw[len(convw)-1])
	Mod=Modify(Dc,u7,u1,u2)

	CbN,CwN=[],[]

	for lm in range(len(convw)-1,-1,-1):
		C2w,C2b,Mod=ConvBackProp(y1[lm],x2[lm],XXX[lm],Mod,convw[lm],RRAMRes,PulseRes,Split,jobs)
		CwN.append(C2w)
		CbN.append(C2b)

	D,Lw,Cb,Cw=[],[],[],[]
	for i in range(len(DN)-1,-1,-1):
		D.append(DN[i])
		Lw.append(LwN[i])



	for i in range(len(CbN)-1,-1,-1):
		Cb.append(CbN[i])
		Cw.append(CwN[i])

	#print(Cw)
	#print(Cb)
	

	return Cw,Cb,Lw,D