
import math
import sys
sys.path.append('../')
import random
import numpy as np
import time
import cmath
from joblib import Parallel, delayed
from copy import deepcopy
import ConvolutionCheck.ConvInMemP
import VMMNew.InMemVMMP

def batchNorm(In,layer,gamma,beta,L,RunMean,RunVar):
	delta=1E-5
	momentum=0.9
	batches,channels,width,length=len(In),len(In[0][layer]),len(In[0][layer][0]),len(In[0][layer][0][0]) 
	BNOut=[[[[0 for i in range(length)] for j in range(width)] for k in range(channels)] for l in range(batches)]

	mean,variance,runmean,runvar=[0 for i in range(channels)],[0 for i in range(channels)],[0 for i in range(channels)],[0 for i in range(channels)]
	for i in range(channels):
		if(L==1):
			mean[i]=sum([sum([sum([In[um][layer][i][j][k] for k in range(length)]) for j in range(width)]) for um in range(batches)])*(1/(batches*width*length))
			variance[i]=sum([sum([sum([(In[um][layer][i][j][k]-mean[i])**2 for k in range(length)]) for j in range(width)]) for um in range(batches)])*(1/(batches*width*length))
			runmean[i]=(momentum*RunMean[i])+((1-momentum)*mean[i])
			runvar[i]=(momentum*RunVar[i])+((1-momentum)*variance[i])
		if(L==0):
			mean[i],variance[i],runmean[i],runvar[i]=RunMean[i],RunVar[i],RunMean[i],RunVar[i]

		A=[[[gamma[i]*((In[um][layer][i][j][k]-mean[i])/math.sqrt(variance[i]+delta))+beta[i] for k in range(length)] for j in range(width)] for um in range(batches)]
		for um in range(batches):
			BNOut[um][i]=A[um]

	return(BNOut,mean,variance,runmean,runvar)

def maxpool(X,poolsize):
	Row,Col,poolX,poolY=len(X[0]),len(X[0][0]),poolsize[0],poolsize[1]
	return [[[max(X[k][poolX*i][poolY*j],X[k][poolX*i][(poolY*(j+1))-1],X[k][(poolX*(i+1))-1][poolY*j],X[k][(poolX*(i+1))-1][(poolY*(j+1))-1]) for j in range(int(Col/poolX))] for i in range(int(Row/poolY))] for k in range(len(X))]

def sigmoid(A):
	return [float(1/(1+math.exp(-1*i))) for i in A]

def CNFC(input):
	C=np.ravel(input)
	return C.tolist()

def ReLu(input):
	return [[[max(i,0) for i in j] for j in k] for k in input]

def convolutionsTop(In,convb):

	l5,l1,l2=int(len(convb)),len(In[0]),len(In[0][0])
	return [[[In[k][i][j]+convb[k] for j in range(l2)] for i in range(l1)] for k in range(l5)]


def PadIt(In,Ker):
	
	X,Y=int((len(Ker)-1)/2),int((len(Ker[0])-1)/2)
	return([[[(In[i][j-X][k-Y] if((j>=X and j<len(In[0])+X) and (k>=Y and k<len(In[0][0])+Y)) else 0) for k in range(len(In[0][0])+2*Y)] for j in range(len(In[0])+2*X)] for i in range(len(In))])
	
		
def TopInference(Input,convw,cW,cW2,dcW,dcW2,mcW,convb,lW,lW2,dlW,dlW2,mlW,lb,poolsize,RRAMRes,PulseRes,DL,gamma,beta,L,RunMean,RunVar):

	#XXX: outputs of conv layers before Maxpool and ReLU, 0-> len(convw)-1. For n layer, XXX should be n
	#x2: conv layer outputs after Maxpool, before ReLU, 0-> len(convw)-1. For n layer, x2 should be n
	#y1: conv layer outputs after ReLU, consists of the input layer as well. For n layer, y1 should be n+1


	DLlayer=0
	DLFinal=DL[0]
	XXX,x2,x21,y1,g1={},{},{},{},{}

	l3=len(Input)

	for k in range(l3):
		XXX[k],x2[k],x21[k],y1[k],g1[k]=[],[],[],[],[]
		x2[k].append(Input[k])
		
	mean,variance=[],[]

	for lm in range(len(cW)):
		if(DL[lm]==DLFinal):
			for j in range(l3):
				for k in range(lm-1,DLlayer-1,-1):
					ma=len(x2[j][k]) if k==DLlayer else 12
					for um in range(ma):
						x2[j][lm].append(x2[j][k][um])
		else:
			DLFinal=DL[lm]
			DLlayer=lm

		y1a,mean1,var1,RunMean[lm],RunVar[lm]=batchNorm(x2,lm,gamma[lm],beta[lm],L,RunMean[lm],RunVar[lm])
		mean.append(mean1); variance.append(var1)
		for j in range(l3):
			y1[j].append(y1a[j])
			g1[j].append(ReLu(y1[j][lm]))
			L1=PadIt(g1[j][lm],convw[lm][0][0])
			P=ConvolutionCheck.ConvInMemP.ConvolutionInMem(cW[lm],dcW[lm],cW2[lm],dcW2[lm],L1,mcW[lm],PulseRes,RRAMRes,-1)
			XXX[j].append(convolutionsTop(P,convb[lm]))
			x2[j].append(maxpool(XXX[j][lm],poolsize[lm]))
			x21[j].append(x2[j][lm])

	#o1: outputs of FCC layers before ReLU, 0-> len(lw)-1. For n layer, o1 should be n
	#a1: FCC layer outputs after ReLU, consists of the input layer as well. For n layer, a1 should be n+1
		

	o1,a1={},{}
	for k in range(l3):
		o1[k]=[]
		a1[k]=[]
		o1[k].append(CNFC(x2[k][len(cW)]))
	
	for k in range(l3):
		for lm in range(len(lW)):
			a1[k].append([max(0,i) for i in o1[k][lm]])
			O1=VMMNew.InMemVMMP.VMMMultiResSplit(lW[lm],dlW[lm],dlW2[lm],a1[k][lm],RRAMRes,PulseRes,mlW[lm],-1)
			o1[k].append([O1[j]+lb[lm][j] for j in range(len(O1))])

		a4=[math.exp(i) for i in o1[k][len(lW)]]
		sumT=sum(a4)
		a3=[float(i)/sumT for i in a4]
		a1[k].append(a3)

	return x2,x21,y1,g1,XXX,o1,a1,mean,variance,RunMean,RunVar	

