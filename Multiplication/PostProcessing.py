
import math
import sys
import random


"""Performs postprocessing of the crossbar output data. It bins the partial products in words and performs the final addition of the intermediate words to determine the output."""


def PostProcess(In,BW):

	IntArr=[[[0 for i in range(15)] for j in range(int(len(In[0])*(180/20)))] for k in range(len(In))]
	for i in range(len(In)):
		for j in range(len(In[0])):
			for k in range(len(In[0][0])):
				am=k%20
				umm=int(k/20)
				if(am<15 and am>=10):
					IntArr[i][j*9+umm][14-am]=In[i][j][k]
				if(am<10):
					IntArr[i][j*9+umm][am+5]=In[i][j][k]

	l1=int(BW/8)
	l2=l1%10
	la=l1+min(l2,10)-1

	if(l1<10):
		lx=la
	if(l1>10):
		lx=l1+9
	
	Out=[[0 for i in range(15+(lx-1)*8)] for j in range(len(In))]
	for i in range(len(In)):
		lim=((i==len(In)-1)*la)+((i!=len(In)-1)*lx)
		##print(lim)
		for j in range(lim):
			if(j==0):
				Out[i][0:14]=IntArr[i][j]
			if(j>0):
				for k in range(15):
					Out[i][8*j+k]=IntArr[i][j][k]+Out[i][8*j+k]

	OutArr=[0 for i in range(2*BW)]
	for i in range(len(Out)):
		for k in range(len(Out[0])):
			if((80*i+k)<2*BW):
				OutArr[80*i+k]=Out[i][k]+OutArr[80*i+k]

	return OutArr


def Finalprocessing(Out,BW):
	BPD=int(math.log(BW,2)+1)
	ND=math.ceil((2*BW)/BPD)


	L1=[[0 for i in range((ND+1)*BPD)] for j in range(BPD)]
	for i in range(BPD):
		j=0
		while((j*BPD+i)<len(Out)):
			k1=Out[j*BPD+i]
			lax=0
			while(k1>0):
				L1[i][j*BPD+i+lax]=int(k1%2)
				k1=int(k1/2)
				lax=lax+1
			j=j+1

	L1x=[0 for i in range(BPD)]

	for i in range(BPD):
		for j in range(2*BW):
			L1x[i]=L1x[i]+(math.pow(2,j-(2*BW-1))*L1[i][j])


	NewOutputs=(sum(L1x))

	return NewOutputs


def PrePostProcess(In,Index,Index21,Index31,BW):
	l1a=int(BW/8)
	l2a=(l1a)%10
	lims=int((l1a+l2a-1)%9)
	ax=len(In)-1
	for i in range(len(In)):
		Z=Index[i][0]
		t1=Index[i][2]
		t21=Index21[i][2]
		t31=Index31[i][2]
		for j in range(20):
			In[i][0][j]=int(In[i][0][j]/8)
			In[i][0][20+j]=int(In[i][0][20+j]/4)

			In[i][Z][(t1*20)+j]=int(In[i][Z][(t1*20)+j]/8)	

			if(BW!=16):
				In[i][Z][(t21*20)+j]=int(In[i][Z][(t21*20)+j]/4)

				if(i==len(In)-1 and l2a>=6):
					In[i][0][40+j]=int(In[i][0][40+j]/2)
					In[i][Z][(t31*20)+j]=int(In[i][Z][(t31*20)+j]/2)

				if(i<len(In)-1):
					In[i][0][40+j]=int(In[i][0][40+j]/2)
					In[i][Z][(t31*20)+j]=int(In[i][Z][(t31*20)+j]/2)

	if(l2a<6):
		Z=Index[ax][0]
		pa=((l2a==2)*4)+((l2a==4)*2)
		Y21=Index21[ax][2]
		for i in range(len(In[0])):	

			if(i!=0 and i!=Z):
				for j in range(len(In[0][0])):
					In[ax][i][j]=int(In[ax][i][j]/pa)
			if(i==0 and i!=Z):
				for j in range(40,len(In[0][0])):
					In[ax][i][j]=int(In[ax][i][j]/pa)
			if(i!=0 and i==Z):
				for j in range(20*Y21):
					In[ax][i][j]=int(In[ax][i][j]/pa)
			if(i==0 and i==Z):
				for j in range(40,20*Y21,1):
					In[ax][i][j]=int(In[ax][i][j]/pa)

	return In

