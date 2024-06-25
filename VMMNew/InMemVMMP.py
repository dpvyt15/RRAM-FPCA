
import math
import sys
sys.path.append('../')

import random
import numpy as np
import time
import cmath
import joblib
import VMMNew.GeneralFunctions
from scipy import signal


def ReRAM_New_tbTUpMultiResSplitNN(ReadA,Weights,Div,jobs):
	dt,len3=12E-10,len(Weights)
	N2,M=int(3.12E-8/dt),float(dt/2E-11)
	X2=2E-6*(math.exp(3.7*0.8)-1)
	X1=6E-6*math.sinh(4.8*0.8)
	weights_pow10 = np.power(Weights, 10)
	Currents=list(np.einsum('ij,j->i', (weights_pow10 *X1)+X2, ReadA)*N2/2*M)
	return Currents


def TopVMMMatMultiResSplit(Weights,ReadA,Div,jobs):
	Splits=len(Weights)
	lenR=int(len(Weights[0])/2)
	lenC=len(Weights[0][0])
	DigitalVoltOut=[[0 for i in range(lenR)] for j in range(Splits)]
	WeightX=[[[ (Weights[tum][ii][jx] if (ii<lenR and jx<lenC) else Weights[tum][lenR+ii][jx-lenC]) for jx in range(2*lenC)] for ii in range(lenR)] for tum in range(Splits)]
	for tum in range(Splits):
		DigitalVoltOut[tum]=ReRAM_New_tbTUpMultiResSplitNN(ReadA,WeightX[tum],Div,jobs)

	M=(pow(2,16)-1)/10
	DigitalVoltOut=[[i*M for i in j] for j in DigitalVoltOut]
	DigitalVoltOutNew=[sum([DigitalVoltOut[j][i] for j in range(Splits)]) for i in range(lenR)]
	tux=0
	if(Splits>2):
		#tux=sum([(DigitalVoltOut[hx][lenR-1] if hx>=2 else 0) for hx in range(Splits)])
		for hx in range(2,Splits):
			tux=DigitalVoltOut[hx][lenR-1]+tux
		DigitalVoltOutNew[lenR-1]=DigitalVoltOutNew[lenR-1]-tux
	return DigitalVoltOutNew


def VMMMultiResSplit(Weights,Del1,Del12,B,R1,R2,minA,jobs):
	if(minA!=0):
		SignA=minA/abs(minA)
	else:
		SignA=0

	STA=1
	Last=(10/(pow(2,16)-1));
	time=15.6E-9;Cap=2E-11;
	Div1=pow(2,R2)-1;
	Div2=pow(2,R1)
	if(R1==3):
		c=7.58E-5; m=(64/Div2)*1.1222E-6
	if(R1==4):
		c=7.54864E-5; m=4.50827E-6;
	if(R1==5):
		c=7.594E-5; m=2.249E-6;
	if(R1==6):
		c=7.598E-5; m=1.1222E-6;
	b=float(c*time)/Cap; a=float(m*time)/Cap; 
	LenB=len(B);
	maxB,minB=max(B),min(B) 
	Del2,STX=float((maxB-minB)/Div1),1
	B1=B
	if(Del2==0):
		B1=B;
		SignB=0;
		if(maxB!=0):
			Del2=float(maxB)/Div1;
		else:
			Del2=1;
    
	B1x=list(np.floor(np.array(B1)/Del2).astype(int))
	B1y=list(np.ceil(np.array(B1)/Del2).astype(int))
	B1F=B1x+B1y
	Splits=len(Weights)
	DigitalVoltOut1= TopVMMMatMultiResSplit(Weights,B1F,Div1,jobs)

	Btx=sum(B1F)*b


	C1x=[0 for i in range(len(DigitalVoltOut1))]

	Last1=len(C1x)

	for j in range(Last1-1):
		C1x[j]=float(((DigitalVoltOut1[j]*Last)-(Splits*Btx))/a)*Del1*Del2*0.5;

	if(Splits>1):
		C1x[Last1-1]=float(((DigitalVoltOut1[Last1-1]*Last)-(2*Btx))/a)*(Del2*Del12*0.5);


	else:
		C1x[Last1-1]=float(((DigitalVoltOut1[Last1-1]*Last)-(Splits*Btx))/a)*(Del2*Del1*0.5);

	if(STX==1):
		if(STA==1):
			C1=[C1x[i]+(SignA*C1x[Last1-1]) for i in range(Last1-1)]
		if(STA==0):
			C1=[C1x[i]+(minA*C1x[Last1-1]) for i in range(Last1-1)]
	else:
		if(STA==1):
			C1=[C1x[i]+(SignA*C1x[Last1-1]) for i in range(Last1-1)]
		if(STA==0):
			C1=[C1x[i]+(minA*C1x[Last1-1]) for i in range(Last1-1)]
	return C1