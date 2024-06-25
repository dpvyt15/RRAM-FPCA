
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


def ReRAM_NewB(V_t,init,ReadA,N2,M):
	Current=sum([((pow(init[i],10)*6E-6*math.sinh(4.8*V_t))+(2E-6*(math.exp(3.7*V_t)-1)))*ReadA[i]*(N2/2) for i in range(len(init))])*M
	return Current


def ReRAM_New_tbTUpMultiResSplitNN(ReadA,Weights,Div,jobs):
	dt,len3=12E-10,len(Weights)
	N2,M=int(3.12E-8/dt),float(dt/2E-11)
	X2=2E-6*(math.exp(3.7*0.8)-1)
	X1=6E-6*math.sinh(4.8*0.8)
	weights_pow10 = np.power(Weights, 10)
	
	Currents=list(np.einsum('ij,j->i', (weights_pow10 *X1)+X2, ReadA)*N2/2*M)

	#Current2=[sum([((pow(Weights[ii][i],10)*X1)+X2)*ReadA[i]*(N2/2) for i in range(len(Weights[ii]))])*M for ii in range(len3)]

	#print(Currents)
	#print(Current2)

	return Currents


def TopVMMMatMultiResSplit(Weights,ReadA,Div,jobs):
	Splits=len(Weights)
	lenR=int(len(Weights[0])/2)
	lenC=len(Weights[0][0])
	DigitalVoltOut=[[0 for i in range(lenR)] for j in range(Splits)]

	WeightX=[[[ (Weights[tum][ii][jx] if (ii<lenR and jx<lenC) else Weights[tum][lenR+ii][jx-lenC]) for jx in range(2*lenC)] for ii in range(lenR)] for tum in range(Splits)]

	#start_time=time.time()

	for tum in range(Splits):
		DigitalVoltOut[tum]=ReRAM_New_tbTUpMultiResSplitNN(ReadA,WeightX[tum],Div,jobs)

	#end_time=time.time()

	#print("Core comp time:")
	#print(end_time-start_time)

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
	#c=7.58E-5; m=(64/Div2)*1.1222E-6;
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
	maxB,minB=VMMNew.GeneralFunctions.sortingSingle(B); 
	Del2,STX=float((maxB-minB)/Div1),1
	UtilB=[0 for i in range(LenB)]
	if(Del2==0):
		B1=B;
		SignB=0;
		if(maxB!=0):
			Del2=maxB;
		else:
			Del2=1;
		UtilB=[0 for i in B]
	else:
		B1=[i-minB for i in B]
		maxB1,minB1=max(B1),min(B1)
		UtilB=[abs(minB) for i in B]
		if (minB!=0):
			SignB=minB/abs(minB)
		else:
			SignB=0
		if(maxB1<abs(minB)):
			Del2=float((abs(minB))/Div1)
			if(maxB1<1):
				Del2,minXB=float(maxB1/Div1),abs(minB)/maxB1
				UtilB=[maxB1 for i in B]
				STX=0

    
	B1x=list(np.floor(np.array(B1)/Del2).astype(int))
	B1y=list(np.ceil(np.array(B1)/Del2).astype(int))
	U2x=list(np.floor(np.array(UtilB)/Del2).astype(int))
	U2y=list(np.ceil(np.array(UtilB)/Del2).astype(int))

	U2F=U2x+U2y
	B1F=B1x+B1y
	Splits=len(Weights)
	DigitalVoltOut1= TopVMMMatMultiResSplit(Weights,B1F,Div1,jobs)
	DigitalVoltOut2= TopVMMMatMultiResSplit(Weights,U2F,Div1,jobs)

	Btx=sum(B1F)*b
	Utx=sum(U2F)*b

	Last1=len(DigitalVoltOut1)

	C1x=[0 for i in range(len(DigitalVoltOut1))]
	C2x=[0 for i in range(len(DigitalVoltOut2))]

	

	for j in range(Last1-1):
		C1x[j]=float(((DigitalVoltOut1[j]*Last)-(Splits*Btx))/a)*Del1*Del2*0.5;
		C2x[j]=float(((DigitalVoltOut2[j]*Last)-(Splits*Utx))/a)*Del1*Del2*0.5;


	if(Splits>1):
		C1x[Last1-1]=float(((DigitalVoltOut1[Last1-1]*Last)-(2*Btx))/a)*(Del2*Del12*0.5);
		C2x[Last1-1]=float(((DigitalVoltOut2[Last1-1]*Last)-(2*Utx))/a)*(Del2*Del12*0.5);


	else:
		C1x[Last1-1]=float(((DigitalVoltOut1[Last1-1]*Last)-(Splits*Btx))/a)*(Del2*Del1*0.5);
		C2x[Last1-1]=float(((DigitalVoltOut2[Last1-1]*Last)-(Splits*Utx))/a)*(Del2*Del1*0.5);		

	if(STX==1):
		if(STA==1):
			C1=[C1x[i]+(SignA*C1x[Last1-1])+(SignB*C2x[i])+(SignA*SignB*C2x[Last1-1]) for i in range(Last1-1)]
		if(STA==0):
			C1=[C1x[i]+(minA*C1x[Last1-1])+(SignB*C2x[i])+(minA*SignB*C2x[Last1-1]) for i in range(Last1-1)]
	else:
		if(STA==1):
			C1=[C1x[i]+(SignA*C1x[Last1-1])+(minXB*SignB*C2x[i])+(SignA*minXB*SignB*C2x[Last1-1]) for i in range(Last1-1)]
		if(STA==0):
			C1=[C1x[i]+(minA*C1x[Last1-1])+(minXB*C2x[i])+(minA*minXB*C2x[Last1-1]) for i in range(Last1-1)]
	return C1