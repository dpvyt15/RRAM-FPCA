

import math
import sys
sys.path.append('../')

import random
import numpy as np
import time
import cmath
#import joblib
import ConvolutionCheck.GeneralFunctions
#from scipy import signal
#from joblib import Parallel, delayed


def ModImages(WeightsC2,B,R1,R2,PX,PY):

	RowB=len(B); ColumnB=len(B[0])
	maxB,minB=ConvolutionCheck.GeneralFunctions.sorting(B)

	Div1=pow(2,R1)-1

	Del2,STX,minXB=float(abs(maxB-minB)/Div1),1,0


	B1=B
	if(Del2==0):
		if(maxB!=0):
			Del2=float(maxB)/Div1;
		else:
			Del2=1;
	
	B1x=[[math.floor(B1[i][j]/Del2) for j in range(len(B[0]))] for i in range(len(B))]
	B1y=[[math.ceil(B1[i][j]/Del2) for j in range(len(B[0]))] for i in range(len(B))]
	Bx=ConvolutionCheck.GeneralFunctions.append_rows(B1x,B1y)

	PulseB=NewConvExecPulse(Bx,PX,PY)
	return Bx,Del2,PulseB


def ConvolutionSingle(WeightsC2,DelC,Bx,Del2,PulseB,Val,R1,R2,XC,YC,X1,X2,RX,RY):

	Div1=pow(2,R1)-1
	Last=float(10/(pow(2,9)-1));
	time=15.6E-9;Cap=2E-11;
	Div2=pow(2,R2)
	if(R1==3):
		c=7.58E-5; m=(64/Div2)*1.1222E-6
	if(R1==4):
		c=7.54864E-5; m=4.50827E-6;
	if(R1==5):
		c=7.594E-5; m=2.249E-6;
	if(R1==6):
		c=7.598E-5; m=1.1222E-6;
	b=float(c*time)/Cap; a=float(m*time)/Cap; 

	Split=len(WeightsC2);    x1=len(WeightsC2[0]); y1=len(WeightsC2[0][0]);
	x2=int(len(Bx)/2);    y2=len(Bx[0]); t1=x2-(math.floor(x1/2))+1;    t2=y2-y1+1; 	
	
	DigitalVoltOutNew=[[[0 for i in range(t2)] for j in range(t1)] for k in range(len(WeightsC2))]

	for i in range(len(WeightsC2)):
		DigitalVoltOutNew[i]=TopConvolution2Mat64Latest(WeightsC2[i],Bx,PulseB,Div1,XC,YC,X1,X2,RX,RY,b);

	if(Val==1):
		COut=[[sum([float((DigitalVoltOutNew[i][j][k])*DelC*Del2*0.5/a) for i in range(Split)]) for k in range(t2)] for j in range(t1)]
	if(Val==0):
		COut=[[sum([float((DigitalVoltOutNew[i][j][k])*Del2*0.5/a) for i in range(Split)]) for k in range(t2)] for j in range(t1)]
	
	return COut


def TopConv(WeightC,DelC,Bx,DelB,PulseB,NegWeightsOut,DelC2,minC,PulseRes,RRAMRes,XC,YC,X1,X2,RX,RY):
	SignC=[abs(i)/i if abs(i)>0 else 0 for i in minC]
	Images=len(Bx); RowB=int(len(Bx[0])/2); ColumnB=len(Bx[0][0]);
	Split=len(WeightC[0]);    x1=len(WeightC[0][0]); y1=len(WeightC[0][0][0]);
	x2=RowB;    y2=ColumnB; t1=x2-(math.floor(x1/2))+1;    t2=y2-y1+1; 
	PosWeightsOut=[[[0 for i in range(t2)] for j in range(t1)] for k in range(Images)]
	for i in range(Images):
		PosWeightsOut[i]=ConvolutionSingle(WeightC[i],DelC[i],Bx[i],DelB[i],PulseB[i],1,PulseRes,RRAMRes,XC,YC,X1,X2,RX,RY)
	OverAllOut=[[sum([PosWeightsOut[i][j][k]+(SignC[i]*NegWeightsOut[i][j][k]*DelC2[i]) for i in range(Images)]) for k in range(t2)] for j in range(t1)]
	return OverAllOut


def ConvolutionInMem(WeightC,DelC,WeightC2,DelC2,B,minC,PulseRes,RRAMRes,jobs):
	NegWeightsOut=[]
	XC,YC=ArrPulseDef(WeightC2[0],B[0])
	RX,RY,PX,PY=ArrDiv(WeightC2[0],B[0])
	X1=6E-6*math.sinh(4.8*0.8)
	X2=(2E-6*(math.exp(3.7*0.8)-1))
	Bx,Ux,PulseB,PulseU=[],[],[],[]
	DelB,SignB,minXB,STX=[],[],[],[]
	for i in range(len(B)):
		Bxa,DelBa,PulseBa=ModImages(WeightC2,B[i],PulseRes,RRAMRes,PX,PY)
		Bx.append(Bxa), PulseB.append(PulseBa), DelB.append(DelBa)

	for i in range(len(B)):
		NegWeightsOut.append(ConvolutionSingle(WeightC2,DelC2,Bx[i],DelB[i],PulseB[i],0,PulseRes,RRAMRes,XC,YC,X1,X2,RX,RY))

	Outputs=[]
	for i in range(len(WeightC)):
		Outputs.append(TopConv(WeightC[i],DelC[i],Bx,DelB,PulseB,NegWeightsOut,DelC2[i],minC[i],PulseRes,RRAMRes,XC,YC,X1,X2,RX,RY))
	return Outputs


def TopConvolution2Mat64Latest(WeightsC2,ConvMatrixX,PulseX,Div,XC,YC,X1,X2,RX,RY,b):

	lenR1,len4,lenR,lenC=int(len(ConvMatrixX)/2),len(ConvMatrixX[0]),int(len(WeightsC2)/2),len(WeightsC2[0])
	WeightX,WeightX2=[[WeightsC2[j][i] for i in range(lenC)] for j in range(lenR)],[[WeightsC2[j+lenR][i] for i in range(lenC)] for j in range(lenR)]
	ConvMatrixX1,ConvMatrixX2=[[ConvMatrixX[j][i] for i in range(len4)] for j in range(lenR1)],[[ConvMatrixX[j+lenR1][i] for i in range(len4)] for j in range(lenR1)]

	RRAM=NewConvExecRRAM(WeightX,WeightX2,RX,RY)

	V1=ConvolutionComp8New(RRAM[0],PulseX[0],RRAM[1],PulseX[1],Div,XC,YC,X1,X2,b)
	#M=float((pow(2,9)-1)/10);
	#DV1=[[[float((i)) for i in j] for j in k] for k in V1]

	Final=ReAdjust(WeightX,ConvMatrixX1,V1)

	return Final

def ReAdjust(W,B,In):
	
	l1,l2=len(B)-len(W)+1,len(B[0])-len(W[0])+1
	BPA,BPSA,TotOutPA=In.shape
	In=In.tolist()

	Out=[[0 for i in range(l2)] for i in range(l1)]

	Xa,Ya=[int(k/10) for k in range(TotOutPA)],[int(k%10) for k in range(TotOutPA)]
	Ba=[j*10 for j in range(BPSA)]
	Bul=[k if Xa[k]%2==0 else (Xa[k]*10)+9-Ya[k] for k in range(TotOutPA)]
	
	l1P=min(180,10*l1)
	l2P=int(l1P/10)

	for i in range(BPA):
		a1=i*l2P
		for j in range(BPSA):
			b1,X=Ba[j],In[i][j]
			for k in range(TotOutPA):
				V1,V2=a1+Xa[k],b1+Ya[k]
				if(V1<l1 and V2<l2):
					Out[V1][V2]=X[Bul[k]]
	return Out


def ArrDiv(W,B):
	RowKer=int(len(W)/2); ColKer=len(W[0])
	TotPArr=18
	RowImg=len(B); ColImg=len(B[0])
	TotPSA=10

	
	BPA=math.ceil(RowImg/TotPArr)
	BPSA=math.ceil(ColImg/TotPSA)

	l1P=min(180,10*(RowImg-RowKer+1))
	l2P=2*int(l1P/10)
	RRAMArrX=[[int(j/9) if int(j/9)<RowKer else -1 for i in range(l1P)] for j in range(9*ColKer)]
	RRAMArrY=[[int(j%9) if int(j%9)<ColKer else -1 for i in range(l1P)] for j in range(9*ColKer)]

	LX=[[[[-1 for i in range(l2P)] for j in range(9*ColKer)] for k in range(BPSA)] for l in range(BPA)]
	LY=[[[[-1 for i in range(l2P)] for j in range(9*ColKer)] for k in range(BPSA)] for l in range(BPA)]


	for ux in range(BPA):
		for um in range(BPSA):
			AC=TotPSA*um
			AR=TotPArr*ux
			for i in range(int(l2P/2)):
				for j in range(9*ColKer):
					l1=int(j/9)
					l2=int(j%9)

					if(l1<RowKer):
						if(i==0 and AR+l1<RowImg and AC+l2<ColImg):
							LX[ux][um][j][2*i]=AR+l1
							LY[ux][um][j][2*i]=AC+l2
							if(l2<ColKer and 9+AC+l2<ColImg):
								LX[ux][um][j][2*i+1]=AR+l1
								LY[ux][um][j][2*i+1]=9+l2+AC

						if(i>0 and i%2==0 and AR+l1+i<RowImg and AC+l2<ColImg):
							if(l1<(RowKer-1) and l2<ColKer and 9+l2+AC<ColImg):
								LX[ux][um][j][2*i+1]=AR+l1+i
								LY[ux][um][j][2*i+1]=9+l2+AC
							if(l1==RowKer-1):
								LX[ux][um][j][2*i]=AR+l1+i
								LY[ux][um][j][2*i]=AC+l2
								if(l2<ColKer and 9+l2+AC<ColImg):
									LX[ux][um][j][2*i+1]=AR+l1+i
									LY[ux][um][j][2*i+1]=9+l2+AC

						if(i%2==1 and AR+l1+i<RowImg and l2+AC<ColImg):
							if(l1<RowKer-1):
								LX[ux][um][j][2*i+1]=l1+i+AR
								LY[ux][um][j][2*i+1]=l2+AC
							if(l1==RowKer-1):
								LX[ux][um][j][2*i+1]=l1+i+AR
								LY[ux][um][j][2*i+1]=l2+AC
								if(l2<ColKer and 9+l2+AC<ColImg):
									LX[ux][um][j][2*i]=l1+i+AR
									LY[ux][um][j][2*i]=9+l2+AC
	
	for ux in range(BPA):
		for um in range(BPSA):
			for i in range(int(l2P/2)):
				for j in range(9*(RowKer-1)):
					if(i>0):
						LX[ux][um][j][2*i]=LX[ux][um][j+9][2*(i-1)+1]
						LY[ux][um][j][2*i]=LY[ux][um][j+9][2*(i-1)+1]

	return np.array(RRAMArrX),np.array(RRAMArrY),np.array(LX),np.array(LY)


def NewConvExecRRAM(W1,W2,RRAMX,RRAMY):
	RRAMArr=[]; WT1=np.array(W1); WT2=np.array(W2);
	K=(RRAMX>=0)*(RRAMY>=0)
	A,B= WT1[RRAMX,RRAMY], WT2[RRAMX,RRAMY]
	RRAMArr.append(np.where(K, A, 0.8795792047429445))
	RRAMArr.append(np.where(K, B, 0.8795792047429445))
	return RRAMArr


def NewConvExecPulse(B,PulseX,PulseY):
	BPA=len(PulseX)
	BPSA=len(PulseX[0])
	ColKer=len(PulseX[0][0])
	len3=len(B); len4=len(B[0])
	lenR1=int(len3/2)
	BX1,BX2=[[B[j][i] for i in range(len4)] for j in range(lenR1)],[[B[j+lenR1][i] for i in range(len4)] for j in range(lenR1)]
	L,BT1,BT2=[],np.array(BX1),np.array(BX2)
	PulseSign=(PulseX>=0)*(PulseY>=0)
	L.append(np.where(PulseSign, BT1[PulseX,PulseY], 0))
	L.append(np.where(PulseSign, BT2[PulseX,PulseY], 0))
	return L


def ArrPulseDef(RRAM,Image):

	Na=int(len(RRAM)/2)*9
	N1=min(180, 10*(len(Image)-int(Na/9)+1))
	L=[int(j/10) for j in range(N1)]
	k=[i%2 for i in L]
	m1=[j%10 for j in range(N1)]
	c=[i%9 for i in range(Na)]
	mt1=[-1*m1[j] if k[j]==1 else m1[j] for j in range(N1)]
	mt2=[9-i for i in m1]
	mx=[[9-c[i] if k[j]==0 else 1+c[i] for j in range(N1)] for i in range(Na)]
	m2=[[m1[j]-c[i]-mx[i][j] for j in range(N1)] for i in range(Na)]
	cond1=[[m1[j]<mx[i][j] for j in range(N1)] for i in range(Na)]
	X=[[i+(mt1[j] if cond1[i][j]==1 else (m2[i][j] if k[j]==0 else mt2[j])) for j in range(N1)] for i in range(Na)]
	Y=[[2*L[j]+(1-cond1[i][j]) for j in range(N1)] for i in range(Na)]
	return np.array(X),np.array(Y)

def ConvolutionComp8New(WeightX,ConvMatrixX,WeightX2,ConvMatrixX2,Div,X,Y,X1,X2,b):
	dt=6E-10;
	N2,M=int(3E-8/dt),int(dt/2E-11)
	term1,term2=(np.power(WeightX, 10)*X1)+X2,(np.power(WeightX2, 10)*X1)+X2
	Mod1,Mod2=ConvMatrixX[:,:,X,Y],ConvMatrixX2[:,:,X,Y]
	currentI1,Out1=np.sum(np.multiply(term1,Mod1[:,:]), axis=2)+np.sum(np.multiply(term2,Mod2[:,:]), axis=2), (np.sum(Mod1,axis=2)+np.sum(Mod2,axis=2))*b
	Current1=(currentI1*(N2/2)*M)-Out1
	return Current1








					

			

















	