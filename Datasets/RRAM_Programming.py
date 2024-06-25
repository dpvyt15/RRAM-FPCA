
import math
import sys
import random
import numpy as np
import time
import cmath
import joblib
from scipy import signal
import GeneralFunctions


def ReRAM_NewC(V_t,dt,init):
	dwdt=10*pow(V_t,13)
	f=(1-pow((pow((init-0.5),2)+0.75),7))
	X_position=float(dwdt*f*dt)+float(init)
	return X_position


def ReRAM_New_circuit2By2C(Vt,dt,x1):
	len1,len2,X=len(Vt),len(Vt[0]),x1
	Weights=[0 for i in range(len2)]
	for jj in range(len2):
		for ii in range(len1):
			X[jj]=ReRAM_NewC(Vt[ii][jj],dt,X[jj])
		Weights[jj]=X[jj]
	return Weights


def ReRAM_New_tbC16(ResetA):
	dt=4.5E-8
	N,N2=int(1.08E-4/dt),int(1.125E-6/dt)
	t= GeneralFunctions.linspace(0,192*math.pi,N)
	Vt1=signal.square(t)*0.835-0.835
	#Vt2=transpose(Vt1)
	len1,len2=len(ResetA),len(ResetA[0])
	Vt=[[0 for i in range(len1)] for j in range(N)]
	Weights=[[0 for i in range(len2)] for j in range(len1)]
	Vt=[[Vt1[ii] for jj in range(len1)] for ii in range(N)]
	Weights=[[0 for i in range(len2)] for j in range(len1)]

	for jj in range(len2):
		ResetAx=[row[jj]*N2 for row in ResetA]
		Vtx=[[(0 if i>ResetAx[k] else Vt[i][k]) for k in range(len(Vt[0]))] for i in range(len(Vt))]
		x1=[float(0.99) for i in range(2048)]
		WeightsN=ReRAM_New_circuit2By2C(Vtx,dt,x1)
		for ii in range(len1):
			Weights[ii][jj]=WeightsN[ii]
	return Weights


def TopSetWeightsMat64NN(ResetA):
	WeightA=ReRAM_New_tbC16(ResetA);    #ReRAM tuning block - works fine
	return WeightA



def SplitIt(A1,t1):
	RowA=len(A1); ColumnA=len(A1[0]);
	A1t=[[(i-t1 if i-t1>0 else 0) for i in j] for j in A1]
	A2t=[[A1[i][j]-A1t[i][j] for j in range(ColumnA)] for i in range(RowA)]
	return A1t,A2t


def ProgramMultiResSplitForJ5(A,R,Sp,Val):
	Div=pow(2,R)-1;
	Splits=pow(2,Sp);
	maxA,minA=GeneralFunctions.sorting(A)
	RowA=len(A); ColumnA=len(A[0]);
	U1xa=[[1 for i in range(ColumnA)] for j in range(RowA)]
	A1=[[A[i][j]-minA for j in range(len(A[0]))] for i in range(len(A))]
	DelA=float((maxA-minA)/(Splits*Div)); 
	DelA2=float(abs(minA)/Div)
	RexA=abs(maxA-minA)/2
	if(Val==0):
		Ax=[[[0 for i in range(ColumnA)] for j in range(RowA+1)] for k in range(Splits)]
		Ay=[[[0 for i in range(ColumnA)] for j in range(RowA+1)] for k in range(Splits)]
		Ax[0][RowA]=[Div for j in range(ColumnA)]
		Ay[0][RowA]=[Div for j in range(ColumnA)]
	if(Val>0):
		Ax=[[[0 for i in range(ColumnA)] for j in range(RowA)] for k in range(Splits)]
		Ay=[[[0 for i in range(ColumnA)] for j in range(RowA)] for k in range(Splits)]

		
	A1t,A2t=SplitIt(A1,RexA)
	maxA1t,minA1t= GeneralFunctions.sorting(A1t)
	maxA2t,minA2t=GeneralFunctions.sorting(A2t)
	if(Splits>=4):
		RexA=RexA/2
		A11t,A12t=SplitIt(A1t,RexA)   
		A21t,A22t=SplitIt(A2t,RexA)
        
		if(Splits==8):
			RexA=RexA/2
			A111t,A112t=SplitIt(A11t,RexA)    
			A121t,A122t=SplitIt(A12t,RexA)
			A211t,A212t=SplitIt(A21t,RexA)    
			A221t,A222t=SplitIt(A22t,RexA)

	if(Splits==2):
		for i in range(RowA):
			Ax[0][i]=[math.floor(A1t[i][j]/DelA) for j in range(ColumnA)]
			Ay[0][i]=[math.ceil(A1t[i][j]/DelA) for j in range(ColumnA)]
			Ax[1][i]=[math.floor(A2t[i][j]/DelA) for j in range(ColumnA)]
			Ay[1][i]=[math.ceil(A2t[i][j]/DelA) for j in range(ColumnA)]

	if(Splits==4):
		for i in range(RowA):
			Ax[0][i]=[math.floor(A11t[i][j]/DelA) for j in range(ColumnA)]
			Ay[0][i]=[math.ceil(A11t[i][j]/DelA) for j in range(ColumnA)] 
			Ax[1][i]=[math.floor(A12t[i][j]/DelA) for j in range(ColumnA)] 
			Ay[1][i]=[math.ceil(A12t[i][j]/DelA) for j in range(ColumnA)] 

			Ax[2][i]=[math.floor(A21t[i][j]/DelA) for j in range(ColumnA)] 
			Ay[2][i]=[math.ceil(A21t[i][j]/DelA) for j in range(ColumnA)] 
			Ax[3][i]=[math.floor(A22t[i][j]/DelA) for j in range(ColumnA)] 
			Ay[3][i]=[math.ceil(A22t[i][j]/DelA) for j in range(ColumnA)] 
    
	if(Splits==8):
		for i in range(RowA):
			Ax[0][i]=[math.floor(A111t[i][j]/DelA) for j in range(ColumnA)]
			Ay[0][i]=[math.ceil(A111t[i][j]/DelA) for j in range(ColumnA)]
			Ax[1][i]=[math.floor(A112t[i][j]/DelA) for j in range(ColumnA)]
			Ay[1][i]=[math.ceil(A112t[i][j]/DelA) for j in range(ColumnA)]

			Ax[2][i]=[math.floor(A121t[i][j]/DelA) for j in range(ColumnA)]
			Ay[2][i]=[math.ceil(A121t[i][j]/DelA) for j in range(ColumnA)]
			Ax[3][i]=[math.floor(A122t[i][j]/DelA) for j in range(ColumnA)]
			Ay[3][i]=[math.ceil(A122t[i][j]/DelA) for j in range(ColumnA)]

			Ax[4][i]=[math.floor(A211t[i][j]/DelA) for j in range(ColumnA)]
			Ay[4][i]=[math.ceil(A211t[i][j]/DelA) for j in range(ColumnA)]
			Ax[5][i]=[math.floor(A212t[i][j]/DelA) for j in range(ColumnA)]
			Ay[5][i]=[math.ceil(A212t[i][j]/DelA) for j in range(ColumnA)]

			Ax[6][i]=[math.floor(A221t[i][j]/DelA) for j in range(ColumnA)]
			Ay[6][i]=[math.ceil(A221t[i][j]/DelA) for j in range(ColumnA)]
			Ax[7][i]=[math.floor(A222t[i][j]/DelA) for j in range(ColumnA)]
			Ay[7][i]=[math.ceil(A222t[i][j]/DelA) for j in range(ColumnA)]

	WeightsA=joblib.Parallel(n_jobs=-1)(joblib.delayed(TopSetWeights)(Ax[tum],Ay[tum],R) for tum in range(Splits))
	return WeightsA,DelA,DelA2,minA


def ProgramMultiResSplitForJ52(A,R,Sp):
	Div=pow(2,R)-1;
	Splits=pow(2,Sp);
	RowA=len(A); ColumnA=len(A[0]);
	Ux=[[[(0 if k==0 else Div) for i in range(ColumnA)] for j in range(RowA)] for k in range(2)]
	Uy=[[[(0 if k==0 else Div) for i in range(ColumnA)] for j in range(RowA)] for k in range(2)]
	WeightsA=joblib.Parallel(n_jobs=-1)(joblib.delayed(TopSetWeights)(Ux[tum],Uy[tum],R) for tum in range(2))
	return WeightsA

def TopSetWeights(Ax,Ay,R):
	Mult=64/pow(2,R)
	RowA=len(Ax)-1
	ColumnA=len(Ax[0])
		
	A1x=[[96-(Mult*Ax[i][j]) for j in range(ColumnA)] for i in range(RowA+1)]
	A1y=[[96-(Mult*Ay[i][j]) for j in range(ColumnA)] for i in range(RowA+1)]

	AxX=GeneralFunctions.append_rows(A1x,A1y)

	WeightsA=TopSetWeightsMat64NN(AxX)	
	return WeightsA



def RRAMProgrammingMergeJ5(Ker,R,Sp,Val):

	if(Val==0):
		Weights,Del,Del2,min=ProgramMultiResSplitForJ5(Ker,R,Sp,Val)
		Weights2=Weights
	if(Val>0):
		Splits=pow(2,Sp); RowA=len(Ker[0][0]); ColumnA=len(Ker[0][0][0])
		Weights=[[[[[0 for i in range(ColumnA)] for j in range(2*(RowA))] for k in range(Splits)] for l in range(len(Ker[0]))] for m in range(len(Ker))]
		Weights2=[[[0 for i in range(ColumnA)] for j in range(2*RowA)] for u in range(2)]
		Del=[[0 for i in range(len(Ker[0]))] for j in range(len(Ker))]
		Del2=[[0 for i in range(len(Ker[0]))] for j in range(len(Ker))]
		min=[[0 for i in range(len(Ker[0]))] for j in range(len(Ker))]

		for i in range(len(Ker[0])):
			for j in range(len(Ker)):
				Weights[j][i],Del[j][i],Del2[j][i],min[j][i]=ProgramMultiResSplitForJ5(Ker[j][i],R,Sp,Val)
		Weights2=ProgramMultiResSplitForJ52(Ker[j][i],R,Sp)
	return Weights,Weights2,Del,Del2,min
