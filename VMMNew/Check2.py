
import math
import sys
sys.path.append('../')
import random
import numpy as np
import time
import cmath
import joblib
from scipy import signal
import Convolution.GeneralFunctions


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
	t= Convolution.GeneralFunctions.linspace(0,192*math.pi,N)
	Vt1=signal.square(t)*0.835-0.835
	#Vt2=transpose(Vt1)
	len1,len2=len(ResetA),len(ResetA[0])
	Vt=[[0 for i in range(len1)] for j in range(N)]
	Weights=[[0 for i in range(len2)] for j in range(len1)]
	Vt=[[Vt1[ii] for jj in range(len1)] for ii in range(N)]
	Weights=[[0 for i in range(len2)] for j in range(len1)]

	for jj in range(len2):
		ResetAx=[row[jj]*N2 for row in ResetA]
		Vtx=[[(0 if i>=ResetAx[k] else Vt[i][k]) for k in range(len(Vt[0]))] for i in range(len(Vt))]
		x1=[float(0.99) for i in range(len1)]
		WeightsN=ReRAM_New_circuit2By2C(Vtx,dt,x1)
		for ii in range(len1):
			Weights[ii][jj]=WeightsN[ii]
	return Weights


def TopSetWeightsMat64NN(ResetA):
	WeightA=ReRAM_New_tbC16(ResetA);    #ReRAM tuning block - works fine
	return WeightA



def SplitIt(A1,t1):
	RowA=len(A1); ColumnA=len(A1[0]);
	A1t=[[max(i-t1,0) for i in j] for j in A1]
	A2t=[[min(i,t1) for i in j] for j in A1]
	return A1t,A2t


def ProgramMultiResSplitForJ5(A,R,Sp,Val,jobs):
	Div=pow(2,R)-1;
	Splits=pow(2,Sp);
	maxA,minA=Convolution.GeneralFunctions.sorting(A)
	RowA=len(A); ColumnA=len(A[0]);
	U1xa=[[1 for i in range(ColumnA)] for j in range(RowA)]
	A1=[[A[i][j]-minA for j in range(len(A[0]))] for i in range(len(A))]
	DelA=float(abs(maxA-minA)/(Splits*Div)); 
	DelA2=float(abs(minA)/Div)
	RexA=abs(maxA-minA)/2


	L=[0.8795792047429445, 0.8818701819017782, 0.8843219277621278, 0.8867421343686188, 0.8891306230665125, 0.8914872294420887, 0.8938118036536142, 0.8961042107374637,0.8983643308880773, 0.9005920597105256, 0.9027873084445562, 0.9049500041591096, 0.9070800899164072, 0.9091775249048408, 0.9112422845400231, 0.9132743605334924, 0.9152737609287087, 0.9172405101041144, 0.9191746487431823, 0.9210762337715157, 0.9229453382612102, 0.92478205130283, 0.9265864778454922, 0.9283587385056906, 0.9300989693456189, 0.9318073216218871, 0.9334839615056368, 0.9351290697751838, 0.9367428414824126, 0.9383254855942506, 0.9398772246106353, 0.9413982941604654, 0.9428889425770947, 0.9443494304549837, 0.9457800301891774, 0.9471810254993037, 0.9485527109398244, 0.9498953913982767, 0.9512093815832571, 0.952495005503889, 0.9537525959425043, 0.9549824939222477, 0.9561850481712774, 0.9573606145852024, 0.9585095556893413, 0.9596322401023452, 0.9607290420026559, 0.9618003405992162,0.9628465196077709, 0.9637902630499349, 0.9647132407240816, 0.9656157624107309, 0.9664981397537165, 0.9673606859381514, 0.9682037153757057, 0.9690275433976513,0.969832485956091, 0.970618859333742, 0.9713869798625998, 0.9721371636517675, 0.9728697263246965, 0.97358498276604, 0.9742832468782843, 0.9749648313482839]

	Mult=64/pow(2,R)

	if(Val==0):
		Ax=[[[0 for i in range(ColumnA)] for j in range(RowA+1)] for k in range(Splits)]
		Ay=[[[0 for i in range(ColumnA)] for j in range(RowA+1)] for k in range(Splits)]
		Ax[0][RowA]=[L[int(Mult*Div)] for j in range(ColumnA)]
		Ay[0][RowA]=[L[int(Mult*Div)] for j in range(ColumnA)]
	if(Val>0):
		Ax=[[[0 for i in range(ColumnA)] for j in range(RowA)] for k in range(Splits)]
		Ay=[[[0 for i in range(ColumnA)] for j in range(RowA)] for k in range(Splits)]

	if(DelA==0):
		DelA=1
	
	if(DelA2==0):
		minA=1

	#print(A1)	

	#print(RexA, DelA, minA, maxA)





	if(Splits>=2):
		A1t,A2t=SplitIt(A1,RexA)
	
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

	if(Splits==1):
		Ax[0]=[[L[int(Mult*min(Div,math.floor(A1[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[0]=[[L[int(Mult*min(Div,math.ceil(A1[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]		

	if(Splits==2):
		Ax[0]=[[L[int(Mult*min(Div,math.floor(A1t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[0]=[[L[int(Mult*min(Div,math.ceil(A1t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ax[1]=[[L[int(Mult*min(Div,math.floor(A2t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[1]=[[L[int(Mult*min(Div,math.ceil(A2t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]

	if(Splits==4):
		Ax[0]=[[L[int(Mult*min(Div,math.floor(A11t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[0]=[[L[int(Mult*min(Div,math.ceil(A11t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ax[1]=[[L[int(Mult*min(Div,math.floor(A12t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[1]=[[L[int(Mult*min(Div,math.ceil(A12t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]

		Ax[2]=[[L[int(Mult*min(Div,math.floor(A21t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[2]=[[L[int(Mult*min(Div,math.ceil(A21t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ax[3]=[[L[int(Mult*min(Div,math.floor(A22t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[3]=[[L[int(Mult*min(Div,math.ceil(A22t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]

    
	if(Splits==8):

		Ax[0]=[[L[int(Mult*min(Div,math.floor(A111[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[0]=[[L[int(Mult*min(Div,math.ceil(A111t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ax[1]=[[L[int(Mult*min(Div,math.floor(A112t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[1]=[[L[int(Mult*min(Div,math.ceil(A112t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]

		Ax[2]=[[L[int(Mult*min(Div,math.floor(A121t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[2]=[[L[int(Mult*min(Div,math.ceil(A121t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ax[3]=[[L[int(Mult*min(Div,math.floor(A122t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[3]=[[L[int(Mult*min(Div,math.ceil(A122t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]

		Ax[4]=[[L[int(Mult*min(Div,math.floor(A211t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[4]=[[L[int(Mult*min(Div,math.ceil(A211t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ax[5]=[[L[int(Mult*min(Div,math.floor(A212t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[5]=[[L[int(Mult*min(Div,math.ceil(A212t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]

		Ax[6]=[[L[int(Mult*min(Div,math.floor(A221t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[6]=[[L[int(Mult*min(Div,math.ceil(A221t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ax[7]=[[L[int(Mult*min(Div,math.floor(A222t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]
		Ay[7]=[[L[int(Mult*min(Div,math.ceil(A222t[i][j]/DelA)))] for j in range(ColumnA)] for i in range(RowA)]

	WeightsA=joblib.Parallel(n_jobs=jobs)(joblib.delayed(TopSetWeights)(Ax[tum],Ay[tum],R) for tum in range(Splits))
	return WeightsA,DelA,DelA2,minA


def ProgramMultiResSplitForJ52(A,R,Sp,jobs):
	Div=pow(2,R)-1;
	RowA=len(A); ColumnA=len(A[0]);

	L=[0.8795792047429445, 0.8818701819017782, 0.8843219277621278, 0.8867421343686188, 0.8891306230665125, 0.8914872294420887, 0.8938118036536142, 0.8961042107374637,0.8983643308880773, 0.9005920597105256, 0.9027873084445562, 0.9049500041591096, 0.9070800899164072, 0.9091775249048408, 0.9112422845400231, 0.9132743605334924, 0.9152737609287087, 0.9172405101041144, 0.9191746487431823, 0.9210762337715157, 0.9229453382612102, 0.92478205130283, 0.9265864778454922, 0.9283587385056906, 0.9300989693456189, 0.9318073216218871, 0.9334839615056368, 0.9351290697751838, 0.9367428414824126, 0.9383254855942506, 0.9398772246106353, 0.9413982941604654, 0.9428889425770947, 0.9443494304549837, 0.9457800301891774, 0.9471810254993037, 0.9485527109398244, 0.9498953913982767, 0.9512093815832571, 0.952495005503889, 0.9537525959425043, 0.9549824939222477, 0.9561850481712774, 0.9573606145852024, 0.9585095556893413, 0.9596322401023452, 0.9607290420026559, 0.9618003405992162,0.9628465196077709, 0.9637902630499349, 0.9647132407240816, 0.9656157624107309, 0.9664981397537165, 0.9673606859381514, 0.9682037153757057, 0.9690275433976513,0.969832485956091, 0.970618859333742, 0.9713869798625998, 0.9721371636517675, 0.9728697263246965, 0.97358498276604, 0.9742832468782843, 0.9749648313482839, 0.9749648313482839]

	Mult=64/pow(2,R)

	Ux=[[[L[int(Mult*(0 if k==0 else Div))] for i in range(ColumnA)] for j in range(RowA)] for k in range(2)]
	Uy=[[[L[int(Mult*(0 if k==0 else Div))] for i in range(ColumnA)] for j in range(RowA)] for k in range(2)]
	WeightsA=joblib.Parallel(n_jobs=jobs)(joblib.delayed(TopSetWeights)(Ux[tum],Uy[tum],R) for tum in range(2))
	return WeightsA

def TopSetWeights(Ax,Ay,R):
	#Mult=64/pow(2,R)
	#RowA=len(Ax)
	#ColumnA=len(Ax[0])


	#print(Ax, Ay)

		
	#A1x=[[L[int(Mult*i)] for i in j] for j in Ax]
	#A1y=[[L[int(Mult*i)] for i in j] for j in Ay]

	AxX=Convolution.GeneralFunctions.append_rows(Ax,Ay)

	#WeightsA=TopSetWeightsMat64NN(AxX)
	#WeightsA=LAX(AxX)	
	return AxX

def LAX(A):
	L=[0.8795792047429445, 0.8818701819017782, 0.8843219277621278, 0.8867421343686188, 0.8891306230665125, 0.8914872294420887, 0.8938118036536142, 0.8961042107374637,0.8983643308880773, 0.9005920597105256, 0.9027873084445562, 0.9049500041591096, 0.9070800899164072, 0.9091775249048408, 0.9112422845400231, 0.9132743605334924, 0.9152737609287087, 0.9172405101041144, 0.9191746487431823, 0.9210762337715157, 0.9229453382612102, 0.92478205130283, 0.9265864778454922, 0.9283587385056906, 0.9300989693456189, 0.9318073216218871, 0.9334839615056368, 0.9351290697751838, 0.9367428414824126, 0.9383254855942506, 0.9398772246106353, 0.9413982941604654, 0.9428889425770947, 0.9443494304549837, 0.9457800301891774, 0.9471810254993037, 0.9485527109398244, 0.9498953913982767, 0.9512093815832571, 0.952495005503889, 0.9537525959425043, 0.9549824939222477, 0.9561850481712774, 0.9573606145852024, 0.9585095556893413, 0.9596322401023452, 0.9607290420026559, 0.9618003405992162,0.9628465196077709, 0.9637902630499349, 0.9647132407240816, 0.9656157624107309, 0.9664981397537165, 0.9673606859381514, 0.9682037153757057, 0.9690275433976513,0.969832485956091, 0.970618859333742, 0.9713869798625998, 0.9721371636517675, 0.9728697263246965, 0.97358498276604, 0.9742832468782843, 0.9749648313482839, 0.9749648313482839]

	W=[[L[int(i)] for i in j] for j in A]
	return W


def RRAMProgrammingMergeJ5(Ker,R,Sp,Val,jobs):

	start_time=time.time()
	if(Val==0):
		Weights,Del,Del2,min=ProgramMultiResSplitForJ5(Ker,R,Sp,Val,jobs)
		Weights2=Weights
	if(Val>0):
		Splits=pow(2,Sp); RowA=len(Ker[0][0]); ColumnA=len(Ker[0][0][0])
		Weights=[[[[[0 for i in range(ColumnA)] for j in range(2*(RowA))] for k in range(Splits)] for l in range(len(Ker[0]))] for m in range(len(Ker))]
		Weights2=[[[0 for i in range(ColumnA)] for j in range(2*RowA)] for u in range(2)]
		Del=[[0 for i in range(len(Ker[0]))] for j in range(len(Ker))]
		Del2=[[0 for i in range(len(Ker[0]))] for j in range(len(Ker))]
		min=[[0 for i in range(len(Ker[0]))] for j in range(len(Ker))]

		for i in range(len(Ker)):
			for j in range(len(Ker[0])):
				Weights[i][j],Del[i][j],Del2[i][j],min[i][j]=ProgramMultiResSplitForJ5(Ker[i][j],R,Sp,Val,jobs)
		Weights2=ProgramMultiResSplitForJ52(Ker[i][j],R,Sp,jobs)

	end_time=time.time()
	
	print(end_time-start_time)

	return Weights,Weights2,Del,Del2,min
