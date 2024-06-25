
import math
import sys
sys.path.append('../')
import random
import numpy as np
import time
import cmath
import joblib
from scipy import signal
import ConvolutionCheck.GeneralFunctions


def SplitIt(A1,t1):
	RowA=len(A1); ColumnA=len(A1[0]);
	A1t=[[max(i-t1,0) for i in j] for j in A1]
	A2t=[[min(i,t1) for i in j] for j in A1]
	return A1t,A2t


def ProgramMultiResSplitForJ5(A,R,Sp,Val,jobs):
	Div=pow(2,R)-1;
	Splits=pow(2,Sp);
	maxA,minA=ConvolutionCheck.GeneralFunctions.sorting(A)
	RowA=len(A); ColumnA=len(A[0]);
	U1xa=[[1 for i in range(ColumnA)] for j in range(RowA)]
	A1=[[A[i][j]-minA for j in range(len(A[0]))] for i in range(len(A))]
	DelA=float(abs(maxA-minA)/(Splits*Div)); 
	DelA2=float(abs(minA)/Div)
	RexA=abs(maxA-minA)/2



	L=[0.8795792047429445, 0.8818701819017782, 0.8843219277621278, 0.8867421343686188, 0.8891306230665125, 0.8914872294420887, 0.8938118036536142, 0.8961042107374637,0.8983643308880773, 0.9005920597105256, 0.9027873084445562, 0.9049500041591096, 0.9070800899164072, 0.9091775249048408, 0.9112422845400231, 0.9132743605334924, 0.9152737609287087, 0.9172405101041144, 0.9191746487431823, 0.9210762337715157, 0.9229453382612102, 0.92478205130283, 0.9265864778454922, 0.9283587385056906, 0.9300989693456189, 0.9318073216218871, 0.9334839615056368, 0.9351290697751838, 0.9367428414824126, 0.9383254855942506, 0.9398772246106353, 0.9413982941604654, 0.9428889425770947, 0.9443494304549837, 0.9457800301891774, 0.9471810254993037, 0.9485527109398244, 0.9498953913982767, 0.9512093815832571, 0.952495005503889, 0.9537525959425043, 0.9549824939222477, 0.9561850481712774, 0.9573606145852024, 0.9585095556893413, 0.9596322401023452, 0.9607290420026559, 0.9618003405992162,0.9628465196077709, 0.9637902630499349, 0.9647132407240816, 0.9656157624107309, 0.9664981397537165, 0.9673606859381514, 0.9682037153757057, 0.9690275433976513,0.969832485956091, 0.970618859333742, 0.9713869798625998, 0.9721371636517675, 0.9728697263246965, 0.97358498276604, 0.9742832468782843, 0.9749648313482839, 0.9749648313482839]

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

	Lim=Mult*Div

	if(Splits==1):
		Ax[0]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A1)/DelA)]
		Ay[0]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A1)/DelA)]		

	if(Splits==2):
		Ax[0]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A1t)/DelA)]
		Ay[0]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A1t)/DelA)]
		Ax[1]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A2t)/DelA)]
		Ay[1]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A2t)/DelA)]

	if(Splits==4):
		Ax[0]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A11t)/DelA)]
		Ay[0]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A11t)/DelA)]
		Ax[1]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A12t)/DelA)]
		Ay[1]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A12t)/DelA)]

		Ax[2]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A21t)/DelA)]
		Ay[2]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A21t)/DelA)]
		Ax[3]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A22t)/DelA)]
		Ay[3]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A22t)/DelA)]

    
	if(Splits==8):

		Ax[0]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A111t)/DelA)]
		Ay[0]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A111t)/DelA)]
		Ax[1]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A112t)/DelA)]
		Ay[1]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A112t)/DelA)]

		Ax[2]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A121t)/DelA)]
		Ay[2]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A121t)/DelA)]
		Ax[3]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A122t)/DelA)]
		Ay[3]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A122t)/DelA)]

		Ax[4]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A211t)/DelA)]
		Ay[4]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A211t)/DelA)]
		Ax[5]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A212t)/DelA)]
		Ay[5]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A212t)/DelA)]

		Ax[6]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A221t)/DelA)]
		Ay[6]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A221t)/DelA)]
		Ax[7]=[[L[int(i)] for i in j] for j in Mult*np.floor(np.array(A222t)/DelA)]
		Ay[7]=[[L[int(i)] for i in j] for j in Mult*np.ceil(np.array(A222t)/DelA)]

	#WeightsA=joblib.Parallel(n_jobs=jobs)(joblib.delayed(TopSetWeights)(Ax[tum],Ay[tum],R) for tum in range(Splits))

	WeightsA=[]
	for i in range(len(Ax)):
		WeightsA.append(ConvolutionCheck.GeneralFunctions.append_rows(Ax[i],Ay[i]))

	return WeightsA,DelA,DelA2,minA


def ProgramMultiResSplitForJ52(A,R,Sp,jobs):
	Div=pow(2,R)-1;
	RowA=len(A); ColumnA=len(A[0]);

	L=[0.8795792047429445, 0.8818701819017782, 0.8843219277621278, 0.8867421343686188, 0.8891306230665125, 0.8914872294420887, 0.8938118036536142, 0.8961042107374637,0.8983643308880773, 0.9005920597105256, 0.9027873084445562, 0.9049500041591096, 0.9070800899164072, 0.9091775249048408, 0.9112422845400231, 0.9132743605334924, 0.9152737609287087, 0.9172405101041144, 0.9191746487431823, 0.9210762337715157, 0.9229453382612102, 0.92478205130283, 0.9265864778454922, 0.9283587385056906, 0.9300989693456189, 0.9318073216218871, 0.9334839615056368, 0.9351290697751838, 0.9367428414824126, 0.9383254855942506, 0.9398772246106353, 0.9413982941604654, 0.9428889425770947, 0.9443494304549837, 0.9457800301891774, 0.9471810254993037, 0.9485527109398244, 0.9498953913982767, 0.9512093815832571, 0.952495005503889, 0.9537525959425043, 0.9549824939222477, 0.9561850481712774, 0.9573606145852024, 0.9585095556893413, 0.9596322401023452, 0.9607290420026559, 0.9618003405992162,0.9628465196077709, 0.9637902630499349, 0.9647132407240816, 0.9656157624107309, 0.9664981397537165, 0.9673606859381514, 0.9682037153757057, 0.9690275433976513,0.969832485956091, 0.970618859333742, 0.9713869798625998, 0.9721371636517675, 0.9728697263246965, 0.97358498276604, 0.9742832468782843, 0.9749648313482839, 0.9749648313482839]

	Mult=64/pow(2,R)

	Ux=[[[L[int(Mult*(0 if k==0 else Div))] for i in range(ColumnA)] for j in range(RowA)] for k in range(2)]
	Uy=[[[L[int(Mult*(0 if k==0 else Div))] for i in range(ColumnA)] for j in range(RowA)] for k in range(2)]
	
	WeightsA=[]
	for i in range(len(Ux)):
		WeightsA.append(ConvolutionCheck.GeneralFunctions.append_rows(Ux[i],Uy[i]))
	return WeightsA

def RRAMProgrammingMergeJ5(Ker,R,Sp,Val,jobs):

	if(Val==0):
		print("Error: This is Matrix-Matrix Multiplication file. Please use the RRAM Programming file in the VMM folder for Matrix-Vector Multplications")
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

	#end_time=time.time()
	
	#print(end_time-start_time)

	return Weights,Weights2,Del,Del2,min
