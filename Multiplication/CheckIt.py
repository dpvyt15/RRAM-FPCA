
#!/usr/bin/python
import math
import sys
sys.path.append('../')

import random
import Multiplication.PostProcessing
import Multiplication.RRAMProgramming
import Multiplication.J4_MultNew_MultForDiv

"""This module determines the number of PS arrays to be used for multiplication based on BW. It then proceeds to prepare the PS array by determining the location of different bits of the operand within the array. Further, it adds redundancy to overcome the variability issues faced by RRAMs"""

def BWSplitAdd(RRAM,Pulse,BW):
	l1a=int(BW/8)
	l2a=(l1a)%10

	t1=[[-8 for i in range(2)] for j in range(len(RRAM))]
	tP=[[-8 for i in range(2)] for j in range(len(RRAM))]
	t21=[[-8 for i in range(2)] for j in range(len(RRAM))]
	tP21=[[-8 for i in range(2)] for j in range(len(RRAM))]
	t22=[[-8 for i in range(2)] for j in range(len(RRAM))]
	tP22=[[-8 for i in range(2)] for j in range(len(RRAM))]
	t31=[[-8 for i in range(2)] for j in range(len(RRAM))]
	tP31=[[-8 for i in range(2)] for j in range(len(RRAM))]
	t32=[[-8 for i in range(2)] for j in range(len(RRAM))]
	tP32=[[-8 for i in range(2)] for j in range(len(RRAM))]
	t33=[[-8 for i in range(2)] for j in range(len(RRAM))]
	tP33=[[-8 for i in range(2)] for j in range(len(RRAM))]
	index=[[-8 for i in range(3)] for j in range(len(RRAM))]
	index21=[[-8 for i in range(3)] for j in range(len(RRAM))]
	index31=[[-8 for i in range(3)] for j in range(len(RRAM))]
	l1=len(RRAM[0])
	l2=len(RRAM[0][0])

	

	#print("Its still going into this")
	for i in range(len(RRAM)):
		check=0
		check1=0
		for k in range(len(RRAM[0][0])):
			if(RRAM[i][0][k][0]>=0):
				t1[i][0]=RRAM[i][0][k][0]
				tP[i][0]=Pulse[i][0][k][0]

			if(RRAM[i][0][k][1]>=0 and check==0):
				t21[i][0]=RRAM[i][0][k][1]
				tP21[i][0]=Pulse[i][0][k][1]
				t22[i][0]=RRAM[i][0][k+1][1]
				tP22[i][0]=Pulse[i][0][k+1][1]		
				check=1

			if(RRAM[i][0][k][2]>=0 and check1==0):
				if(i==len(RRAM)-1 and l2a>2):
					t31[i][0]=RRAM[i][0][k][2]
					tP31[i][0]=Pulse[i][0][k][2]
					t32[i][0]=RRAM[i][0][k+1][2]
					tP32[i][0]=Pulse[i][0][k+1][2]	
					t33[i][0]=RRAM[i][0][k+2][2]
					tP33[i][0]=Pulse[i][0][k+2][2]	
					check1=1

				if(i<len(RRAM)-1):
					t31[i][0]=RRAM[i][0][k][2]
					tP31[i][0]=Pulse[i][0][k][2]
					t32[i][0]=RRAM[i][0][k+1][2]
					tP32[i][0]=Pulse[i][0][k+1][2]	
					t33[i][0]=RRAM[i][0][k+2][2]
					tP33[i][0]=Pulse[i][0][k+2][2]	
					check1=1

	for i in range(len(RRAM)):
		count=0
		for j in range(len(RRAM[0][0][0])-1,-1,-1):
			for k in range(len(RRAM[0][0])):
				if(RRAM[i][l1-1][k][j]>0 and count==0):
					t1[i][1]=RRAM[i][l1-1][k][j]
					tP[i][1]=Pulse[i][l1-1][k][j]
					index[i][0]=l1-1
					index[i][1]=k
					index[i][2]=j
					ta=j
					count=1
				if(RRAM[i][l1-1][k][j]>0 and count==1 and j<ta):
					t21[i][1]=RRAM[i][l1-1][k][j]
					tP21[i][1]=Pulse[i][l1-1][k][j]
					index21[i][0]=l1-1
					index21[i][1]=k
					index21[i][2]=j
					t22[i][1]=RRAM[i][l1-1][k+1][j]
					tP22[i][1]=Pulse[i][l1-1][k+1][j]
					ta1=j
					count=2
				if(RRAM[i][l1-1][k][j]>0 and count==2 and j<ta1):
					if(i==len(RRAM)-1 and l2a>2):
						#print("Third value determination")
						t31[i][1]=RRAM[i][l1-1][k][j]
						tP31[i][1]=Pulse[i][l1-1][k][j]
						index31[i][0]=l1-1
						index31[i][1]=k
						index31[i][2]=j
						t32[i][1]=RRAM[i][l1-1][k+1][j]
						tP32[i][1]=Pulse[i][l1-1][k+1][j]
						t33[i][1]=RRAM[i][l1-1][k+2][j]
						tP33[i][1]=Pulse[i][l1-1][k+2][j]
						count=3

					if(i<len(RRAM)-1):
						#print("Third value determination")
						t31[i][1]=RRAM[i][l1-1][k][j]
						tP31[i][1]=Pulse[i][l1-1][k][j]
						index31[i][0]=l1-1
						index31[i][1]=k
						index31[i][2]=j
						t32[i][1]=RRAM[i][l1-1][k+1][j]
						tP32[i][1]=Pulse[i][l1-1][k+1][j]
						t33[i][1]=RRAM[i][l1-1][k+2][j]
						tP33[i][1]=Pulse[i][l1-1][k+2][j]
						count=3

		if(t1[i][1]<0):
			count=0
			for j in range(len(RRAM[0][0][0])-1,-1,-1):
				for k in range(len(RRAM[0][0])):
					if(RRAM[i][l1-2][k][j]>0 and count==0):
						t1[i][1]=RRAM[i][l1-2][k][j]
						tP[i][1]=Pulse[i][l1-2][k][j]
						index[i][0]=l1-2
						index[i][1]=k
						index[i][2]=j
						ta=j
						count=1
					if(RRAM[i][l1-2][k][j]>0 and count==1 and j<ta):
						t21[i][1]=RRAM[i][l1-2][k][j]
						tP21[i][1]=Pulse[i][l1-2][k][j]
						index21[i][0]=l1-2
						index21[i][1]=k
						index21[i][2]=j
						t22[i][1]=RRAM[i][l1-2][k+1][j]
						tP22[i][1]=Pulse[i][l1-2][k+1][j]
						ta1=j
						count=2	
					if(RRAM[i][l1-2][k][j]>0 and count==2 and j<ta1):
						if(i==len(RRAM)-1 and l2a>2):
							t31[i][1]=RRAM[i][l1-2][k][j]
							tP31[i][1]=Pulse[i][l1-2][k][j]
							index31[i][0]=l1-2
							index31[i][1]=k
							index31[i][2]=j
							t32[i][1]=RRAM[i][l1-2][k+1][j]
							tP32[i][1]=Pulse[i][l1-2][k+1][j]
							t33[i][1]=RRAM[i][l1-2][k+2][j]
							tP33[i][1]=Pulse[i][l1-2][k+2][j]
							count=3
						if(i<len(RRAM)-1):
							t31[i][1]=RRAM[i][l1-2][k][j]
							tP31[i][1]=Pulse[i][l1-2][k][j]
							index31[i][0]=l1-2
							index31[i][1]=k
							index31[i][2]=j
							t32[i][1]=RRAM[i][l1-2][k+1][j]
							tP32[i][1]=Pulse[i][l1-2][k+1][j]
							t33[i][1]=RRAM[i][l1-2][k+2][j]
							tP33[i][1]=Pulse[i][l1-2][k+2][j]
							count=3

	for i in range(len(RRAM)):
		Z=index[i][0]
		X=index[i][1]
		Y=index[i][2]
		X21=index21[i][1]
		Y21=index21[i][2]
		X31=index31[i][1]
		Y31=index31[i][2]

		for j in range(10):
			if(j>1):
				RRAM[i][0][j][0]=t1[i][0]
				Pulse[i][0][j][0]=tP[i][0]
	
				if(j%2==0):
					RRAM[i][0][j][1]=t21[i][0]
					Pulse[i][0][j][1]=tP21[i][0]

				if(j%2==1):
					RRAM[i][0][j][1]=t22[i][0]
					Pulse[i][0][j][1]=tP22[i][0]

			if(j%3==0 and j<3):
				RRAM[i][0][j][2]=t31[i][0]
				Pulse[i][0][j][2]=tP31[i][0]

			if(j%3==1 and j<3):
				RRAM[i][0][j][2]=t32[i][0]
				Pulse[i][0][j][2]=tP32[i][0]

			if(j%3==2 and j<3):
				RRAM[i][0][j][2]=t33[i][0]
				Pulse[i][0][j][2]=tP33[i][0]

		for j in range(10):
			if((X>1 and j>1) or (X<1 and j<8)):
				RRAM[i][Z][j][Y]=t1[i][1]
				Pulse[i][Z][j][Y]=tP[i][1]
				if(j%2==0):
					RRAM[i][Z][j][Y21]=t21[i][1]
					Pulse[i][Z][j][Y21]=tP21[i][1]
				if(j%2==1):
					RRAM[i][Z][j][Y21]=t22[i][1]
					Pulse[i][Z][j][Y21]=tP22[i][1]

			if(X31>=5 and t31[i][1]>0):
				if(j%3==0 and j<3):
					RRAM[i][Z][j][Y31]=t31[i][1]
					Pulse[i][Z][j][Y31]=tP31[i][1]

				if(j%3==1 and j<3):
					RRAM[i][Z][j][Y31]=t32[i][1]
					Pulse[i][Z][j][Y31]=tP32[i][1]

				if(j%3==2 and j<3):
					RRAM[i][Z][j][Y31]=t33[i][1]
					Pulse[i][Z][j][Y31]=tP33[i][1]

			if(X31<5  and t31[i][1]>0):
				if(j%3==1 and j>6):
					RRAM[i][Z][j][Y31]=t31[i][1]
					Pulse[i][Z][j][Y31]=tP31[i][1]

				if(j%3==2 and j>6):
					RRAM[i][Z][j][Y31]=t32[i][1]
					Pulse[i][Z][j][Y31]=tP32[i][1]

				if(j%3==0 and j>6):
					RRAM[i][Z][j][Y31]=t33[i][1]
					Pulse[i][Z][j][Y31]=tP33[i][1]

	l1a=int(BW/8)
	l2a=(l1a)%10
	ax=len(RRAM)-1

	if(l2a<6):
		pa=((l2a==2)*8)+((l2a==4)*6)
		ran=((l2a==2)*8)+((l2a==4)*4)
		fir=((l2a==2)*2)+((l2a==4)*0)
		for i in range(len(RRAM[0])):	
			for j in range(ran):
				Z=index[ax][0]
				Y=index[ax][2]
				Y21=index21[ax][2]
				Y31=index31[ax][2]
				k=int(j%l2a)
				if(i!=0 and i!=Z):
					for a in range(len(RRAM[0][0][0])):
						RRAM[ax][i][fir+j][a]=RRAM[ax][i][pa+k][a]
						Pulse[ax][i][fir+j][a]=Pulse[ax][i][pa+k][a]
				if(i==0 and i!=Z):
					for a in range(2,len(RRAM[0][0][0])):
						RRAM[ax][i][fir+j][a]=RRAM[ax][i][pa+k][a]
						Pulse[ax][i][fir+j][a]=Pulse[ax][i][pa+k][a]
				if(i!=0 and i==Z):
					for a in range(Y21):
						RRAM[ax][i][fir+j][a]=RRAM[ax][i][pa+k][a]
						Pulse[ax][i][fir+j][a]=Pulse[ax][i][pa+k][a]
				if(i==0 and i==Z):
					for a in range(2,Y21,1):
						RRAM[ax][i][fir+j][a]=RRAM[ax][i][pa+k][a]
						Pulse[ax][i][fir+j][a]=Pulse[ax][i][pa+k][a]	

	#print(RRAM,Pulse)	

	return(RRAM,Pulse,index,index21,index31)


def BWSplit(BW):
	l1=int(BW/8)
	t1=int(math.ceil(l1/10))
	t2=int(math.ceil((l1+9)/9))
	ArrPulse=[[[[-8 for i in range(9)] for j in range(10)] for k in range(t2)] for e in range(t1)]
	ArrRRAM=[[[[-8 for i in range(9)] for j in range(10)] for k in range(t2)] for e in range(t1)]
	for u in range(t1):
		start1=0
		for i in range(t2):
			for j in range(10):
				ax=u*10+(9-j)
				if(ax<l1):
					for k in range(9):
						t=-8*(9-j)+start1+8*k	
						if(t<BW):
							ArrPulse[u][i][j][k]=t
						else:
							ArrPulse[u][i][j][k]=-8
			start1=start1+(9*8)

	for u in range(t1):
		count=0
		Lo=9
		Up=10
		for i in range(t2):
			for k in range(9):
				for j in range(Lo,Up,1):
					t=u*10+9-j	
					if(t<l1):
						ArrRRAM[u][i][j][k]=t*8
					else:
						ArrRRAM[u][i][j][k]=-8
				if(count<l1):
					count=count+1
				if(Lo>0):
					Lo=Lo-1
				if(count==l1):
					Up=Up-1
	return(ArrRRAM,ArrPulse)




def SplitInputs(BW,AA,BA,Var):	
	ArrRRAM,ArrPulse=BWSplit(BW)
	ArrRRAMNew,ArrPulseNew,index,index21,index31=BWSplitAdd(ArrRRAM,ArrPulse,BW)
	FullArrRRAM=Multiplication.RRAMProgramming.IndiAssignRRAM(ArrRRAMNew,AA)
	FullArrPulse=Multiplication.RRAMProgramming.IndiAssignPulses(ArrPulseNew,BA)
	Output=Multiplication.J4_MultNew_MultForDiv.Multiply(FullArrRRAM,FullArrPulse,BW,Var)
	OutA=Multiplication.PostProcessing.PrePostProcess(Output,index,index21,index31,BW)
	Out=Multiplication.PostProcessing.PostProcess(OutA,BW)
	FinalOut=Multiplication.PostProcessing.Finalprocessing(Out,BW)
	return(FinalOut)


