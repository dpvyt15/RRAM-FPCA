
import math
import sys
import random

"""Based on location of operand bits within the PS array determined in CheckIt module, this module programs the RRAM array. It then proceeds to place the 2nd operand at the appropriate input locations for all the necessary PS arrays. The pulse location is determined in CheckIt module as well. The RRAM array and pulse information are then passed onto the compute module."""

def IndiAssignRRAM(ArrRRAM,AA):
	Arr=[[[[0 for i in range(180)] for j in range(90)] for k in range(len(ArrRRAM[0]))] for l in range(len(ArrRRAM))]
	for um in range(len(ArrRRAM)):
		for ux in range(len(ArrRRAM[0])):
			for tip in range(10):
				for i in range(18):
					ax=int(i/2)
					B=ArrRRAM[um][ux][tip][ax]
					if(i%2==0):
						Up=8
						Lo=2
						for j in range(10):
							u=10*i+j
							for k in range(Lo,Up,1):
								v=9*tip+k
								Arr[um][ux][v][u]=AA[B+7-k]
								if(Arr[um][ux][v][u]<0):
									Arr[um][ux][v][u]=0
							if(j>=2):
								Up=Up-1
							if(Lo>0):
								Lo=Lo-1
					if(i%2==1):
						Lo=8
						for j in range(5,10,1):
							u=10*i+9-j
							for k in range(Lo,9,1):
								v=9*tip+k
								Arr[um][ux][v][u]=AA[B+8-k]
								if(Arr[um][ux][v][u]<0):
									Arr[um][ux][v][u]=0
							if(Lo>4):
								Lo=Lo-1

	return(Arr)
	

def IndiAssignPulses(ArrPulse,BA):
	ArrP=[[[[0 for i in range(4*len(ArrPulse[0][0][0]))] for j in range(len(ArrPulse[0][0])*9)] for k in range(len(ArrPulse[0]))] for u in range(len(ArrPulse))]
	for u in range(len(ArrPulse)):
		for k in range(len(ArrPulse[0])):
			for i in range(len(ArrPulse[0][0])):
				for j in range(len(ArrPulse[0][0][0])):
					B=ArrPulse[u][k][i][j]
					for pix in range(9):
						if(pix>1 and B>=0):
							ArrP[u][k][9*i+pix][4*j]=BA[B+pix-2]

						if(pix==0 and B>=0):
							ArrP[u][k][9*i+pix][4*j+1]=BA[B+7]

						if(pix>3 and B>=0):
							ArrP[u][k][9*i+pix][4*j+2]=BA[B+pix-4]
	for u in range(len(ArrPulse)):
		for k in range(len(ArrPulse[0])):
			for i in range(1,len(ArrPulse[0][0])):
				for j in range(len(ArrPulse[0][0][0])):
					for pix in range(9):
						if(pix>3):
							ArrP[u][k][9*i+pix][4*j+1]=ArrP[u][k][9*(i-1)+pix][4*j+2]
		
						if(pix>1 and j<len(ArrPulse[0][0][0])-1):
							ArrP[u][k][9*i+pix][4*j+3]=ArrP[u][k][9*(i-1)+pix][4*(j+1)]
	for u in range(len(ArrPulse)):
		for k in range(len(ArrPulse[0])):
			for i in range(len(ArrPulse[0][0])-1):
				for j in range(len(ArrPulse[0][0][0])):
					for pix in range(9):
						if(pix==0):
							ArrP[u][k][9*i+pix][4*j+2]=ArrP[u][k][9*(i+1)+pix][4*j+1]
	ArrF=[[[[0 for i in range(4*len(ArrPulse[0][0][0]))] for j in range(len(ArrPulse[0][0])*9)] for k in range(len(ArrPulse[0]))] for u in range(len(ArrPulse))]
	for u in range(len(ArrPulse)):
		for k in range(len(ArrPulse[0])):
			for i in range(len(ArrPulse[0][0])):
				for j in range(len(ArrPulse[0][0][0])):
					for pix in range(9):
						ArrF[u][k][9*i+pix][4*j]=ArrP[u][k][9*i+pix][4*j]
						ArrF[u][k][9*i+pix][4*j+1]=ArrP[u][k][9*i+pix][4*j+1]
						ArrF[u][k][9*i+pix][4*j+2]=ArrP[u][k][9*i+pix][4*j+3]
						ArrF[u][k][9*i+pix][4*j+3]=ArrP[u][k][9*i+pix][4*j+2]

	return(ArrP)