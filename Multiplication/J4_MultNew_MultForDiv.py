#!/usr/bin/python
import math
import sys
import random
import numpy as np
from joblib import Parallel, delayed


""" This module takes in a PS array, inputs and performs in-memory compute. The output currents are then converted to digital equivalents using Sense amplifiers and ADCs. The ADC outputs are then returned to control plane. PS array size is predetermined. """


def Top(RRAM, N1, Pulse,Var):

	#print("In the in-mem com block")
	Out=[[[0 for i in range(180)] for j in range(N1)] for k in range(len(RRAM))]
	for XA in range(len(RRAM)):
		for XM in range(N1):
			for um in range(18):
				k=(um%2)
				if(k==0):
					m1=0
					for i in range(10):
						for kl in range(10):
							m2=9*kl
							m3=m1
							for j in range(9):
								if(m3<9):
									#print(XA,XM,9*kl+j, 10*um+i, 9*kl+m3, 2*um)
									Res=((RRAM[XA][XM][9*kl+j][10*um+i])/19400)+((1-RRAM[XA][XM][9*kl+j][10*um+i])/7770000)
									Out[XA][XM][10*um+i]=(np.random.normal(1,Var)*Res*0.1*Pulse[XA][XM][9*kl+m3][2*um])+Out[XA][XM][10*um+i]
								if(m3>=9):
									#print(XA,XM,9*kl+j,10*um+i,m2,2*um+1)
									Res=((RRAM[XA][XM][9*kl+j][10*um+i])/19400)+((1-RRAM[XA][XM][9*kl+j][10*um+i])/7770000)
									Out[XA][XM][10*um+i]=(np.random.normal(1,Var)*Res*0.1*Pulse[XA][XM][m2][2*um+1])+Out[XA][XM][10*um+i]
									m2=m2+1
								m3=m3+1
						m1=m1+1

				if(k==1):
					m1=8
					for i in range(10):
						for kl in range(10):
							m2=8+(9*kl)
							m3=m1
							for j in range(9):
								#l1=(9*kl)+8-j
								l1=(9*kl)+8-j
								if(m3>=0):
									#print(XA,XM,l1, 10*um+i, 9*kl+m3, 2*um)
									Res=((RRAM[XA][XM][l1][10*um+i])/19400)+((1-RRAM[XA][XM][l1][10*um+i])/7770000)
									Out[XA][XM][10*um+i]=(np.random.normal(1,Var)*Res*0.1*Pulse[XA][XM][9*kl+m3][2*um])+Out[XA][XM][10*um+i]
								if(m3<=-1):
									#print(XA,XM,l1, 10*um+i, m2, 2*um+1)
									Res=((RRAM[XA][XM][l1][10*um+i])/19400)+((1-RRAM[XA][XM][l1][10*um+i])/7770000)
									Out[XA][XM][10*um+i]=(np.random.normal(1,Var)*Res*0.1*Pulse[XA][XM][m2][2*um+1])+Out[XA][XM][10*um+i]
									m2=m2-1
								m3=m3-1
						m1=m1-1
	return Out


def Multiply(RRAM,Pulse,BW,Var):
	
	N1=len(RRAM[0])
	X1,LA=9,0.03125

	Na=6 if (Var<0.3) else 8

	#Out=[[[[0 for i in range(180)] for j in range(N1)] for k in range(len(RRAM))] for l in range(Na)]

	#for i in range(Na):
	#	Out[i]=Top(RRAM,N1,Pulse,Var)

	Out=Parallel(n_jobs=Na)(delayed(Top)(RRAM,N1,Pulse,Var) for i in range(Na))
	
	M22=float((math.pow(2,X1)-1)/(90*0.1/19400))

	#if(BW<0):
	#	Output=[math.floor(i*M22*LA) for i in Out[0]]

	if(BW>=0):
		Output=[[[0 for i in range(180)] for j in range(N1)] for k in range(len(RRAM))]
		for i in range(Na):
			for j in range(len(RRAM)):
				for k in range(N1):
					for l in range(180):
						Output[j][k][l]=Output[j][k][l]+math.floor(Out[i][j][k][l]*M22)

		for XA in range(len(RRAM)):
			for k in range(N1):
				for i in range(180):
					Output[XA][k][i]=math.floor(Output[XA][k][i]*LA)

	return(Output)
