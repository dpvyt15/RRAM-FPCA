#!/usr/bin/python
import math
import sys
import random
import numpy as np


#######################Code#############################################################
def Multiply(RRAM,Pulse,BW,Var):

	N1=1 if BW<64 else 2
	Out=[[0 for i in range(180)] for j in range(N1)]


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
								Res=((RRAM[XM][9*kl+j][10*um+i])/19400)+((1-RRAM[XM][9*kl+j][10*um+i])/7770000)
								Out[XM][10*um+i]=(random.uniform(1-Var,1+Var)*Res*0.1*Pulse[XM][2*um][9*kl+m3])+Out[XM][10*um+i]
							if(m3>=9):
								Res=((RRAM[XM][9*kl+j][10*um+i])/19400)+((1-RRAM[XM][9*kl+j][10*um+i])/7770000)
								Out[XM][10*um+i]=(random.uniform(1-Var,1+Var)*Res*0.1*Pulse[XM][2*um+1][m2])+Out[XM][10*um+i]
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
								Res=((RRAM[XM][l1][10*um+i])/19400)+((1-RRAM[XM][l1][10*um+i])/7770000)
								Out[XM][10*um+i]=(random.uniform(1-Var,1+Var)*Res*0.1*Pulse[XM][2*um][9*kl+m3])+Out[XM][10*um+i]
							if(m3<=-1):
								Res=((RRAM[XM][l1][10*um+i])/19400)+((1-RRAM[XM][l1][10*um+i])/7770000)
								Out[XM][10*um+i]=(random.uniform(1-Var,1+Var)*Res*0.1*Pulse[XM][2*um+1][m2])+Out[XM][10*um+i]
								m2=m2-1
							m3=m3-1
					m1=m1-1


	#X1=1+((math.log(90,2)))

	X1=9

	M22=float((math.pow(2,X1)-1)/(90*0.1/19400))

	#print(Out)

	if(BW<64):
		Output=[math.floor(i*M22*0.18) for i in Out[0]]

	if(BW==64):
		Output=[[0 for i in range(180)] for j in range(2)]

		for k in range(N1):
			for i in range(180):
				Output[k][i]=math.floor(Out[k][i]*M22*0.18)

	#print(Output)

	return(Output)
