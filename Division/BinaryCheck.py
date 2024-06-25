
import numpy as np
import math
import sys
import random
import time
import cmath

"""converts a fractional number into binary equivalent. support functions"""

def dec2bin(A):
	B=[]
	remainder=A
	while remainder>0:
		B.append(int(remainder%2))
		remainder=int(remainder/2)
	
	P=len(B)
	return(B,P)


def frac2bin(A,P,BW):

	B=[]
	remainder=A

	while(len(B)<BW-P):

		x1=remainder*2
		B.append(int(x1))
		remainder=x1-int(x1)

	
	B.reverse()

	return(B)


def fractions(B,BW,M):

	ID,P=dec2bin(B)
	x=P
	if(M>=0):
		i=P
		while(i<M):
			ID.append(0)
			i=i+1
		x=M
	
	inp=B-int(B)
	frac=frac2bin(inp,x,BW)

	tot=[]
	for i in frac:
		tot.append(i)

	for i in ID:
		tot.append(i)

	Sum=0

	for i in range(len(tot)):
		Sum=Sum+(math.pow(2,i)*tot[i])	

	Sum=Sum/math.pow(2,BW-M)

	return(tot,P)



def CalMax(A,B):

	A1,P1=dec2bin(A)
	B1,P2=dec2bin(B)

	Max=max(P1,P2)	
	
	return Max
