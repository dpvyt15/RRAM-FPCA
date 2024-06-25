
import numpy as np
import math
import sys
import random
import time
import cmath

"""Determines the mantissa and exponent for FP calculations. Also converts mantissa to decimal notation. The binary notations of the decimal numbers (floating-point) are used for multiplication, addition and division operations.
fractions is the top module. It takes a floating-point number, expected bit-width and the bit-width of a bigger number. If we are trying to operate on two numbers with different bit-widths, we can append zeros (using M) to make them the same size and change the exponent. """


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
	i=P
	while(i<M):
		ID.append(0)
		i=i+1
	
	inp=B-int(B)
	frac=frac2bin(inp,M,BW)

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
