
import numpy as np
import math
import sys
sys.path.append('../')

import random
import time
import cmath
import Multiplication.PSMult
import Division.MHAdder
import Division.BinaryCheck



"""Control unit for division. sends data to appropriate adder and multiplier modules to get the output. Top is the module that performs the division. It takes in Dividend (DD), divisor (B), bit-width (any size) and device variability (0-1) as inputs and generates the output. It first calculates the reciprocal value of B, then multiplies this reciprocal with the divisor to get the output."""


def bitWidth(A):
	P=0
	for i in range(len(A)-1,-1,-1):
		if(A[i]==1):
			P=i+1
			break
	return(P)

def bitWidth2(A):
	P=0
	for i in range(len(A)):
		if(A[i]==1):
			P=i+1
			break
	return(P)

def Complement2(A,Var):
	D=[1-i for i in A]
	carry=[0 for i in A]
	carry[0]=1
	TBP=Division.MHAdder.ArrayProgramF(carry,len(D),Var)
	Com,NewCarry=Division.MHAdder.AddF(carry,D,len(D),Var,TBP)
	Com2=[]
	for i in Com:
		Com2.append(i)		
	Com2.append(1)

	return(Com2)


def shift(A):
	D=[0 for i in A]
	for i in range(0,len(A)-1):
		D[i+1]=A[i]

	return(D)




def Top(DD,B,BW=32,Var=0):

	ID,P=Division.BinaryCheck.fractions(B,BW,-1)
	Div=Complement2(ID,Var)
	A=[0 for i in range(BW+1)]
	A[len(A)-1]=1
	Quo=[]
	TBP=Division.MHAdder.ArrayProgramF(Div,len(Div),Var)

	i=0
	while i<(BW):
		Sum,carry=Division.MHAdder.AddF(Div,A,len(Div),Var,TBP)
		if(carry==1):
			A=shift(Sum)
		elif(carry==0):
			A=shift(A)
		i=i+1
		Quo.append(carry)

		B1=bitWidth(A)
		B2=bitWidth2(A)
		if(B1==B2):
			ind=A.index(1)
			for j in range(len(A)-1-ind):
				Quo.append(0)
			break
	
		else:
			if(B1<len(A)-1):
				for j in range(len(A)-2-(B1-1)):
					if(i<BW):
						A=shift(A)
						Quo.append(0)
						i=i+1

	Dividend,P2=Division.BinaryCheck.fractions(DD,BW,-1)
	l1=len(Quo)
	i=0
	for j in range(BW-l1):
		Quo.append(Quo[i])
		i=i+1

	Quo.reverse()
	#Reci=0
	#for ux in range(len(Quo)):
	#	Reci=Reci+(Quo[ux]*math.pow(2, ux-(P+len(Quo)-1)))

	Quo2=Quo

	m2=BW-len(Quo)
	if(m2<0):
		Quo2=[Quo[i-m2] for i in range(len(Quo)+m2)]

	Quo=Quo2

	Output=Multiplication.PSMult.MultF(Dividend,Quo,BW,Var)
	Out=Output*(math.pow(2,-(P+len(Quo)-P2-BW)))

	return(Out)


def main():

	Test=1000
	for a in range(3,11):
		BW1=int(math.pow(2,a))
		BW=16 if BW1<16 else BW1
		for b in range(10):
			Error=0
			Var=0.1*b
			for j in range(Test):

				A=random.uniform(0,pow(2,BW))
				B=random.uniform(0,pow(2,BW))
				while(B==0 or B==1):
					B=int(random.uniform(0,pow(2,BW)))

				while(A==0 or A==1):
					DD=int(random.uniform(0,pow(2,BW)))
				C1=Top(A,B,BW,Var)
				X=float(float(A)/B)
				Error+=abs(C1-X)*100/X

			print("The division error for bit width of {0} and {1}% variability is: {2}%".format(BW1, Var*100, Error/Test))
	
	return



if __name__ =="__main__":
	main()


		

		