#!/usr/bin/python
import math
import sys
sys.path.append('../')

import random
import Adder.BinaryCheck


""""Compares two different numbers -A and B- and determines if A>B, A<B or A=B. Uses the MIAMI technique to program the array and DICP to determine the output and mitigate device issues. Running this script generates 1000 sample decimal numbers each for bit-widths from 0 to 11. The overall output error is then given out. If we want to compare 2 numbers without running the main function, then Top is the top function. It takes 4 parameters - the two operands, bit-width for the general operation (>= bit-width of the larger number), device-to-device variability""""" 


def bitWidth(A):
	P=0
	for i in range(len(A)-1,-1,-1):
		if(A[i]==1):
			P=i+1
			break
	return(P)


def Top(A,B,BW=32,Var=0):

	
	PA=Adder.BinaryCheck.CalMax(A,B)
	aB1,P1=Adder.BinaryCheck.fractions(A,BW,PA)
	aB2,P2=Adder.BinaryCheck.fractions(B,BW,PA)

	Out,OutA=Compare(aB1,aB2,BW,Var)

	C1,C2=DICP(Out,OutA,BW)

	return C1,C2



def DICP(Out,OutA,BW):
	
	OutN=[(1 if i>0 else 0) for i in Out]
	Out2N=[(1 if i>0 else 0) for i in OutA]
	
	F1=OutN[0]
	F2=Out2N[0]

	P2=1

	for i in range(BW):
		P11=OutN[i]
		P12=Out2N[i]
		F1=(P2 if F1==1 else P11)
		F2=(P2 if F2==1 else P12)

	C1=1 if(F1==1 or F2==1) else 0

	C2=0

	if(C1==1):
		U1=bitWidth(OutN)
		U2=bitWidth(Out2N)
		if(U1>U2):
			C2=1

	return C1,C2


def Compare(A,B,BW,Var):
	x1=math.ceil(BW/8)
	
	Out=[]
	OutA=[]

	for i in range(x1):
		A1=A[min(8*i,BW-1):min(8*(i+1),BW)]
		B1=B[min(8*i,BW-1):min(8*(i+1),BW)]

		Out.append(Program(A1,B1,Var))
		OutA.append(Program(B1,A1,Var))

	

	Out2=[]
	Out2A=[]
	for i in range(x1):
		for j in range(8):
			if(8*i+j<BW):
				Out2.append(Out[i][j])
				Out2A.append(OutA[i][j])

	return(Out2,Out2A)

	


def ReRAM_NewC(V_t,dt,init):

	if(V_t>0):
		final=0.99
	if(V_t<0):
		final=0.1
	if(V_t==0):
		final=init
	return final

def Program(A,B,Var):
	
	Arr=[[0.1 for i in range(8)] for j in range(8)]

	W1,W2,B1,B2=[],[],[],[]

	for i in range(8):

		if(i%2==0):
			if(len(A)>i):
				W1.append(A[i]*1.5)
				B1.append(B[i]*1.5)

			else:
				W1.append(0)
				B1.append(0)
			W2.append(0)
			B2.append(0)

		if(i%2==1):
			if(len(A)>i):
				W2.append(A[i]*1.5)
				B2.append(B[i]*1.5)

			else:
				W2.append(0)
				B2.append(0)

			W1.append(0)
			B1.append(0)

	Arr=[[(ReRAM_NewC(W1[i]-B1[j],100E-9,Arr[i][j]) if(i%2==0 and j%2==0) else Arr[i][j]) for j in range(8)] for i in range(8)]
	Arr=[[(ReRAM_NewC(W2[i]-B2[j],100E-9,Arr[i][j]) if(i%2==1 and j%2==1) else Arr[i][j]) for j in range(8)] for i in range(8)]

	W1,W2=[],[]

	for i in range(8):
		if(i<4):
			W1.append(-0.6)
			W2.append(0)

		if(i>=4):
			W1.append(0)
			W2.append(-0.6)	

	Arr=[[(ReRAM_NewC(W1[i],100E-9,Arr[i][j]) if(i<4 and j>=4) else Arr[i][j]) for j in range(8)] for i in range(8)]
	Arr=[[(ReRAM_NewC(W2[i],100E-9,Arr[i][j]) if(i>=4 and j<4) else Arr[i][j]) for j in range(8)] for i in range(8)]


	W1,W2=[],[]

	for i in range(8):
		if(i%4<2):
			W1.append(-0.6)
			W2.append(0)


		if(i%4>=2):
			W1.append(0)
			W2.append(-0.6)	

	Arr=[[(ReRAM_NewC(W1[i],100E-9,Arr[i][j]) if((i<2 or 4<=i<6) and (2<=j<4 or 6<=j<8)) else Arr[i][j]) for j in range(8)] for i in range(8)]

	Arr=[[(ReRAM_NewC(W2[i],100E-9,Arr[i][j]) if((2<=i<4 or 6<=i<8) and (j<2 or 4<=j<6)) else Arr[i][j]) for j in range(8)] for i in range(8)]

	Arr=[[7770000 if i==0.1 else 19400 for i in j] for j in Arr]


	Out=[sum([random.uniform(1-Var,1+Var)*0.1/Arr[i][j] for j in range(8)]) for i in range(8)]

	LX1=[7, 8, 8, 8, 9, 9]

	XN1=int(math.log(8,2))-3

	X1=LX1[XN1]

	M22=float((math.pow(2,X1)-1)/(8*0.1/19400))

	FR2=[(int(i*M22)) for i in Out]

	return FR2
	
			

	



def main():

	Test=1000
	for a in range(11):
		BW=int(math.pow(2,a))
		for b in range(10):
			Accu=0
			Var=0.1*b
			for j in range(Test):

				Max=int(math.pow(2,a))
				A=random.randint(0,Max)
				B=random.randint(0,Max)
				C1,C2=Top(A,B,BW,Var)

				if(A==B and C1==0):
					Accu=Accu+1
	
				if(C1==1 and A!=B):
					if(C2==1 and A>B):
						Accu=Accu+1
					if(C2==0 and A<B):
						Accu=Accu+1

			print("The output error for bit width of {0} and {1}% variability is: {2}%".format(BW, Var*100, (Test-Accu)*100/Test))
	
	return


	


	


if __name__ =="__main__":
	main()
