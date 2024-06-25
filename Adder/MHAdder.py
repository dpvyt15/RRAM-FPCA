#!/usr/bin/python
import math
import sys
sys.path.append('../')

import random
import Adder.BinaryCheck


"""Performs in memory addition. Top is the top module and acts as the control unit. It takes 2 numbers, the bitwidth of execution, RRAM device to device variability as inputs. converts numbers into binary equivalents and passes the data to AddF2. 
AddF2 takes the binary inputs and maps it to 8x8 Manhattan (MH) RRAM arrays. 
ArrrayFunction2 maps operand A to the RRAM conductance and then applies pulses to the crossbar. The crossbar currents are converted to digital equivalents using ADCs here. At the end of processing cycles, the ADC outputs are DICP'd and fed to 2:1 MUXes to get the output."""


def ArrayFunction2(N,A,B,BW,Var):
	l1=math.ceil(BW/N)
	Outs=N*4
	Ins=N*3
	TBP=[[[0 for i in range(Outs)] for j in range(Ins)] for k in range(l1)]
	TBP2=[[0 for i in range(Ins)] for k in range(l1)]
	for i in range(l1):
		for j in range(N):
			if(((N*i)+j)<BW):
				TBP[i][3*j][4*j+1]=1
				TBP[i][3*j+1][4*j+1]=1
				TBP[i][3*j][4*j]=A[(N*i)+j]
				TBP[i][3*j][4*j+3]=A[(N*i)+j]
				TBP[i][3*j][4*j+2]=1-A[(N*i)+j]
				TBP[i][3*j+2][4*j+2]=A[(N*i)+j]
				TBP[i][3*j+2][4*j+3]=1-A[(N*i)+j]
	TBP=[[[(19400*i)+((1-i)*7770000) for i in j] for j in k] for k in TBP]
	for i in range(l1):
		for j in range(N):
			if(((N*i)+j)<BW):
				TBP2[i][3*j]=B[(N*i)+j]
				TBP2[i][3*j+1]=A[(N*i)+j]
				TBP2[i][3*j+2]=1-TBP2[i][3*j]
	FR=[[[((random.uniform(1-Var,1+Var)*0.1*TBP2[i][j])/TBP[i][j][k]) for k in range(Outs)] for j in range(Ins)] for i in range(l1)]
	FR2=[[0 for i in range(Outs)] for j in range(l1)]
	for i in range(l1):
		for j in range(Outs):
			for k in range(Ins):
				FR2[i][j]=FR2[i][j]+FR[i][k][j]

	LX1=[7, 8, 8, 8, 9, 9]

	XN1=int(math.log(Outs,2))-3

	X1=LX1[XN1]

	M22=float((math.pow(2,X1)-1)/(Outs*0.1/19400))

	FR2=[[(int(i*M22)) for i in k] for k in FR2]

	FR2=[[(1 if i>0 else 0) for i in k] for k in FR2]

	T1=[0 for i in range(BW+1)]
	Cin=0
	Result=0
	
	for i in range(l1):
		for j in range(N):
			mn=(N*i)+j
			if(mn<BW):
				T1[mn] = FR2[i][4*j+2] if(Cin==0)  else FR2[i][4*j+3]
				Cin = FR2[i][4*j] if(Cin==0) else FR2[i][4*j+1]
				Result=Result+(T1[mn]*(2**mn))
	T1[BW]=Cin
	Result=Result+(Cin*(2**BW))
	return(T1)


def AddF2(aB1,aB2,BW,Var):

	Arr=3
	Outs=math.pow(2,Arr)
	Nums=int(Outs/4)
	Ins=Nums*3
	Out=ArrayFunction2(Nums,aB1,aB2,BW,Var)	
	Sum=Out[0:len(Out)-1]
	Carry=Out[len(Out)-1]
	return(Sum,Carry)



def Top(A,B,BW=32,Var=0):

	#if(A-int(A)!=0 or B-int(B)!=0):
	PA=Adder.BinaryCheck.CalMax(A,B)
	aB1,P1=Adder.BinaryCheck.fractions(A,BW,PA)
	aB2,P2=Adder.BinaryCheck.fractions(B,BW,PA)
	Sum,Carry=AddF2(aB1,aB2,BW,Var)
	
	Out=0
	for i in range(len(Sum)):
		Out=Out+(Sum[i]*(2**i))
	
	Out=Out+(Carry*(2**BW))

	FinalOut=Out*(2**-(BW-PA))
	return FinalOut


def main():

	Test=1000
	for a in range(2,10):
		BW1=int(math.pow(2,a))
		BW=16 if BW1<16 else BW1
		for b in range(10):
			Error=0
			Var=0.1*b
			for j in range(Test):

				A=random.randint(0,pow(2,BW))
				B=random.randint(0,pow(2,BW))
				while(A==0):
					A=random.randint(0,pow(2,BW))

				while(B==0):
					B=random.randint(0,pow(2,BW))

				C1=Top(A,B,BW,Var)
				X=A+B
				Error+=abs(C1-X)*100/X

			print("The multiplication error for bit width of {0} and {1}% variability is: {2}%".format(BW1, Var*100, Error/Test))
	
	return



if __name__ =="__main__":
	main()

			
			
