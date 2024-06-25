
#!/usr/bin/python
import math
import sys
sys.path.append('../')
import random
import Multiplication.J4_RRAMProgramming_MultForDiv
import Multiplication.J4_Pulse_MultForDiv
import Multiplication.J4_Mult_MultForDiv
import Multiplication.J4_PostProcess_MultForDiv
import Multiplication.CheckIt
import time
import Multiplication.BinaryCheck
from joblib import Parallel, delayed

""" This is the top module. Takes in inputs, converts them into binary representation, then sends the binary data to planar staircase array tiles. Receives the outputs and conveys them to the external output.
Top module performs the multiplication. It takes the two operands, bit-width (can be anything, the higher the better the results) and variability (0-1). Simply running this script performs the multiplication for different bit-widths at different variability and dumps out the error for each case. """


def MultF(aB1,aB2,BW,Var):


	if(BW<16):

		XN1=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		x=int(math.pow(2,BW)-1)
		N1=int(math.log2(BW))
		Tel1=XN1[N1]

		Arr=[[[0 for i in range(BW)] for j in range(2)] for k in range(Tel1)]

		#print("In Mult for div")
		#print(aB1,aB2,BW)

		for k in range(BW):
			Arr[0][0][k]=aB1[k]
			Arr[0][1][k]=aB2[k]

		RRAM=Multiplication.J4_RRAMProgramming_MultForDiv.ArrayDef(Arr,Tel1,BW)
		Pulses=Multiplication.J4_Pulse_MultForDiv.ArrayPulses(Arr,Tel1,BW)
		Output=Multiplication.J4_Mult_MultForDiv.Multiply(RRAM,Pulses,BW,Var)
		sedOutput=Multiplication.J4_PostProcess_MultForDiv.Add(BW, Tel1, Output)

		ProcessedOutput=sedOutput[0]*(2**(1-2*BW))

	if(BW>=16):
		ProcessedOutput=Multiplication.CheckIt.SplitInputs(BW,aB1,aB2,Var)
	
	return(ProcessedOutput)


def checkIt(A,BW):
	X=[0 for i in range(BW)]
	for i in range(len(A)):
		X[i]=A[i]
	return X





def Top(A,B,BW=32,Var=0):
	PA=Multiplication.BinaryCheck.CalMax(A,B)
	aB1,P1=Multiplication.BinaryCheck.fractions(A,BW,PA)
	aB2,P2=Multiplication.BinaryCheck.fractions(B,BW,PA)
	Out=MultF(aB1,aB2,BW,Var)
	FinalOut=Out*(2**(2*PA-1))
	return FinalOut

	
def main():

	Test=1000
	for a in range(2,11):
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
				X=A*B
				Error+=abs(C1-X)*100/X

			print("The multiplication error for bit width of {0} and {1}% variability is: {2}%".format(BW1, Var*100, Error/Test))
	
	return



if __name__ =="__main__":
	main()
