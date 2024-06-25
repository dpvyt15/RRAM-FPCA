
import sys

sys.path.append('../')

import ConvolutionCheck.Top
import Adder.MHAdder
import Multiplication.PSMult
import Division.Divison
import VMMNew.Top
import Compare.CMP
import random



def main():
	
	command=input("What is the instruction you wish to execute? Options: Multiplication, Division, Compare, Add, VMM, Convolution\n")
	inputP=input("Do you want to provide inputs? 0 for no, 1 for yes\n")
	
	if(command=="Multiplication"):
		if(inputP=="1"):
			A=float(input("Enter input1\n"))
			B=float(input("Enter input2\n"))
			Out=Multiplication.PSMult.Top(A,B,BW=32,Var=0)
			print("Output:{0}".format(Out))

		if(inputP=="0"):
			Multiplication.PSMult.main()


	if(command=="Division"):
		if(inputP=="1"):
			A=float(input("Enter Dividend\n"))
			B=float(input("Enter Divisor\n"))
			Out=Division.Divison.Top(A,B,BW=64,Var=0)
			print("Output:{0}".format(Out))

		if(inputP=="0"):
			Division.Divison.main()


	if(command=="Compare"):
		if(inputP=="1"):
			A=float(input("Enter input1\n"))
			B=float(input("Enter input2\n"))
			C1,C2=Compare.CMP.Top(A,B,BW=32,Var=0)
			if(C1==0):
				print("input1==input2")
			if(C1==1):
				if(C2==1):
					print("input1>input2")
				if(C2==0):
					print("input1<input2")

		if(inputP=="0"):
			Compare.CMP.main()


	if(command=="Add"):
		if(inputP=="1"):
			A=float(input("Enter input1\n"))
			B=float(input("Enter input2\n"))
			Out=Adder.MHAdder.Top(A,B,BW=32,Var=0)
			print("Output:{0}".format(Out))

		if(inputP=="0"):
			Adder.MHAdder.main()


	if(command=="Convolution"):
		if(inputP=="1"):
			print("Weights should be provided as an array in this format: (number of output images)*(number of input images)*(kernel rows)*(kernel columns)")
			k1=int(input("Enter kernel columns\n"))
			k2=int(input("Enter number of input images\n"))
			k3=int(input("Enter number of output images\n"))
			A=[[[[0 for i in range(k1)] for j in range(k1)] for k in range(k2)] for l in range(k3)]
			for i in range(k3):
				for j in range(k2):
					for k in range(k1):
						for l in range(k1):
							A[i][j][k][l]=float(input("Enter a number\n"))

			print("IFMs should be provided as an array in this format: (number of input images)*(IFM rows)*(IFM columns)")
			k4=int(input("Enter IFM columns\n"))
			B=[[[0 for i in range(k4)] for j in range(k4)] for k in range(k2)]
			for i in range(k2):
				for j in range(k4):
					for k in range(k4):
						B[i][j][k]=float(input("Enter a number\n"))
			Out=ConvolutionCheck.Top.Top(A,B,RRAMRes=6,PulseRes=6,Split=2)
			print("Weights:{0}".format(A))
			print("IFM:{0}".format(B))
			print("Output:{0}".format(Out))

		if(inputP=="0"):
			ConvolutionCheck.Top.main()


	if(command=="VMM"):
		if(inputP=="1"):
			print("Weights should be provided as an array in this format: (number of output neurons)*(number of input neurons)")
			k1=int(input("Enter number of input neurons\n"))
			k2=int(input("Enter number of output neurons\n"))

			A=[[0 for i in range(k1)] for j in range(k2)]
			for i in range(k2):
				for j in range(k1):
					A[i][j]=float(input("Enter a number\n"))

			print("IFMs should be provided as an array in this format: (number of input neurons)")

			B=[0 for i in range(k1)]
			for i in range(k1):
				B[i]=float(input("Enter a number\n"))
			
			Out=VMMNew.Top.Top(A,B,RRAMRes=6,PulseRes=6,Split=2)
			print("Weights:{0}".format(A))
			print("input:{0}".format(B))
			print("Output:{0}".format(Out))

		if(inputP=="0"):
			VMMNew.Top.main()


	return	


if __name__ =="__main__":
	main()
