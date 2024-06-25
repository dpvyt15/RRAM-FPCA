
import math
import sys
sys.path.append('../')
import random
import numpy as np
import time
import cmath
import joblib
import VMMNew.RRAM_Programming
import VMMNew.GeneralFunctions
import VMMNew.InMemVMMP
import VMMNew.InMemVMM
from scipy import signal

#devices = tf.config.list_physical_devices()
#tf.debugging.set_log_device_placement(True)


def convolution(Input,Weight):
	x1=len(Input)-len(Weight)+1
	x2=len(Input[0])-len(Weight[0])+1
	I=[[sum(sum([[Input[i+a][j+b]*Weight[a][b] for b in range(len(Weight[0]))] for a in range(len(Weight))],[])) for j in range(x2)] for i in range(x1)]
	return I



def Top(Weight,B,RRAMRes=4,PulseRes=3,Split=1,jobs=-1):
	start_time=time.time()

	#print("In VMM")

	BN=[[(B[128*i+j] if(128*i+j<len(B)) else 0) for j in range(128)] for i in range(math.ceil(len(B)/128))]

	WeightN=[[[[(Weight[128*i+k][128*j+l] if (128*i+k<len(Weight) and 128*j+l<len(Weight[0])) else 0) for l in range(128)] for k in range(128)] for j in range(math.ceil(len(Weight[0])/128))] for i in range(math.ceil(len(Weight)/128))]

	
	WeightsW1N=[[[[0 for i in range(len(WeightN[0][0][0]))] for j in range(2*(len(WeightN[0][0])+1))] for k in range(len(WeightN[0]))] for l in range(len(WeightN))]
	WeightsW2N=[[[[0 for i in range(len(WeightN[0][0][0]))] for j in range(2*(len(WeightN[0][0])+1))] for k in range(len(WeightN[0]))] for l in range(len(WeightN))]
	DelWN=[[0 for k in range(len(WeightN[0]))] for l in range(len(WeightN))]
	DelW2N=[[0 for k in range(len(WeightN[0]))] for l in range(len(WeightN))]
	minN=[[0 for k in range(len(WeightN[0]))] for l in range(len(WeightN))]
	for i in range(len(WeightN)):
		for j in range(len(WeightN[0])):
			WeightsW1N[i][j],WeightsW2N[i][j],DelWN[i][j],DelW2N[i][j],minN[i][j]= VMMNew.RRAM_Programming.RRAMProgrammingMergeJ5(WeightN[i][j],RRAMRes,Split,0,jobs);

	end_time=time.time()
	print("Programming time:")
	print(end_time-start_time)
	
	start_time=time.time()
	#print("Done with VMM programming")
	OutputN=[[[0 for i in range(128)] for j in range(len(WeightN[0]))] for k in range(len(WeightN))]

	for i in range(len(WeightN)):
		for j in range(len(WeightN[0])):
			OutputN[i][j]=VMMNew.InMemVMM.VMMMultiResSplit(WeightsW1N[i][j],DelWN[i][j],DelW2N[i][j],BN[j],RRAMRes,PulseRes,minN[i][j],jobs)


	OutputFinal=[0 for i in range(len(Weight))]

	for i in range(len(OutputN)):
		for k in range(len(OutputN[0][0])):
			if(len(OutputN[0][0])*i+k<len(OutputFinal)):
				OutputFinal[len(OutputN[0][0])*i+k]=sum([OutputN[i][j][k] for j in range(len(OutputN[0]))])
			
	#print("Exit VMM")
	
	end_time=time.time()
	print("Processing time:")
	print(end_time-start_time)

	return OutputFinal


def main():
	start_time=time.time()
	RRAMRes=4
	PulseRes=3
	Split=1

	Weight=[[random.uniform(-1,1) for i in range(400)] for j in range(120)]
	B=[random.uniform(0,1) for i in range(400)]

	minB=min(B)
	B=[i-minB for i in B]

	OutNew=[sum([Weight[i][j]*B[j] for j in range(len(B))]) for i in range(len(Weight))]
	Output=Top(Weight,B,RRAMRes,PulseRes,Split)

	Error=0
	Sum=0

	for i in range(len(OutNew)):
		Error=Error+(abs(OutNew[i]-Output[i])*100)
		Sum=abs(OutNew[i])+Sum

	Error=Error/Sum
	
	print("Expected Output:")
	print(OutNew)
	print("Generated Output:")
	print(Output)
	print("Error:")
	print(Error)
	
	end_time=time.time()
	print(end_time-start_time)

	


if __name__ =="__main__":
	main()

