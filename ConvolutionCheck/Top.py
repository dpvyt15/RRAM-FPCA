
import math
import sys

sys.path.append('../')

import random
import numpy as np
import time
import cmath
import joblib
import ConvolutionCheck.Programming
import ConvolutionCheck.GeneralFunctions
import ConvolutionCheck.ConvoInMem
import ConvolutionCheck.ConvInMemP
from scipy import signal
import cProfile

#devices = tf.config.list_physical_devices()
#tf.debugging.set_log_device_placement(True)


""""This is the top module for low-accuracy convolution for neural networks. Top function performs the convolution operation within the 90*180 PS arrays. It takes the IFM, Weights, device resolution, input resolution, number of times that the weight matrix needs to be split as inputs. The IFMs and weight matrices are split into smaller portion that can be processed within the PS arrays. Simply running this script performs convolutions between 10*(32*32) IFMs and 120*10*(3*3) weights to generate 120*(30*30) OFMs."""""

def convolution(Input,Weight):
	x1=len(Input)-len(Weight)+1
	x2=len(Input[0])-len(Weight[0])+1
	I=[[sum(sum([[Input[i+a][j+b]*Weight[a][b] for b in range(len(Weight[0]))] for a in range(len(Weight))],[])) for j in range(x2)] for i in range(x1)]
	return I


def Top(Weight,B,RRAMRes=4,PulseRes=3,Split=1,jobs=-1):
	start_time=time.time()
	#print("In Conv")
	WeightsW1,WeightsW2,DelW,DelW2,min= ConvolutionCheck.Programming.RRAMProgrammingMergeJ5(Weight,RRAMRes,Split,1,jobs);
	end_time=time.time()
	print("Time required to program the PS arrays with weights:{0}".format(end_time-start_time))

	start_time=time.time()
	
	Output=ConvolutionCheck.ConvoInMem.ConvolutionInMem(WeightsW1,DelW,WeightsW2,DelW2,B,min,PulseRes,RRAMRes,jobs)

	end_time=time.time()
	print("Time taken to perform the convolutions within the PS arrays, along with post-processing:{0}".format(end_time-start_time))

	return Output


def main():
	start_time=time.time()
	RRAMRes=6
	PulseRes=6
	Split=2

	Weight=[[[[random.uniform(-1,1) for i in range(3)] for j in range(3)] for k in range(10)] for l in range(120)]
	B=[[[random.uniform(-1,1) for i in range(32)] for j in range(32)] for k in range(10)]

	OutNew=[[[[0 for i in range(30)] for j in range(30)] for k in range(len(B))] for l in range(len(Weight))] 
	Output1=Top(Weight,B,RRAMRes,PulseRes,Split)

	for i in range(len(Weight)):
		for j in range(len(Weight[0])):
			OutNew[i][j]=convolution(B[j],Weight[i][j])

	
	OutNewX=[[[sum([OutNew[i][j][k][l] for j in range(len(OutNew[0]))]) for l in range(len(OutNew[0][0][0]))] for k in range(len(OutNew[0][0]))] for i in range(len(OutNew))]

	ErrorA=0
	

	for i in range(len(OutNewX)):
		Error=0
		Sum=0
		for j in range(len(OutNewX[0])):
			for k in range(len(OutNewX[0][0])):
				Error=Error+(abs(OutNewX[i][j][k]-Output1[i][j][k])*100)
				Sum=abs(OutNewX[i][j][k])+Sum

		ErrorA=(Error/Sum)+ErrorA
	
	print("Expected Output:")
	print(OutNewX)
	
	print("Simulated Output:")
	print(Output1)

	ErrorA=ErrorA/len(OutNewX)

	print("\n")
	print("Device resolution:{0},Pulse resolution:{1}, Kernel size:{2}, IFM size: {3}, Output Error:{4} %".format(RRAMRes,PulseRes,"120*10*3*3","10*32*32",ErrorA))
	
	
	end_time=time.time()
	print("Total time taken for convolution:{0}".format(end_time-start_time))

	


if __name__ =="__main__":
	main()
	#cProfile.run('main()')

