
import numpy as np
import math
import sys
sys.path.append('../')
import random
import time
#import ImgShuffle
import joblib
from keras.datasets import cifar10
import ImgShuffle_CIFAR10
import FP_D3
import BP_D3
import VMMNew.RRAM_Programming
import ConvolutionCheck.Programming
from copy import deepcopy
import Initialization2



#devices = tf.config.list_physical_devices()
#tf.debugging.set_log_device_placement(True)


"""50 layer dense-net for CIFAR-10 database using CPU/GPU. This is the top module that calls the different scripts for different tasks."""



def add(in1,in2):
	u=[(in1[i]+in2[i]) for i in range(len(in1))]
	return u


def accu(input,output):
	u=1 if input[output]==max(input) else 0
	return u

def main():
	start_time = time.time()
	RRAMRes,PulseRes,Split=4,3,1
	Ni1,Ni2,Ni3=5,3,3
	TotalTrain=400
	TotalValid=100
	TotalTest=100
	RGB=3
	L1O=6
	L2O=16
	L30=120

	L=[3,16,28,40,52,64,76,88,12,24,36,48,60,72,84,96,108,120,132,144,156,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252,264,276,288]
	O=[16,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12]
	KS=[3,3,3,3,3,3,3,1,3,3,3,3,3,3,3,3,3,3,3,3,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
	MP=[2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
	DL=[0,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
	FCI=[192,1000]
	FCO=[1000,10]
	
	poolsize=[]
	f1=[]
	for i in range(len(L)):
		f1.append([KS[i],KS[i],O[i],L[i]])
		poolsize.append([MP[i],MP[i]])

	
	FCC=[]
	for i in range(len(FCI)):
		FCC.append([FCI[i],FCO[i]])
	
	start_time1=time.time()
	TI,TO,VI,VO,ti,to,cw,cb,lw,lb=Initialization2.Initialization(f1,poolsize,FCC)
	gamma,beta,RunMean,RunVar=Initialization2.BN(L)
	minibatch,eta,lmbda,epochs=10,float(0.08),float(5),5
	M1,M2=float(eta)/minibatch,0.99995
	TrainBatches=int(TotalTrain/minibatch)

	end_time1=time.time()
	print("Unpacking time: {0}".format(end_time1-start_time))

	
	for Epo in range(epochs):
		start_time2= time.time()
		count=0
		for k in range(TrainBatches):
			print("Epoch: {0}, TrainBatch: {1}".format(Epo,k))

			x2,x21,y1,g1,XXX,o1,a1,mean,variance,RunMean,RunVar=FP_D3.TopInference(TI[k*minibatch:(k+1)*minibatch],cw,cb,lw,lb,poolsize,DL,gamma,beta,1,RunMean,RunVar)
			Cw,Cb,Lw,D,dGamma,dBeta=BP_D3.TopBackPropNew(a1,o1,lw,XXX,g1,y1,x2,x21,cw,TI[k*minibatch:(k+1)*minibatch],TO[k*minibatch:(k+1)*minibatch],poolsize,gamma,beta,mean,variance,DL)

			lb=[[(lb[i][j]-(D[i][j]*M1)) for j in range(len(lb[i]))] for i in range(len(lb))]
			lw=[[[(lw[i][j][k]*M2)-(Lw[i][j][k]*M1) for k in range(len(lw[i][j]))] for j in range(len(lw[i]))] for i in range(len(lw))]
			cw=[[[[[(cw[i][j][k][l][m]*M2)-(Cw[i][j][k][l][m]*M1) for m in range(len(cw[i][j][k][l]))] for l in range(len(cw[i][j][k]))] for k in range(len(cw[i][j]))] for j in range(len(cw[i]))] for i in range(len(cw))] 
			cb=[[cb[i][j]-(Cb[i][j]*M1) for j in range(len(cb[i]))] for i in range(len(cb))]
			gamma=[[gamma[xa][i]-(M1*dGamma[xa][i]) for i in range(len(gamma[xa]))] for xa in range(len(gamma))]
			beta=[[beta[xa][i]-(M1*dBeta[xa][i]) for i in range(len(beta[xa]))] for xa in range(len(beta))]
			
		Accuracy=0
		for ick in range(int(TotalValid/minibatch)):
			x2,x21,y1,g1,XXX,o1,a1,mean,variance,RunMean,RunVar=FP_D3.TopInference(VI[minibatch*ick:minibatch*(ick+1)],cw,cb,lw,lb,poolsize,DL,gamma,beta,0,RunMean,RunVar)
			for x in range(minibatch):
				Accuracy=Accuracy+accu(a1[x][len(a1[x])-1],VO[minibatch*ick+x])
		print("Epoch {0}: Validation accuracy {1}".format(Epo, Accuracy))
		#print(Accuracy)
		TI,TO=ImgShuffle_CIFAR10.Shuffle(TI,TO)
		VI,VO=ImgShuffle_CIFAR10.Shuffle(VI,VO)
		elapsed_time_secs2 = time.time() - start_time2
		print("Epoch time:")
		print(elapsed_time_secs2)

	Accuracy=0
	for i in range(int(TotalTest/minibatch)):
		x2,x21,y1,g1,XXX,o1,a1,mean,variance,RunMean,RunVar=FP_D3.TopInference(ti[minibatch*i:minibatch*(i+1)],cw,cb,lw,lb,poolsize,DL,gamma,beta,0,RunMean,RunVar)
		for x in range(minibatch):
			Accuracy=Accuracy+accu(a1[x][len(a1[x])-1],to[minibatch*i+x])
		#Accuracy=Accuracy+accu(a1[len(a1)-1],to[i])
	print("Test accuracy {0}".format(Accuracy))

	elapsed_time_secs = time.time() - start_time
	print("Total Time:")
	print(elapsed_time_secs)

if __name__ =="__main__":
	random.seed(4)
	main()
