
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
import FP_InMem2
import BP_NN
import VMM.RRAM_Programming
import Convolution.RRAM_Programming


#devices = tf.config.list_physical_devices()
#tf.debugging.set_log_device_placement(True)


def Separate(input,output):
    TIX=[[[[float(input[l,j,i,k]/255) for i in range(len(input[0,0]))] for j in range(len(input[0]))] for k in range(len(input[0,0,0]))] for l in range(len(input))]
    TOX=[output[i,0] for i in range(len(input))]
    return TIX,TOX

def Separation(input,output):
    TI=[[[[input[l][k][j][i] for i in range(len(input[0][0][0]))] for j in range(len(input[0][0]))] for k in range(len(input[0]))] for l in range(40000)]
    TO=[output[i] for i in range(40000)]
    VI=[[[[input[l][k][j][i] for i in range(len(input[0][0][0]))] for j in range(len(input[0][0]))] for k in range(len(input[0]))] for l in range(40000,50000)]
    VO=[output[i] for i in range(40000,50000)]
    return TI,TO,VI,VO

def weight_init_conv(f,other):
    X=float((f[1]*f[0]*f[2])/(other[0]*other[1]))                                                            #calculates the normalization factor for the weights
    weights=[[[[float(random.gauss(0,1))*math.sqrt(1/X) for i in range(f[0])] for j in range(f[1])] for k in range(f[3])] for mux in range(f[2])]    #weight initialization
    bias=[random.gauss(0,1) for mux in range(f[2])]                                                #bias initialization
    return weights, bias


def weight_init(out_size,in_size):
    weights=[[float(random.gauss(0,1))*math.sqrt(1/in_size) for i in range(in_size)] for j in range(out_size)]
    bias=[random.gauss(0,1) for i in range(out_size)]
    return weights, bias


#Top function used to call functions that read the img, labels, initialize weights and biases
def Initialization(f1,m,l):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    TIX,TOX=Separate(x_train,y_train)
    train_input,train_output,validation_input,validation_output=Separation(TIX,TOX)
    test_input,test_output=Separate(x_test,y_test)

    y1,y2=len(train_input[0][0]),len(train_input[0][0][0])
    cw,cb,lw,lb=[],[],[],[]

    for i in range(len(f1)):
        c1w,c1b=weight_init_conv(f1[i],m)
        cw.append(c1w)
        cb.append(c1b)
        y1,y2=int((y1-f1[i][0]+1)/m[0]),int((y2-f1[i][1]+1)/m[1])

    length=int(y1*y2*f1[len(f1)-1][2])

    for i in range(len(l)):
        l1w,l1b=weight_init(l[i],length)
        lw.append(l1w)
        lb.append(l1b)
        length=l[i]

    return train_input,train_output,validation_input,validation_output,test_input,test_output,cw,cb,lw,lb


def add(in1,in2):
    u=[(in1[i]+in2[i]) for i in range(len(in1))]
    return u


def accu(input,output):
    u=1 if input[output]==max(input) else 0
    return u

def main():
    start_time = time.time()
    RRAMRes,PulseRes,Split=4,3,1
    Ni1,Ni2=5,3
    TotalTrain=40000
    TotalValid=10000
    TotalTest=10000
    RGB=3
    L1O=6
    L2O=16
    
    f1=[]
    f1.append([Ni1,Ni1,6,RGB])
    f1.append([Ni2,Ni2,16,6])
    
    start_time1=time.time()
    TI,TO,VI,VO,ti,to,cw,cb,lw,lb=Initialization(f1,(2,2),(120,84,10))
    minibatch,eta,lmbda,epochs,poolsize=10,float(0.1),float(5),10,2
    M1,M2=float(eta)/minibatch,0.99995
    TrainBatches=int(TotalTrain/minibatch)

    cW,cW2,dcW,dcW2,mcW=[],[],[],[],[]    
    lW,lW2,dlW,dlW2,mlW=[],[],[],[],[]
    for axe in range(len(cw)):
        WeightsW1,WeightsW2,DelW,DelW2,min=Convolution.RRAM_Programming.RRAMProgrammingMergeJ5(cw[axe],RRAMRes,Split,1,-1);
        cW.append(WeightsW1)
        cW2.append(WeightsW2)
        dcW.append(DelW)
        dcW2.append(DelW2)
        mcW.append(min)
    for axe in range(len(lw)):
        WeightsW1,WeightsW2,DelW,DelW2,min=VMM.RRAM_Programming.RRAMProgrammingMergeJ5(lw[axe],RRAMRes,Split,0,-1);
        lW.append(WeightsW1)
        lW2.append(WeightsW2)
        dlW.append(DelW)
        dlW2.append(DelW2)
        mlW.append(min)

    end_time1=time.time()

    print("Unpacking time:")
    print(end_time1-start_time1)
    
    for Epo in range(epochs):
        start_time2= time.time()
        count=0
        for k in range(TrainBatches):
            #print(k)
            Scb,Slb=[[0 for i in j] for j in cb],[[0 for i in j] for j in lb]
            Scw,Slw=[[[[[0 for i in j] for j in kk] for kk in l] for l in p] for p in cw], [[[0 for i in j] for j in l] for l in lw]

            for start in range(minibatch*k,minibatch*(k+1)):
                print(start)
                XXX,x2,y1,o1,a1=FP_InMem2.TopInference(TI,start,cW,cW2,dcW,dcW2,mcW,cb,lW,lW2,dlW,dlW2,mlW,lb,RRAMRes,PulseRes)
                Cw,Cb,Lw,D=BP_NN.TopBackPropNew(a1,o1,lw,y1,x2,XXX,cw,TI[start],TO[start])
                Slb=[[Slb[i][j]+D[i][j] for j in range(len(lb[i]))] for i in range(len(lb))]
                Slw=[[[Slw[i][j][k]+Lw[i][j][k] for k in range(len(lw[i][j]))] for j in range(len(lw[i]))] for i in range(len(lw))]
                Scw=[[[[[Scw[i][j][k][l][m]+Cw[i][j][k][l][m] for m in range(len(cw[i][j][k][l]))] for l in range(len(cw[i][j][k]))] for k in range(len(cw[i][j]))] for j in range(len(cw[i]))] for i in range(len(cw))] 
                Scb=[[Scb[i][j]+Cb[i][j] for j in range(len(cb[i]))] for i in range(len(cb))]
            lb=[[(lb[i][j]-(Slb[i][j]*M1)) for j in range(len(lb[i]))] for i in range(len(lb))]
            lw=[[[(lw[i][j][k]*M2)-(Slw[i][j][k]*M1) for k in range(len(lw[i][j]))] for j in range(len(lw[i]))] for i in range(len(lw))]
            cw=[[[[[(cw[i][j][k][l][m]*M2)-(Scw[i][j][k][l][m]*M1) for m in range(len(cw[i][j][k][l]))] for l in range(len(cw[i][j][k]))] for k in range(len(cw[i][j]))] for j in range(len(cw[i]))] for i in range(len(cw))] 
            cb=[[cb[i][j]-(Scb[i][j]*M1) for j in range(len(cb[i]))] for i in range(len(cb))]
            cW,cW2,dcW,dcW2,mcW=[],[],[],[],[]    
            lW,lW2,dlW,dlW2,mlW=[],[],[],[],[]
            for axe in range(len(cw)):
                WeightsW1,WeightsW2,DelW,DelW2,min= Convolution.RRAM_Programming.RRAMProgrammingMergeJ5(cw[axe],RRAMRes,Split,1,-1);
                cW.append(WeightsW1)
                cW2.append(WeightsW2)
                dcW.append(DelW)
                dcW2.append(DelW2)
                mcW.append(min)
            for axe in range(len(lw)):
                WeightsW1,WeightsW2,DelW,DelW2,min= VMM.RRAM_Programming.RRAMProgrammingMergeJ5(lw[axe],RRAMRes,Split,0,-1);
                lW.append(WeightsW1)
                lW2.append(WeightsW2)
                dlW.append(DelW)
                dlW2.append(DelW2)
                mlW.append(min)
            
        Accuracy=0
        for ick in range(TotalValid):
            XXX,x2,y1,o1,a1=FP_InMem2.TopInference(VI,ick,cW,cW2,dcW,dcW2,mcW,cb,lW,lW2,dlW,dlW2,mlW,lb,RRAMRes,PulseRes)
            Accuracy=Accuracy+accu(a1[len(a1)-1],VO[ick])
        print("Epoch {0}: Validation accuracy {1}".format(Epo, Accuracy))
        #print(Accuracy)
        TI,TO=ImgShuffle_CIFAR10.Shuffle(TI,TO)
        VI,VO=ImgShuffle_CIFAR10.Shuffle(VI,VO)
        elapsed_time_secs2 = time.time() - start_time2
        print("Epoch time:")
        print(elapsed_time_secs2)

    Accuracy=0
    for i in range(TotalTest):
        XXX,x2,y1,o1,a1=FP_InMem2.TopInference(ti,i,cW,cW2,dcW,dcW2,mcW,cb,lW,lW2,dlW,dlW2,mlW,lb,RRAMRes,PulseRes)
        Accuracy=Accuracy+accu(a1[len(a1)-1],to[i])
    print("Test accuracy {0}".format(Accuracy))

    elapsed_time_secs = time.time() - start_time
    print("Total Time:")
    print(elapsed_time_secs)

if __name__ =="__main__":
    random.seed(2)
    main()
