
import math
import sys
import random
import numpy as np
import time


#Combines the input pixel data with the classification label data 
#This takes 3 parameters- img pixel data, img label data, num of train/test/validations imgs
def Combine(I,O,NI):
	w,h=len(I[0][0]),NI								
	Combined=[[[[(I[y][umm][xc][x] if x<w else (O[y] if (umm==0 and xc==0 and x==w) else 0)) for x in range(w+1)] for xc in range(w)] for umm in range(len(I[0]))] for y in range(NI)] 
	#Combined=[[[[0 for x in range(w+1)] for xc in range(w)] for umm in range(len(I[0]))] for y in range(NI)]


	#for i in range(h):
	#	for umm in range(len(I[0])):								
	#		for j in range(w):
	#			for k in range(w+1):
	#				if(k<w):
	#					Combined[i][umm][j][k]=I[i][umm][j][k]
	#				if(umm==0 and k==w and j==0):
	#					Combined[i][umm][j][k]=O[i]
	return Combined,w


#Separates the final shuffled output with label data appended at the end into 2 separate arrays- the pixel data and the label data
#Takes 3 parameters- shuffled numpy output, num of imgs, width of the individual imgs. Returns 2 arrays- one containing the shuffled img pixel data and another containing the img label data
def Separate(SO,NI,w):
	h=NI
	uxm=len(SO[0])										
	SeparatedIn=[[[[float(SO[y,umm,xc,x]) for x in range(w)] for xc in range(w)] for umm in range(uxm)] for y in range(NI)] 	
	SeparatedOut=[int(SO[x,0,0,w]) for x in range(NI)]
	#SeparatedIn=[[[[0 for x in range(w)] for xc in range(w)] for umm in range(uxm)] for y in range(NI)] 	
	#SeparatedOut=[0 for x in range(NI)]						
	#for i in range(h):
	#	for umm in range(uxm):								
	#		for j in range(w):
	#			for k in range(w+1):
	#				if(k<w):
	#					SeparatedIn[i][umm][j][k]=float(SO[i,umm,j,k])
	#				if(umm==0 and k==w and j==0):
	#					SeparatedOut[i]=int(SO[i,umm,j,k])
	return SeparatedIn,SeparatedOut


#Top function used to call functions that perform the shuffling
def Shuffle(I,O):
	NI=len(O)
	Combined,w=Combine(I,O,NI)
	Arr=np.array(Combined)
	Shuffled=np.random.shuffle(Arr)
	SeparatedI,SeparatedO=Separate(Arr,NI,w)
	return SeparatedI,SeparatedO