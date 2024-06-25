#!/usr/bin/python
import math
import sys
import random


#######################Code#############################################################
def Add(BW,Tel1,Output):

	BPD=int(math.log(BW,2)+1)
	ND=math.ceil((2*BW)/BPD)
	
	FinalOut=[[0 for i in range(2*BW-1)] for j in range(Tel1+2)]
	
	#print(BPD, ND)

	#print("In Post Processing:")
	
	if(BW==2):
		X1=10*math.ceil(Tel1/3)
		#print(X1)
		m1=0
		for i in range(X1):
			if(i%10>0):
				l1=i-(int(i/10)*10)
				l1x=int(i/10)*3
				l2=l1-((m1-l1x)*3)
				FinalOut[m1][3-l2]=Output[i]
				m1=m1+1 if l1%3==0 else m1


	if(BW==4):
		X1=10*math.ceil(Tel1)
		m1=0
		for i in range(X1):
			if(i%10>2):
				FinalOut[m1][6-((i%10)-3)]=Output[i]
				m1=m1+1 if i%10==9 else m1

	if(BW==8):
		X1=20*math.ceil(Tel1)
		m1=0
		for i in range(X1):
			l1=int(i/10)
			if(l1%2==0):
				#print(5+(i%10))
				#print
				FinalOut[m1][5+(i%10)]=Output[i]
			if(l1%2==1):
				if((i%10)<5):
					#print(l1, 4-(i%10))
					#print
					FinalOut[m1][4-(i%10)]=Output[i]

				m1=m1+1 if (i%10)==5 else m1


	if(BW==16):
		X1=60*math.ceil(Tel1)
		m1=0
		lam=(2*int(BW/8))-1
		FinalOut2=[[0 for i in range(15*lam)] for j in range(Tel1)]
		for i in range(X1):
			l1=int(i/10)
			l2=l1%6
			if(l2%2==0):
				FinalOut2[m1][5+(i%10)+(15*int(l2/2))]=Output[i]
			if(l1%2==1):
				if((i%10)<5):
					#print(l1, 4-(i%10))
					#print
					FinalOut2[m1][4-(i%10)+(15*int(l2/2))]=Output[i]

			m1=m1+1 if (i%10==5 and l2==5) else m1

		for j in range(Tel1):
			for i in range(2*BW-1):
				if(i<8):
					FinalOut[j][i]=FinalOut2[j][i]
				if(15>i>7):
					FinalOut[j][i]=FinalOut2[j][i]+FinalOut2[j][i+7]

				FinalOut[j][15]=FinalOut2[j][22]

				if(23>i>15):
					FinalOut[j][i]=FinalOut2[j][i+7]+FinalOut2[j][i+14]

				if(i>22):
					FinalOut[j][i]=FinalOut2[j][i+14]


	if(BW==32):
		X1=140*math.ceil(Tel1)
		m1=0
		lam=(2*int(BW/8))-1
		FinalOut2=[[0 for i in range(15*lam)] for j in range(Tel1)]
		for i in range(X1):
			l1=int(i/10)
			l2=l1%14
			if(l2%2==0):
				FinalOut2[m1][5+(i%10)+(15*int(l2/2))]=Output[i]
			if(l1%2==1):
				if((i%10)<5):
					#print(l1, 4-(i%10))
					#print
					FinalOut2[m1][4-(i%10)+(15*int(l2/2))]=Output[i]

			m1=m1+1 if (i%10==5 and l2==13) else m1

		for j in range(Tel1):
			for i in range(2*BW-1):
				if(i<8):
					FinalOut[j][i]=FinalOut2[j][i]
				if(15>i>7):
					FinalOut[j][i]=FinalOut2[j][i]+FinalOut2[j][i+7]

				FinalOut[j][15]=FinalOut2[j][22]

				if(23>i>15):
					FinalOut[j][i]=FinalOut2[j][i+7]+FinalOut2[j][i+14]
				
				FinalOut[j][23]=FinalOut2[j][37]

				if(31>i>23):
					FinalOut[j][i]=FinalOut2[j][i+14]+FinalOut2[j][i+21]
				
				FinalOut[j][31]=FinalOut2[j][52]

				if(39>i>31):
					FinalOut[j][i]=FinalOut2[j][i+21]+FinalOut2[j][i+28]

				FinalOut[j][39]=FinalOut2[j][67]

				if(47>i>39):
					FinalOut[j][i]=FinalOut2[j][i+28]+FinalOut2[j][i+35]

				FinalOut[j][47]=FinalOut2[j][82]

				if(55>i>47):
					FinalOut[j][i]=FinalOut2[j][i+35]+FinalOut2[j][i+42]

				if(i>54):
					FinalOut[j][i]=FinalOut2[j][i+42]

	if(BW==64):
		X1=300*math.ceil(Tel1)
		m1=0
		lam=(2*int(BW/8))-1
		FinalOut2=[[0 for i in range(15*lam)] for j in range(Tel1)]
		Array=0
		for i in range(X1):
			
			if(Array==0):
				l1=int(i/10)
				l2=l1%18
				if(l2%2==0):
					FinalOut2[m1][5+(i%10)+(15*int(l2/2))]=Output[0][i]
					#print(i, m1, 5+(i%10)+(15*int(l2/2)), l1, l2, Array)
				if(l1%2==1):
					if((i%10)<5):
						FinalOut2[m1][4-(i%10)+(15*int(l2/2))]=Output[0][i]
						#print(i, m1, 4-(i%10)+(15*int(l2/2)), l1, l2, Array)

			if(Array==1):
				l1=int((i-180)/10)
				l2=l1%12
				Nax=135
				if(l2%2==0):
					FinalOut2[m1][5+(i%10)+(15*int(l2/2))+Nax]=Output[1][i-180]
					#print(i, m1, 5+(i%10)+(15*int(l2/2))+Nax, l1, l2, Array, i-180)
				if(l1%2==1):
					if((i%10)<5):
						FinalOut2[m1][4-(i%10)+(15*int(l2/2))+Nax]=Output[1][i-180]
						#print(i, m1, 4-(i%10)+(15*int(l2/2))+Nax, l1, l2, Array, i-180)

				m1=m1+1 if (i%10==5 and l2==11) else m1
				

			Array=Array+1 if (l1==17 and i%10==9) else Array
			#print(m1, Array)
			

		for j in range(Tel1):
			for i in range(2*BW-1):
				if(i<8):
					FinalOut[j][i]=FinalOut2[j][i]
				if(15>i>7):
					FinalOut[j][i]=FinalOut2[j][i]+FinalOut2[j][i+7]

				FinalOut[j][15]=FinalOut2[j][22]

				if(23>i>15):
					FinalOut[j][i]=FinalOut2[j][i+7]+FinalOut2[j][i+14]
				
				FinalOut[j][23]=FinalOut2[j][37]

				if(31>i>23):
					FinalOut[j][i]=FinalOut2[j][i+14]+FinalOut2[j][i+21]
				
				FinalOut[j][31]=FinalOut2[j][52]

				if(39>i>31):
					FinalOut[j][i]=FinalOut2[j][i+21]+FinalOut2[j][i+28]

				FinalOut[j][39]=FinalOut2[j][67]

				if(47>i>39):
					FinalOut[j][i]=FinalOut2[j][i+28]+FinalOut2[j][i+35]

				FinalOut[j][47]=FinalOut2[j][82]

				if(55>i>47):
					FinalOut[j][i]=FinalOut2[j][i+35]+FinalOut2[j][i+42]

				FinalOut[j][55]=FinalOut2[j][97]

				if(63>i>55):
					FinalOut[j][i]=FinalOut2[j][i+42]+FinalOut2[j][i+49]

				FinalOut[j][63]=FinalOut2[j][112]

				if(71>i>63):
					FinalOut[j][i]=FinalOut2[j][i+49]+FinalOut2[j][i+56]

				FinalOut[j][71]=FinalOut2[j][127]

				if(79>i>71):
					FinalOut[j][i]=FinalOut2[j][i+56]+FinalOut2[j][i+63]

				FinalOut[j][79]=FinalOut2[j][142]

				if(87>i>79):
					FinalOut[j][i]=FinalOut2[j][i+63]+FinalOut2[j][i+70]

				FinalOut[j][87]=FinalOut2[j][157]

				if(95>i>87):
					FinalOut[j][i]=FinalOut2[j][i+70]+FinalOut2[j][i+77]

				FinalOut[j][95]=FinalOut2[j][172]

				if(103>i>95):
					FinalOut[j][i]=FinalOut2[j][i+77]+FinalOut2[j][i+84]

				FinalOut[j][103]=FinalOut2[j][187]

				if(111>i>103):
					FinalOut[j][i]=FinalOut2[j][i+84]+FinalOut2[j][i+91]

				FinalOut[j][111]=FinalOut2[j][202]

				if(119>i>111):
					FinalOut[j][i]=FinalOut2[j][i+91]+FinalOut2[j][i+98]

				if(i>118):
					FinalOut[j][i]=FinalOut2[j][i+98]
	
	#print(FinalOut)

	L2=[[0 for i in range(BPD*ND)] for j in range(Tel1+2)]
	L1=[[0 for i in range(2*BW)] for j in range(BPD)]

	for i in range(Tel1):
		for j in range(2*BW-1):
			L2[i][j]=FinalOut[i][j]

	#print(Output)
	#print(FinalOut)
	#print
	#print

	NewOutputs=[0 for i in range(Tel1+2)]

	for l4 in range(Tel1):
		L1=[[0 for i in range((ND+1)*BPD)] for j in range(BPD)]
		for i in range(BPD):
			for j in range(ND):
				k1=L2[l4][j*BPD+i]
				for lax in range(BPD):
					L1[i][j*BPD+i+lax]=k1%2
					k1=int(k1/2)
		#print(L1)
		#print

		L1x=[0 for i in range(BPD)]

		for i in range(BPD):
			for j in range(2*BW):
				#if(BW==2):
				L1x[i]=L1x[i]+(math.pow(2,j)*L1[i][j])
				#if(BW==4):
				#	L1x[i]=L1x[i]+(math.pow(2,7-j)*L1[i][j])

		for i in range(BPD):
			NewOutputs[l4]=NewOutputs[l4]+L1x[i]

	#print(NewOutputs)

	if(BW==2):
		max=math.ceil(Tel1/3)
		#print(max)
		for i in range(max):
			if(i%2==1):
				k1=NewOutputs[3*i]
				#print(i, k1, NewOutputs[3*i], NewOutputs[3*i+2])
				NewOutputs[3*i]=NewOutputs[3*i+2]
				NewOutputs[3*i+2]=k1
				#print(i, k1, NewOutputs[3*i], NewOutputs[3*i+2])
			

	#print(NewOutputs)
				
	return(NewOutputs)
