#!/usr/bin/python
import math
import sys
import random


#######################Code#############################################################
def ArrayDef(Arr,n,BW):

	N1=1 if BW<64 else 2
	Arr1=[[[0 for i in range(180)] for j in range(90)] for k in range(N1)]
	#Arr1=[[0 for i in range(180)] for j in range(90)]
	
	NX11=[1,1,1,2,4,8]
	
	if(BW==2):
		Nx1=math.ceil(n/3)
		for i in range(Nx1):
			L1=88 if i%2==0 else 81
			U1=90 if i%2==0 else 83
			for j in range(L1,U1,1):
				lx1=10*i
				mx1=0 if j%2==0 else 1
				for m in range(3):
					if(3*i+m<n):
						lx2=3*i+m
						L2=1 if (m==0 and mx1==0) else 0	
						for p in range(L2, 3, 1):
							if(i%2==0):
								AN1=(lx1+1+(3*m)+p) 
							else:
								AN1=(lx1+9-(3*m)-p)
							#print(j, AN1, lx2, mx1)
							#print
							Arr1[0][j][AN1]=Arr[lx2][0][mx1]

	if(BW==4):
		Nx1=math.ceil(n/1)
		for i in range(Nx1):
			L1=86 if i%2==0 else 81
			U1=90 if i%2==0 else 85
			Add=3 if i%2==0 else 0
			mx1=0 if i%2==0 else 3
			for j in range(L1,U1,1):
				lx1=10*i+3+Add
				lx2=i	
				for p in range(4):
					AN1=lx1+p
					Arr1[0][j][AN1]=Arr[lx2][0][mx1]
				Add=Add-1 if i%2==0 else Add+1
				mx1=mx1+1 if i%2==0 else mx1-1


	if(BW==8):
		Nx1=math.ceil(2*n)
		for i in range(Nx1):				
			L1=81 if i%2==0 else 85
			U1=89 if i%2==0 else 90
			AddL1=0 
			AddU1=10 if i%2==0 else 4
			mx1=7 if i%2==0 else 4
			for j in range(L1,U1,1):
				lx2=int(i/2)	
				for p in range(AddL1,AddU1,1):
					p1=i*10+p
					Arr1[0][j][p1]=Arr[lx2][0][mx1]
				if i%2==0: 
					AddU1=AddU1-1
				else:
					AddU1=AddU1+1 if j==85 else AddU1
				mx1=mx1-1


	if(BW==16):
		Nx1=math.ceil(6*n)
		for i in range(Nx1):

			if(i%6==0):
				Preset,L1,U1,AddL1,AddU1,mx1=0,81,89,0,10,7
			if(i%6==1):
				Preset,L1,U1,AddL1,AddU1,mx1=0,85,90,0,4,4
			if(i%6==2):
				Preset,L1,U1,AddL1,AddU1,mx1=1,81,89,0,10,7
				L2,U2,AddL2,AddU2,mx2=72,80,0,10,15
			if(i%6==3):
				Preset,L1,U1,AddL1,AddU1,mx1=1,85,90,0,4,4
				L2,U2,AddL2,AddU2,mx2=76,81,0,4,12
			if(i%6==4):
				Preset,L2,U2,AddL2,AddU2,mx2=2,72,80,0,10,15
			if(i%6==5):
				Preset,L2,U2,AddL2,AddU2,mx2=2,76,81,0,4,12

			if(Preset<2):			
				for j in range(L1,U1,1):
					lx2=int(i/6)	
					for p in range(AddL1,AddU1,1):
						p1=i*10+p
						Arr1[0][j][p1]=Arr[lx2][0][mx1]
					if i%2==0: 
						AddU1=AddU1-1
					else:
						AddU1=AddU1+1 if j==85 else AddU1
					mx1=mx1-1
			if(Preset>0):			
				for j in range(L2,U2,1):
					lx22=int(i/6)	
					for p in range(AddL2,AddU2,1):
						p1=i*10+p
						Arr1[0][j][p1]=Arr[lx22][0][mx2]
					if i%2==0: 
						AddU2=AddU2-1
					else:
						AddU2=AddU2+1 if j==76 else AddU2
					mx2=mx2-1


	if(BW==32):
		Nx1=math.ceil(14*n)
		for i in range(Nx1):

			if(i%2==0):
				L1,U1,AddL1,AddU1,mx1=81,89,0,10,7
				L2,U2,AddL2,AddU2,mx2=72,80,0,10,15
				L3,U3,AddL3,AddU3,mx3=63,71,0,10,23
				L4,U4,AddL4,AddU4,mx4=54,62,0,10,31

			if(i%2==1):
				L1,U1,AddL1,AddU1,mx1=85,90,0,4,4
				L2,U2,AddL2,AddU2,mx2=76,81,0,4,12
				L3,U3,AddL3,AddU3,mx3=67,72,0,4,20
				L4,U4,AddL4,AddU4,mx4=58,63,0,4,28

			Preset1=0 if i<8 else 1
			Preset2=0 if 1<i<10 else 1
			Preset3=0 if 3<i<12 else 1
			Preset4=0 if i>5 else 1

			if(Preset1==0):			
				for j in range(L1,U1,1):
					lx2=int(i/14)	
					for p in range(AddL1,AddU1,1):
						p1=i*10+p
						Arr1[0][j][p1]=Arr[lx2][0][mx1]
					if i%2==0: 
						AddU1=AddU1-1
					else:
						AddU1=AddU1+1 if j==85 else AddU1
					mx1=mx1-1
			if(Preset2==0):			
				for j in range(L2,U2,1):
					lx22=int(i/14)	
					for p in range(AddL2,AddU2,1):
						p1=i*10+p
						Arr1[0][j][p1]=Arr[lx22][0][mx2]
					if i%2==0: 
						AddU2=AddU2-1
					else:
						AddU2=AddU2+1 if j==76 else AddU2
					mx2=mx2-1

			if(Preset3==0):			
				for j in range(L3,U3,1):
					lx23=int(i/14)	
					for p in range(AddL3,AddU3,1):
						p1=i*10+p
						Arr1[0][j][p1]=Arr[lx23][0][mx3]
					if i%2==0: 
						AddU3=AddU3-1
					else:
						AddU3=AddU3+1 if j==67 else AddU3
					mx3=mx3-1

			if(Preset4==0):			
				for j in range(L4,U4,1):
					lx24=int(i/14)	
					for p in range(AddL4,AddU4,1):
						p1=i*10+p
						Arr1[0][j][p1]=Arr[lx24][0][mx4]
					if i%2==0: 
						AddU4=AddU4-1
					else:
						AddU4=AddU4+1 if j==58 else AddU4
					mx4=mx4-1
						


	if(BW==64):

		L=[[81, 72, 63, 54, 45, 36, 27, 18], [85, 76, 67, 58, 49, 40, 31, 22]]
		U=[[89, 80, 71, 62, 53, 44, 35, 26], [90, 81, 72, 63, 54, 45, 36, 27]]
		AddL=0
		AddU=[10, 4]
		mx=[[7, 15, 23, 31, 39, 47, 55, 63], [4, 12, 20, 28, 36, 44, 52, 60]]

		for TBA in range(2):

			Nx1=12 if TBA>0 else 18 
			for i in range(Nx1):

				tA=int(i%2)

				if(TBA==0):
					Preset1=0 if i<16 else 1
					Preset2=0 if 1<i else 1
					Preset3=0 if 3<i else 1
					Preset4=0 if i>5 else 1
					Preset5=0 if i>7 else 1
					Preset6=0 if 9<i else 1
					Preset7=0 if 11<i else 1
					Preset8=0 if i>13 else 1

				if(TBA==1):
					Preset1=1
					Preset2=1
					Preset3=0 if i<2 else 1
					Preset4=0 if i<4 else 1
					Preset5=0 if i<6 else 1
					Preset6=0 if i<8 else 1
					Preset7=0 if i<10 else 1
					Preset8=0
			

				if(Preset1==0):
					mxA, LA, UA, AddLA, AddUA=mx[tA][0], L[tA][0], U[tA][0], AddL, AddU[tA]		
					for j in range(LA,UA,1):
						lx2=int(i/Nx1)	
						for p in range(AddLA,AddUA,1):
							p1=i*10+p
							Arr1[TBA][j][p1]=Arr[lx2][0][mxA]
						if i%2==0: 
							AddUA=AddUA-1
						else:
							AddUA=AddUA+1 if j==L[tA][0] else AddUA
						mxA=mxA-1

				if(Preset2==0):			
					mxA, LA, UA, AddLA, AddUA=mx[tA][1], L[tA][1], U[tA][1], AddL, AddU[tA]		
					for j in range(LA,UA,1):
						lx2=int(i/Nx1)	
						for p in range(AddLA,AddUA,1):
							p1=i*10+p
							Arr1[TBA][j][p1]=Arr[lx2][0][mxA]
						if i%2==0: 
							AddUA=AddUA-1
						else:
							AddUA=AddUA+1 if j==L[tA][1] else AddUA
						mxA=mxA-1
	
				if(Preset3==0):			
					mxA, LA, UA, AddLA, AddUA=mx[tA][2], L[tA][2], U[tA][2], AddL, AddU[tA]		
					for j in range(LA,UA,1):
						lx2=int(i/Nx1)	
						for p in range(AddLA,AddUA,1):
							p1=i*10+p
							Arr1[TBA][j][p1]=Arr[lx2][0][mxA]
						if i%2==0: 
							AddUA=AddUA-1
						else:
							AddUA=AddUA+1 if j==L[tA][2] else AddUA
						mxA=mxA-1

				if(Preset4==0):			
					mxA, LA, UA, AddLA, AddUA=mx[tA][3], L[tA][3], U[tA][3], AddL, AddU[tA]		
					for j in range(LA,UA,1):
						lx2=int(i/Nx1)	
						for p in range(AddLA,AddUA,1):
							p1=i*10+p
							Arr1[TBA][j][p1]=Arr[lx2][0][mxA]
						if i%2==0: 
							AddUA=AddUA-1
						else:
							AddUA=AddUA+1 if j==L[tA][3] else AddUA
						mxA=mxA-1

				if(Preset5==0):			
					mxA, LA, UA, AddLA, AddUA=mx[tA][4], L[tA][4], U[tA][4], AddL, AddU[tA]		
					for j in range(LA,UA,1):
						lx2=int(i/Nx1)	
						for p in range(AddLA,AddUA,1):
							p1=i*10+p
							Arr1[TBA][j][p1]=Arr[lx2][0][mxA]
						if i%2==0: 
							AddUA=AddUA-1
						else:
							AddUA=AddUA+1 if j==L[tA][4] else AddUA
						mxA=mxA-1

				if(Preset6==0):			
					mxA, LA, UA, AddLA, AddUA=mx[tA][5], L[tA][5], U[tA][5], AddL, AddU[tA]		
					for j in range(LA,UA,1):
						lx2=int(i/Nx1)	
						for p in range(AddLA,AddUA,1):
							p1=i*10+p
							Arr1[TBA][j][p1]=Arr[lx2][0][mxA]
						if i%2==0: 
							AddUA=AddUA-1
						else:
							AddUA=AddUA+1 if j==L[tA][5] else AddUA
						mxA=mxA-1

				if(Preset7==0):			
					mxA, LA, UA, AddLA, AddUA=mx[tA][6], L[tA][6], U[tA][6], AddL, AddU[tA]		
					for j in range(LA,UA,1):
						lx2=int(i/Nx1)	
						for p in range(AddLA,AddUA,1):
							p1=i*10+p
							Arr1[TBA][j][p1]=Arr[lx2][0][mxA]
						if i%2==0: 
							AddUA=AddUA-1
						else:
							AddUA=AddUA+1 if j==L[tA][6] else AddUA
						mxA=mxA-1

				if(Preset8==0):			
					mxA, LA, UA, AddLA, AddUA=mx[tA][7], L[tA][7], U[tA][7], AddL, AddU[tA]		
					for j in range(LA,UA,1):
						lx2=int(i/Nx1)	
						for p in range(AddLA,AddUA,1):
							p1=i*10+p
							Arr1[TBA][j][p1]=Arr[lx2][0][mxA]
						if i%2==0: 
							AddUA=AddUA-1
						else:
							AddUA=AddUA+1 if j==L[tA][7] else AddUA
						mxA=mxA-1

	return(Arr1)

