#!/usr/bin/python
import math
import sys
import random


#######################Code#############################################################
def ArrayPulses(Arr,n,BW):

	N1=1 if BW<64 else 2
	Arr1=[[[0 for i in range(90)] for j in range(36)] for k in range(N1)]
	
	if(BW==2):
		Nx1=math.ceil(n/3)
		for i in range(Nx1):
			L1=81 if i%2==0 else 82
			U1=89 if i%2==0 else 90
			m=0
			Start=1 if i%2==0 else 0
			Zero=2 if i%2==0 else 0
			Zero1=0 if i%2==0 else 1
			for j in range(L1,U1,1):
				if(3*i+m<n):
					#m1=3*i+((1-(i%2))*m)+((i%2)*(2-m))
					m1=3*i+m
					if(i%2==0):
						if(j%3==Zero1): 
							mx1=Start
						else:
							mx1=mx1-1
					if(i%2==1):
						if(j%3==Zero1): 
							mx1=Start
						else:
							mx1=mx1+1
					if(j%3!=Zero):
						Arr1[0][2*i+1][j]=Arr[m1][1][mx1]
						if(2*(i+1)<36):
							Arr1[0][2*(i+1)][j-9]=Arr[m1][1][mx1]
					m=m+1 if j%3==Zero else m

	if(BW==4):
		Nx1=math.ceil(n/1)
		for i in range(Nx1):
			L1=83 if i%2==0 else 84
			U1=87 if i%2==0 else 88
			Start=3 if i%2==0 else 0
			for j in range(L1,U1,1):
				m1=i
				if(i%2==0):
					if(j==L1): 
						mx1=Start
					else:
						mx1=mx1-1
				if(i%2==1):
					if(j==L1): 
						mx1=Start
					else:
						mx1=mx1+1
				Arr1[0][2*i+1][j]=Arr[m1][1][mx1]
				if(2*(i+1)<36):
					Arr1[0][2*(i+1)][j-9]=Arr[m1][1][mx1]

	if(BW==8):
		Nx1=math.ceil(2*n)
		for i in range(Nx1):
			L1=83 if i%2==0 else 85
			U1=90
			m1=int(i/2)
			mx1=0
			for j in range(L1,U1,1):
				Arr1[0][2*i][j]=Arr[m1][1][mx1]
				mx1=mx1+1
			if(i%2==0):
				Arr1[0][2*i+1][81]=Arr[m1][1][7]
				if(2*(i+1)<36):
					Arr1[0][2*(i+1)][72]=Arr[m1][1][7]

	if(BW==16):
		Nx1=math.ceil(6*n)
		for i in range(Nx1):
			Preset1=0 if i%6<4 else 1
			Preset2=0 if i%6>1 else 1

			if(Preset1==0):
				L1=83 if i%2==0 else 85
				U1=90
				m1=int(i/6)
				mx1=8*int((i%6)/2)
				for j in range(L1,U1,1):
					Arr1[0][2*i][j]=Arr[m1][1][mx1]
					if(2*(i+1)+1<36):
						Arr1[0][2*(i+1)+1][j]=Arr[m1][1][mx1]
					mx1=mx1+1
				mx2=8*int((i%6)/2)+7
				if(i%2==0):
					Arr1[0][2*i+1][81]=Arr[m1][1][mx2]

				for j in range(81,90,1):
					if(2*(i+1)<36):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(i%6==4):
				for j in range(81,90,1):
					if(2*(i+1)<36):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Preset2==0 and i%2==0):
				m1=int(i/6)
				mx2=8*(int((i%6)/2)-1)+7
				Arr1[0][2*i+1][72]=Arr[m1][1][mx2]
				if(2*(i+1)<36):
					Arr1[0][2*(i+1)][63]=Arr1[0][2*i+1][72]
				
	if(BW==32):
		Nx1=math.ceil(14*n)
		for i in range(Nx1):
			Rem=i%14
			Preset1=0 if Rem<8 else 1
			Preset2=0 if 10>Rem>1 else 1
			Preset3=0 if 12>Rem>3 else 1
			Preset4=0 if Rem>5 else 1

			if(Preset1==0):
				L1=83 if i%2==0 else 85
				U1=90
				m1=int(i/14)
				mx1=8*int(Rem/2)
				for j in range(L1,U1,1):
					Arr1[0][2*i][j]=Arr[m1][1][mx1]
					if(2*(i+1)+1<36):
						Arr1[0][2*(i+1)+1][j]=Arr[m1][1][mx1]
					mx1=mx1+1
				mx2=8*int(Rem/2)+7
				if(i%2==0):
					Arr1[0][2*i+1][81]=Arr[m1][1][mx2]

				for j in range(81,90,1):
					if(2*(i+1)<36):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Rem==8):
				for j in range(81,90,1):
					if(2*(i+1)<36):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Preset2==0):
				L1=74 if i%2==0 else 76
				U1=81
				m1=int(i/14)
				for j in range(L1,U1,1):
					if(2*(i+1)+1<36):
						Arr1[0][2*(i+1)+1][j]=Arr1[0][2*i][j]
				mx2=8*(int(Rem/2)-1)+7
				if(i%2==0):
					Arr1[0][2*i+1][72]=Arr[m1][1][mx2]

				for j in range(72,81,1):
					if(2*(i+1)<36):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Rem==10):
				for j in range(72,81,1):
					if(2*(i+1)<36):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Preset3==0):
				L1=65 if i%2==0 else 67
				U1=72
				m1=int(i/14)
				for j in range(L1,U1,1):
					if(2*(i+1)+1<36):
						Arr1[0][2*(i+1)+1][j]=Arr1[0][2*i][j]
				mx2=8*(int(Rem/2)-2)+7
				if(i%2==0):
					Arr1[0][2*i+1][63]=Arr[m1][1][mx2]

				for j in range(63,72,1):
					if(2*(i+1)<36):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Rem==12):
				for j in range(63,72,1):
					if(2*(i+1)<36):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Preset4==0 and i%2==0):
				m1=int(i/14)
				mx2=8*(int(Rem/2)-3)+7
				Arr1[0][2*i+1][54]=Arr[m1][1][mx2]
				if(2*(i+1)<36):
					Arr1[0][2*(i+1)][45]=Arr1[0][2*i+1][54]


	if(BW==64):

		Nx1=18

		for i in range(Nx1):
			Rem=i%18
			Preset1=0 if Rem<16 else 1
			Preset2=0 if Rem>1 else 1
			Preset3=0 if Rem>3 else 1
			Preset4=0 if Rem>5 else 1
			Preset5=0 if Rem>7 else 1
			Preset6=0 if Rem>9 else 1
			Preset7=0 if Rem>11 else 1
			Preset8=0 if Rem>13 else 1

			if(Preset1==0):
				L1=83 if i%2==0 else 85
				U1=90
				m1=int(i/18)
				mx1=8*int(Rem/2)
				for j in range(L1,U1,1):
					Arr1[0][2*i][j]=Arr[m1][1][mx1]
					Arr1[0][2*(i+1)+1][j]=Arr[m1][1][mx1]
					mx1=mx1+1
				mx2=8*int(Rem/2)+7
				if(i%2==0):
					Arr1[0][2*i+1][81]=Arr[m1][1][mx2]
				for j in range(81,90,1):
					Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Rem==16):
				for j in range(81,90,1):
					Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]

			if(Preset2==0):
				L1=74 if i%2==0 else 76
				U1=81
				m1=int(i/18)
				for j in range(L1,U1,1):
					if(i<16):
						Arr1[0][2*(i+1)+1][j]=Arr1[0][2*i][j]
				mx2=8*(int(Rem/2)-1)+7
				if(i%2==0):
					Arr1[0][2*i+1][72]=Arr[m1][1][mx2]
				if(i<17):
					for j in range(72,81,1):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]


			if(Preset3==0):
				L1=65 if i%2==0 else 67
				U1=72
				m1=int(i/18)
				for j in range(L1,U1,1):
					if(i<16):
						Arr1[0][2*(i+1)+1][j]=Arr1[0][2*i][j]
				mx2=8*(int(Rem/2)-2)+7
				if(i%2==0):
					Arr1[0][2*i+1][63]=Arr[m1][1][mx2]
				if(i<17):
					for j in range(63,72,1):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]


			if(Preset4==0):
				L1=56 if i%2==0 else 58
				U1=63
				m1=int(i/18)
				for j in range(L1,U1,1):
					if(i<16):
						Arr1[0][2*(i+1)+1][j]=Arr1[0][2*i][j]
				mx2=8*(int(Rem/2)-3)+7
				if(i%2==0):
					Arr1[0][2*i+1][54]=Arr[m1][1][mx2]
				if(i<17):
					for j in range(54,63,1):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]


			if(Preset5==0):
				L1=47 if i%2==0 else 49
				U1=54
				m1=int(i/18)
				for j in range(L1,U1,1):
					if(i<16):
						Arr1[0][2*(i+1)+1][j]=Arr1[0][2*i][j]
				mx2=8*(int(Rem/2)-4)+7
				if(i%2==0):
					Arr1[0][2*i+1][45]=Arr[m1][1][mx2]

				if(i<17):
					for j in range(45,54,1):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]


			if(Preset6==0):
				L1=38 if i%2==0 else 40
				U1=45
				m1=int(i/18)
				for j in range(L1,U1,1):
					if(i<16):
						Arr1[0][2*(i+1)+1][j]=Arr1[0][2*i][j]
				mx2=8*(int(Rem/2)-5)+7
				if(i%2==0):
					Arr1[0][2*i+1][36]=Arr[m1][1][mx2]
				if(i<17):
					for j in range(36,45,1):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]


			if(Preset7==0):
				L1=29 if i%2==0 else 31
				U1=36
				m1=int(i/18)
				for j in range(L1,U1,1):
					if(i<16):
						Arr1[0][2*(i+1)+1][j]=Arr1[0][2*i][j]
				mx2=8*(int(Rem/2)-6)+7
				if(i%2==0):
					Arr1[0][2*i+1][27]=Arr[m1][1][mx2]

				if(i<17):
					for j in range(27,36,1):
						Arr1[0][2*(i+1)][j-9]=Arr1[0][2*i+1][j]


			if(Preset8==0 and i%2==0):
				m1=int(i/18)
				mx2=8*(int(Rem/2)-7)+7
				Arr1[0][2*i+1][18]=Arr[m1][1][mx2]
				Arr1[0][2*(i+1)][9]=Arr1[0][2*i+1][18]


		Nx1=12
		Rem2=8*(int(18/2)-2)

		mx2=Rem2-8

		for j in range(56,63,1):
			Arr1[1][0][j]=Arr[0][1][mx2]
			Arr1[1][0][j-9]=Arr[0][1][mx2-8]
			Arr1[1][0][j-18]=Arr[0][1][mx2-16]
			Arr1[1][0][j-27]=Arr[0][1][mx2-24]
			Arr1[1][0][j-36]=Arr[0][1][mx2-32]
			mx2=mx2+1

		for i in range(Nx1):
			Rem=i%12
			Preset1=1
			Preset2=1
			Preset3=0 if Rem<2 else 1
			Preset4=0 if Rem<4 else 1
			Preset5=0 if Rem<6 else 1
			Preset6=0 if Rem<8 else 1
			Preset7=0 if Rem<10 else 1
			Preset8=0

			if(Preset3==0):
				L1=65 if i%2==0 else 67
				U1=72
				m1=0
				mx1=Rem2+8*int(Rem/2)
				for j in range(L1,U1,1):
					Arr1[1][2*i][j]=Arr[m1][1][mx1]
					Arr1[1][2*(i+1)+1][j]=Arr[m1][1][mx1]
					mx1=mx1+1
				mx2=Rem2+(8*(int(Rem/2))+7)
				if(i%2==0):
					Arr1[1][2*i+1][63]=Arr[m1][1][mx2]

				for j in range(63,72,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]

			if(Rem==2):
				for j in range(63,72,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]			


			if(Preset4==0):
				L1=56 if i%2==0 else 58
				U1=63
				m1=int(i/18)
				for j in range(L1,U1,1):
					Arr1[1][2*(i+1)+1][j]=Arr1[1][2*i][j]
				mx2=Rem2+(8*(int(Rem/2)-1))+7
				if(i%2==0):
					Arr1[1][2*i+1][54]=Arr[m1][1][mx2]

				for j in range(54,63,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]

			if(Rem==4):
				for j in range(54,63,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]


			if(Preset5==0):
				L1=47 if i%2==0 else 49
				U1=54
				m1=int(i/18)
				for j in range(L1,U1,1):
					Arr1[1][2*(i+1)+1][j]=Arr1[1][2*i][j]
				mx2=Rem2+(8*(int(Rem/2)-2))+7
				if(i%2==0):
					Arr1[1][2*i+1][45]=Arr[m1][1][mx2]

				for j in range(45,54,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]
			if(Rem==6):
				for j in range(45,54,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]


			if(Preset6==0):
				L1=38 if i%2==0 else 40
				U1=45
				m1=int(i/18)
				for j in range(L1,U1,1):
					Arr1[1][2*(i+1)+1][j]=Arr1[1][2*i][j]
				mx2=Rem2+(8*(int(Rem/2)-3))+7
				if(i%2==0):
					Arr1[1][2*i+1][36]=Arr[m1][1][mx2]

				for j in range(36,45,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]
			if(Rem==8):
				for j in range(36,45,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]


			if(Preset7==0):
				L1=29 if i%2==0 else 31
				U1=36
				m1=int(i/18)
				for j in range(L1,U1,1):
					Arr1[1][2*(i+1)+1][j]=Arr1[1][2*i][j]
				mx2=Rem2+(8*(int(Rem/2)-4))+7
				if(i%2==0):
					Arr1[1][2*i+1][27]=Arr[m1][1][mx2]

				for j in range(27,36,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]

			if(Rem==10):
				for j in range(27,36,1):
					Arr1[1][2*(i+1)][j-9]=Arr1[1][2*i+1][j]


			if(Preset8==0 and i%2==0):
				m1=int(i/18)
				mx2=Rem2+(8*(int(Rem/2)-5))+7
				Arr1[1][2*i+1][18]=Arr[m1][1][mx2]
				Arr1[1][2*(i+1)][9]=Arr1[1][2*i+1][18]

		for i in range(Nx1):
			for j in range(72,81,1):
				Arr1[1][2*i+1][j]=Arr1[1][2*(i+1)][j-9]


	return(Arr1)
			


