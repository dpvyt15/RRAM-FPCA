import math
import sys
import random
import numpy as np
import time
import cmath
import joblib
from scipy import signal




def linspace(a, b, n):
	if n < 2:
		return b
	diff = (float(b) - a)/(n - 1)
	return [diff * i + a  for i in range(n)]


def sortingSingle(A):
	min=A[0]
	max=A[0]
	for i in range(len(A)):
		if(A[i]<min):
			min=A[i]
		if(A[i]>max):
			max=A[i]
	return max,min

def sorting(A):
	min=A[0][0]
	max=A[0][0]
	for i in range(len(A)):
		for j in range(len(A[0])):
			if(A[i][j]<min):
				min=A[i][j]
			if(A[i][j]>max):
				max=A[i][j]
	return max,min

def append_rows(matrix1, matrix2):
	new_matrix = []
	for row in matrix1:
		new_matrix.append(row)
	for row in matrix2:
		new_matrix.append(row)
	return new_matrix



def row(X,i):
	len1=len(X)
	M=[X[j][i] for j in range(len1)]
	return M