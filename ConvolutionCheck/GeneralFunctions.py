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
	minA=min(A)
	maxA=max(A)
	return maxA,minA

def sorting(A):
	minA=min(map(min, A))
	maxA=max(map(max, A))
	return maxA,minA

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