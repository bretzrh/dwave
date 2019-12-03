from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import numpy as np

x = np.matrix('1 2')
y = np.matrix('1 2')
z = np.matrix('1 2 4')
s = np.matrix('1 2 4 8')
t = np.matrix('1 2 4 8')
P = 100

v1 = np.append(-2*x,-3*y,axis=1)
v1 = np.append(v1,-4*z,axis=1)
v1 = np.append(v1,0*s,axis=1)
v1 = np.append(v1,0*t,axis=1)

v2 = np.append(3*x,2*y,axis=1)
v2 = np.append(v2,z,axis=1)
v2 = np.append(v2,s,axis=1)
v2 = np.append(v2,0*t,axis=1)

v3 = np.append(2*x,5*y,axis=1)
v3 = np.append(v3,3*z,axis=1)
v3 = np.append(v3,0*s,axis=1)
v3 = np.append(v3,t,axis=1)

M1=np.diagflat(v1)
M2=np.transpose(v2)*v2-2*10*np.diagflat(v2)
M3=np.transpose(v3)*v3-2*15*np.diagflat(v3)

M=M1+P*M2+P*M3

Q = {}

for i in range(0,np.shape(M)[0]):
	Q[('x%02d' % i,'x%02d' % i)] = M[i,i]
	for j in range(i+1,np.shape(M)[1]):
		Q[('x%02d' % i,'x%02d' % j)] = M[i,j]+M[j,i]


response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=100)
for datum in response.data(['sample', 'energy', 'num_occurrences']):
	X = datum.sample['x00']+2*datum.sample['x01']
	Y = datum.sample['x02']+2*datum.sample['x03']
	Z = datum.sample['x04']+2*datum.sample['x05']+4*datum.sample['x06']
	S = datum.sample['x07']+2*datum.sample['x08']+4*datum.sample['x09']+8*datum.sample['x10']
	T = datum.sample['x11']+2*datum.sample['x12']+4*datum.sample['x13']+8*datum.sample['x14']
	C1 = 3*X+2*Y+Z
	C2 = 2*X+5*Y+3*Z
	E=-2*X-3*Y-4*Z+P*np.power(3*X+2*Y+Z+S-10, 2)+P*np.power(2*X+5*Y+3*Z+T-15, 2)-P*325
	print("X: ", X, "| Y: ", Y, "| Z: ", Z, "| S: ", S, "| T: ", T, "| Condition 1: ", C1, "| Condition 2: ", C2, "| Energy_Calc: ", E, "| Energy_Out: ", datum.energy, "| Num Occurrences: ", datum.num_occurrences)
