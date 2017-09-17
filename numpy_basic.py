from urllib.request import urlopen
import numpy as np
'''print(np.array([[2,1,0,0],[0,2,1,0],[0,0,2,1]]))

mat = np.array([[2,1,0,0],[0,2,1,0],[0,0,2,1]])
mat = mat.reshape(12, 1)
print(mat)
print(np.ndarray.flatten(mat))'''

'''x_shape = tuple(map(int, input().split()))
X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)
y_shape = tuple(map(int, input().split()))
Y = np.fromiter(map(int, input().split()), np.int).reshape(y_shape)

if x_shape[1] == y_shape[1]:
    mul = X.dot(Y.T)
    print(mul)
else:
    print('matrix shapes do not match')'''

filename = input()
f = urlopen(filename)

sbux = np.loadtxt(f, skiprows=1, delimiter=',')
print(sbux.mean(axis=0))
#print(sbux)