#from __future__ import division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import csv

m = []
n = []

with open("train.csv","r+") as csvfile:
    cr = csv.reader(csvfile)
    cr.next()
    for row in cr:
        if len(row) <= 1 :
            continue
            m.append(row[0])
            n.append(row[1])

print m
#print (len(m))
print n
#print (len(n))

x = np.array([m], dtype = float).T
#print (len(x))
y = np.array([n], dtype = float).T
#print (len(y))

plt.plot(x, y, 'ro')
plt.axis([-100, 150, -50, 150])
plt.xlabel('truc x')
plt.ylabel('truc y')
plt.show()

# Building Xbar
one = np.ones((x.shape[0], 1))
#print (len(one))
xbar = np.concatenate((one, x), axis = 1)
#print (len(xbar))

# Calculating weights of the fitting line
A = np.dot(xbar.T, xbar)
b = np.dot(xbar.T, y)
print (' A= ', A)
print ('b = ', b)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(-100, 185, 2)
y0 = w_0 + w_1*x0
print (w_0)
print "w_0= ", round(w_0, 3)
print (w_1)
print "w_1= ", round(w_1, 3)

# Drawing the fitting line
plt.plot(x.T, y.T, 'ro')     # data
plt.plot(x0, y0)               # the fitting line
plt.axis([-100, 150, -50, 150])
plt.xlabel('truc x')
plt.ylabel('truc y')
plt.show()

with open('t.csv',"r+") as csvtest :
    test = csv.reader(csvtest)
    test.next()
    for i in test:
        if(len(i) <= 1) :
            continue
        row.append(i[0]);

data = np.array(row, dtype = float)

with open('predict.csv',"w+") as output :
    writer = csv.writer(output)
    writer.writerow(['x','y'])
    for j in data :
        y1 = j*w_1 + w_0
        writer.writerow([j, y1])

#print( 'gia tri x : ' ,y1 )
#print( 'gia tri y : ' ,y2 )




