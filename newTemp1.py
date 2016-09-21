# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 01:32:49 2016

@author: navin
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plot3d
import pandas as pan
import sys
import math

from numpy import genfromtxt

def getMatrixFromFile(inputFileName):
    matrix = genfromtxt(inputFileName, dtype = 'float',delimiter=',',skip_header = 1)
    matrix = np.round(matrix,2)
    return matrix

def Err(X,y,w):
    prediction = np.dot(X,w)
    #print('Prediction',prediction) genfromtxt
    difference = np.subtract(y,prediction)
    #print('Difference',difference)
    #print (np.dot(np.transpose(difference),difference))
    divisor = 2*(len(X[:,1]))
    return (np.dot(np.transpose(difference),difference))/divisor
      #(y-Xw)'(y-Xw)
    #print(ERR)
    #return 1

def GradientDescent(X,y,w):
    prediction = np.dot(X,w)
    '''
    print prediction.shape;
    xTranspose = np.transpose(X)
    print xTranspose.shape;
    answer = ((np.dot(xTranspose,prediction)) - (np.dot(xTranspose,y)))
    print answer.shape;
    print answer
    '''
    difference = prediction - y;
    difference = np.transpose(difference)
    difference = np.dot(difference,X);
    difference = np.transpose(difference);
    return difference

def NormalEquation(X,y,w):
  xTranspose = np.transpose(X)
  xTransposeXInverse = np.linalg.inv(np.dot(xTranspose,X))
  xTransposeY = (np.dot(xTranspose,y))
  w = np.dot(xTransposeXInverse,xTransposeY)
  #w = np.dot(np.linalg.inv(np.dot(xTranspose,X)),np.dot(xTranspose,y))
  return w

'''
def LinearRegression(X,y,w):
    alpha  = 0.01
    listErr= np.zeros((100,1),dtype = np.double)
    listWt= np.zeros((100,2),dtype = np.double)
    for i in range(100):
        listErr[i,0] = Err(X,y,w)
        #print(i,"   ",Err(X,y,w))
        listWt[i,0] = w[0,0]
        listWt[i,1] = w[1,0]
        w = w - alpha * GradientDescent(X,y,w)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A,B = np.meshgrid(listWt[:,0],listWt[:,1])
    #print(listWt)
    ax.plot_surface(A,B,listErr[:,0])
    A = np.arange(100).reshape(100,1)
    #print(np.shape(A))
    #plt.plot(A[:,0],listErr[:,0])
    #plt.xticks(np.arange(min(A), max(A)+1, 200))
    return w
'''

def LinearRegressionJaspal(X,y,w):
    alpha = 0.001
    listErr= np.zeros((10000,1),dtype = np.double)
    

    for i in range(10000):
        w = w-((alpha*GradientDescent(X,y,w))/len(X[:,1]))
        Error = Err(X,y,w)
        listErr[i,0] = Error
        #print ('w after ',i,'iteration is ',w)
        #print ('error after ',i,'iteration is',Error)
        
    fig = plt.figure()
    
    A = np.arange(10000).reshape(10000,1)
    #print(np.shape(A))
    plt.plot(A[:,0],listErr[:,0])
    plt.xlabel('Iterations')
    plt.ylabel('Loss Function')
    
    return w

def KFold(X,y,w,K):
    #print(np.shape(X))
    #X = np.random.shuffle(X)
    numberOfRecords = len(X[:,1])
    np.random.shuffle(X)
    sizeOfFold = numberOfRecords // K
    #print(sizeOfFold)
    sumErr = 0
    for looper in range(K):
        #Train
        print(looper,"   ")
        if K-1 == looper:
            XTest = X[looper * sizeOfFold:,:]
            yTest = y[looper * sizeOfFold:,:]
        else:
            XTest = X[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]
            yTest = y[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]
        XTrain = np.delete(X,np.s_[looper * sizeOfFold:(looper + 1) * sizeOfFold],0)
        yTrain = np.delete(y,np.s_[looper * sizeOfFold:(looper + 1) * sizeOfFold],0)
        #w = LinearRegressionJaspal(XTrain,yTrain,w)
        w = NormalEquation(X,y,w)
        sumErr += Err(XTest,yTest,w)
        print("K Fold",(Err(XTest,yTest,w)))
        w = np.zeros((len(X[1,:]),1),dtype = int)
        #print(yTrain)
    return sumErr / K

def Swap(X,XTemp,XTemp1,numberOfRecords,sizeOfFold,looper):
    for i in range(len(XTemp[:,1])):
      X[looper * sizeOfFold + i,:] = XTemp1[i,:]
      X[numberOfRecords - sizeOfFold + i,:] = XTemp[i,:]
    return X

'''X = np.array([[0.86,0.09,-0.85,0.87,-0.44,-0.43,-1.10,0.40,-0.96,0.17]
,[1,1,1,1,1,1,1,1,1,1]])

X = np.transpose(X)
#print(2 * (w))

y = np.array([[2.49,0.83,-0.25,3.10,0.87,0.02,-0.12,1.81,-0.83,0.43]])
y = np.transpose(y)
print(np.shape(y))'''
'''y = np.transpose(y)
print(np.dot(x,y))
#print(x * y)
print(x.T)
plt.plot([1,2,3,4],[5,3,5,6])
plt.xlabel('X AXIS')
plt.ylabel('Y AXIS')

plt.show()'''

sys.stdout=open("test.txt","w")

X = getMatrixFromFile('MontrealOnMarathon.csv')
y = getMatrixFromFile('YMarathon.csv')
X = np.round(X,2)
#print(X)
X[:,0:4] = X[:,0:4] / 3600
bias = np.ones((len(X[:,1]),1))
#print (bias)
X = np.append(X,bias,axis = 1)
#print X
#X = np.c_(X, np.ones(len(X[:,1])))
#print(X)
y = y[np.newaxis]
y = np.transpose(y)
y = y / 3600
#print len(X[:,1])
#print X.shape
#print y.shape
#print X,y
w = np.zeros((len(X[1,:]),1),dtype = int)
#print GradientDescent(X,y,w)
w = NormalEquation(X[0:2100,:],y[0:2100],w)
#print(X[0:2100,:],"--------------------------",y[0:2100])
#print("123",Err(X[2100:,:],y[2100:],w))
#print (LinearRegressionJaspal(X,y,w))
#w = LinearRegressionJaspal(X,y,w)
#print('------------------------------------------------------')
difference = (np.dot(X,w) - y)
difference = np.abs(difference)
#print(np.mean(difference,axis=0))
#for i in range(8000):
 #   print(difference[i,0])
print("AvgError",KFold(X,y,w,10))