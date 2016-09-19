import pickle
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plot3d
import pandas as pan


def Err(X,y,w):
    prediction = np.dot(X,w)
    difference = np.subtract(y,prediction)
        
    return np.dot(np.transpose(difference),difference)  #(y-Xw)'(y-Xw)
    #print(ERR)
    #return 1
    

def GradientDescent(X,y,w):
    prediction = np.dot(X,w)
    xTranspose = np.transpose(X)
    
    return 2 * (np.dot(xTranspose,prediction) - (np.dot(xTranspose,y)))
    
    
def NormalEquation(X,y,w):   
  #ŵ = (X T X) -1 X T Y
  xTranspose = np.transpose(X)
  w = np.dot(np.linalg.inv(np.dot(xTranspose,X)),np.dot(xTranspose,y))
  return w

def LinearRegression(X,y,w):
    alpha  = 0.01
    listErr= np.zeros((100,1),dtype = np.double)
    listWt= np.zeros((100,2),dtype = np.double)
    for i in range(100):
        listErr[i,0] = Err(X,y,w)
        print(i,"   ",Err(X,y,w))
        listWt[i,0] = w[0,0]
        listWt[i,1] = w[1,0]
        w = w - alpha * GradientDescent(X,y,w)    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    A,B = np.meshgrid(listWt[:,0],listWt[:,1])
    #print(listWt)
    ax.plot_surface(A,B,listErr[:,0])  
    A = np.arange(100).reshape(100,1)
    print(np.shape(A))
    #plt.plot(A[:,0],listErr[:,0])
    #plt.xticks(np.arange(min(A), max(A)+1, 200))
    return w


def KFold(X,y,w,K):
    print(np.shape(X))
    
    numberOfRecords = len(X[:,1])
    sizeOfFold = numberOfRecords // K
    
    print(sizeOfFold)
    #print(X[0:5,:])
    maskX = np.ones(np.shape(x),dtype = np.bool)
    maskY = np.ones(np.shape(y),dtype = np.bool)
    sumErr = 0
    for looper in range(K):
        #Train
        print(looper,"   ")
        if K-1 == looper:
            XTest = X[looper * sizeOfFold:,:]        
            maskX[looper * sizeOfFold:,:] = False
            maskY[looper * sizeOfFold:,:] = False
            yTest = y[looper * sizeOfFold:,:]        
            
        else:
            XTest = X[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]        
            maskX[looper * sizeOfFold:(looper + 1) * sizeOfFold,:] = False
            maskY[looper * sizeOfFold:(looper + 1) * sizeOfFold,:] = False
            yTest = y[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]        

        XTrain = (X[maskX])
        yTrain = y[maskY]
        
        XTrain = np.delete(X,np.s_[looper * sizeOfFold:(looper + 1) * sizeOfFold],0)
        yTrain = np.delete(y,np.s_[looper * sizeOfFold:(looper + 1) * sizeOfFold],0)
        w = LinearRegression(XTrain,yTrain,w)
        sumErr += Err(XTest,yTest,w)
        print(Err(XTest,yTest,w))
        w = np.zeros((len(X[1,:]),1),dtype = int)
        
        #XTemp1 = X[numberOfRecords - sizeOfFold:,:]
        #XTemp = XTemp1
        
        #np.copyto(X[looper * sizeOfFold:(looper + 1) * sizeOfFold,:],X[numberOfRecords - sizeOfFold:,:])
        #X[looper * sizeOfFold:(looper + 1) * sizeOfFold,:] = X[numberOfRecords - sizeOfFold:,:] 
        #X[numberOfRecords - sizeOfFold:,:] = XTemp
        #XDup = X
        #X = Swap(X,XTemp,XTemp1,numberOfRecords,sizeOfFold,looper)
        #print(XTemp)
        #print(XTemp1)      
        #print(X)
        #XDup[looper * sizeOfFold:(looper + 1) * sizeOfFold,:] = XTemp1
        #X = XDup
        print(XTrain)
        
        #yTemp = y[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]
        #y[looper * sizeOfFold:(looper + 1) * sizeOfFold,:] = y[numberOfRecords - sizeOfFold:,:] 
        #y[numberOfRecords - sizeOfFold:,:] = yTemp
        print(yTrain)
        
            
    return sumErr / K

def Swap(X,XTemp,XTemp1,numberOfRecords,sizeOfFold,looper):
    
    
    for i in range(len(XTemp[:,1])):
      
      X[looper * sizeOfFold + i,:] = XTemp1[i,:] 
      X[numberOfRecords - sizeOfFold + i,:] = XTemp[i,:]

    return X


db = 2
    
pickle.dump(db,open('xyz.txt',"wb"))
X = np.array([[0.86,0.09,-0.85,0.87,-0.44,-0.43,-1.10,0.40,-0.96,0.17]
,[1,1,1,1,1,1,1,1,1,1]])

X = np.transpose(X)
w = np.zeros((len(X[1,:]),1),dtype = int)
#print(2 * (w))

y = np.array([[2.49,0.83,-0.25,3.10,0.87,0.02,-0.12,1.81,-0.83,0.43]])
y = np.transpose(y)
print(np.shape(y))
'''y = np.transpose(y)
print(np.dot(x,y))
#print(x * y)
print(x.T)
plt.plot([1,2,3,4],[5,3,5,6])
plt.xlabel('X AXIS')
plt.ylabel('Y AXIS')

plt.show()'''
#print(Err(x,y,w))
#print(GradientDescent(x,y,w))
#print(w-w)
#w = (NormalEquation(x,y,w))
#print(w)
#print(Err(x,y,w))
#LinearRegression(x,y,w)
#
#print(x)
#print(y)
print(KFold(X,y,w,3))

