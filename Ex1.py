# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:02:25 2021

@author: Yakov
"""
import sys # To remove

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import random
import timeit
from sklearn.datasets.samples_generator import make_regression


#Batch Gradient Descent
def Batch(DF, L, V, n):
    start = timeit.default_timer()
    D_m=0
    D_c=0
    M=0
    C=0
    E=0
    z=0
    ERROR = list(range(V))
    
    
    for j in range(V):
        
        for i in range(n):
            D_m = D_m + (DF.X[i] * (DF.y[i] - (M*DF.X[i] + C)))
        
        D_m = D_m * ((-2)/n)
    
    
        for i in range(n):
             D_c = D_c + ((DF.y[i] - (M*DF.X[i] + C)))
        
        D_c = (D_c * ((-2)/n))
    
            
        M = M - (L * D_m)
        C = C - (L * D_c)
    
        
        for i in range(n):
             z =  (DF.y[i] - (M*DF.X[i] + C))
             #z = z * z
             E = E + z
             
        
        ERROR[j] =( E * (1/n))
        
     
        D_m=0
        D_c=0
        E=0
        z=0
        
              
        
    his = "Batch Hist - L:{}, Size:{}".format(L,n)
    hdf = pd.DataFrame(ERROR, columns=[his])
    hdf.hist(bins=50)
    plt.show() 
    
    stop = timeit.default_timer()
    print("n:",n,"L:" ,L)
    print("TIME : ", stop - start)
        
#Stochastic gradient descent
def SGD(DF, L, V, n) :
    
    start = timeit.default_timer()
    
    D_m=0
    D_c=0
    M=0 
    C=0
    E = 0
    ERROR = list(range(V))
    
    for j in range(V):
        
        rnd = random.randint(0,n-1)
        
        D_m =  DF.X[rnd] * (DF.y[rnd]- (M*DF.X[rnd] + C))
        D_m = D_m *((-2)/n)
        
        D_c = (DF.y[rnd] - (M*DF.X[rnd] + C))
        D_c = D_c *((-2)/n)
        
        M = M - L * D_m
        C = C - L * D_c
        
        E = (DF.y[rnd] - (M*DF.X[rnd] + C))
        ERROR[j] = E * (1/n)
        
        
        D_m=0
        D_c=0
        E = 0
        
        
    his = "Stochastic Hist - L:{}, Size:{}".format(L,n)
    hdf = pd.DataFrame(ERROR, columns=[his])
    hdf.hist(bins=20)
    plt.show()
    
    stop = timeit.default_timer()
    
    
#Mini batch gradient descent
def MiniBGD(DF, L, V, n):
    
    start = timeit.default_timer()
    D_m=0
    D_c=0
    M=0 
    C=0
    E = 0
    ERROR = list(range(V))
    
    for j in range(V):
        
        rnd_1 = random.randint(0,n)
        R = list(range(n))
        
        for i in range(rnd_1):
            rnd_2 = random.randint(0,n-1)
            
            while(R[rnd_2] == n+1):
                rnd_2 = random.randint(0,n-1)
           
            R[rnd_2] = n+1
            
            D_m =  DF.X[rnd_2] * (DF.y[rnd_2]- (M*DF.X[rnd_2] + C))
            D_m = D_m *((-2)/n)
        
            D_c = (DF.y[rnd_2] - (M*DF.X[rnd_2] + C))
            D_c = D_c *((-2)/n)
        
            M = M - L * D_m
            C = C - L * D_c
        
            E = (DF.y[rnd_2] - (M*DF.X[rnd_2] + C))
            
            ERROR[j] = E * (1/n)
        
        
            D_m=0
            D_c=0
            E = 0
            
    his = "Mini Batch Hist - L:{}, Size:{}".format(L,n)
    hdf = pd.DataFrame(ERROR, columns=[his])
    hdf.hist(bins=50)
    plt.xlabel('Error')
    plt.show()        
       
    
    stop = timeit.default_timer()
    
    
    
#Create new Data
def Data(n_samples):
    
    X, y = make_regression(n_samples, n_features=1, n_informative=1, noise=0.01, 
                           random_state=0)
    DF = pd.DataFrame({'X':X.flatten(),'y':y})
    
    print(DF)
    plt.scatter(X,y, color='red')
    #plt.plot(X,y, color='blue', linewidth=1)
    plt.show()
   
    return (DF)


Data_20= Data(20)
Data_50= Data(50)
Data_100= Data(100)
Data_200= Data(200)




#Data_20

#Batch(Data_20,0.00001,10,20)
#Batch(Data_20,0.00001,500,20)
Batch(Data_20,0.00001,1000,20)


#Batch(Data_20,0.000001,10,20)
#Batch(Data_20,0.000001,500,20)
Batch(Data_20,0.000001,1000,20)

#Data_50
#Batch(Data_50,0.00001,10,50)
#Batch(Data_50,0.00001,500,50)
Batch(Data_50,0.00001,1000,50)

#Batch(Data_50,0.000001,10,50)
#Batch(Data_50,0.000001,500,50)
Batch(Data_50,0.000001,1000,50)

#Data_100
#Batch(Data_100,0.00001,10,100)
#Batch(Data_100,0.00001,500,100)
Batch(Data_100,0.00001,1000,100)

#Batch(Data_100,0.000001,10,100)
#Batch(Data_100,0.000001,500,100)
Batch(Data_100,0.000001,1000,100)

#Data_200
#Batch(Data_200,0.00001,10,200)
#Batch(Data_200,0.00001,500,200)
Batch(Data_200,0.00001,1000,200)

#Batch(Data_200,0.000001,10,200)
#Batch(Data_200,0.000001,500,200)
Batch(Data_200,0.000001,1000,200)



#Data_20

#SGD(Data_20,0.00001,10,20)
#SGD(Data_20,0.00001,500,20)
SGD(Data_20,0.00001,1000,20)

#SGD(Data_20,0.000001,10,20)
#SGD(Data_20,0.000001,500,20)
SGD(Data_20,0.000001,1000,20)

#Data_50
#SGD(Data_50,0.00001,10,50)
#SGD(Data_50,0.00001,500,50)
SGD(Data_50,0.00001,1000,50)

#SGD(Data_50,0.000001,10,50)
#SGD(Data_50,0.000001,500,50)
SGD(Data_50,0.000001,1000,50)

#Data_100
#SGD(Data_100,0.00001,10,100)
#SGD(Data_100,0.00001,500,100)
SGD(Data_100,0.00001,1000,100)

#SGD(Data_100,0.000001,10,100)
#SGD(Data_100,0.000001,500,100)
SGD(Data_100,0.000001,1000,100)

#Data_200
#SGD(Data_200,0.00001,10,200)
#SGD(Data_200,0.00001,500,200)
SGD(Data_200,0.00001,1000,200)

#SGD(Data_200,0.000001,10,200)
#SGD(Data_200,0.000001,500,200)
SGD(Data_200,0.000001,1000,200)

"""
#Data_20
MiniBGD(Data_20,0.00001,10,20)
MiniBGD(Data_20,0.00001,500,20)
MiniBGD(Data_20,0.00001,1000,20)

MiniBGD(Data_20,0.000001,10,20)
MiniBGD(Data_20,0.000001,500,20)
MiniBGD(Data_20,0.000001,1000,20)

#Data_50
MiniBGD(Data_50,0.00001,10,50)
MiniBGD(Data_50,0.00001,500,50)
MiniBGD(Data_50,0.00001,1000,50)

MiniBGD(Data_50,0.000001,10,50)
MiniBGD(Data_50,0.000001,500,50)
MiniBGD(Data_50,0.000001,1000,50)

#Data_100
MiniBGD(Data_100,0.00001,10,100)
MiniBGD(Data_100,0.00001,500,100)
MiniBGD(Data_100,0.00001,1000,100)

MiniBGD(Data_100,0.000001,10,100)
MiniBGD(Data_100,0.000001,500,100)
MiniBGD(Data_100,0.000001,1000,100)

#Data_200
MiniBGD(Data_200,0.00001,10,200)
MiniBGD(Data_200,0.00001,500,200)
MiniBGD(Data_200,0.00001,1000,200)

MiniBGD(Data_200,0.000001,10,200)
MiniBGD(Data_200,0.000001,500,200)
MiniBGD(Data_200,0.000001,1000,200)

"""
