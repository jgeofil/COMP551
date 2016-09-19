# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 04:19:52 2016

@author: navin
"""
from __future__ import division
import pandas as pd
import numpy as np
import csv
import re
from datetime import datetime
import sys
import random
import math

result = pd.DataFrame(columns=('participant_id','time_in_2012','time_in_2013','time_in_2014','time_in_2015','timsec'))
i =0;
for df in pd.read_csv('XY1.csv',sep=',', chunksize=1):
    id   = df.ix[:,0:1].iloc[0]['participant_id']
    col1 = df.ix[:,1:2].iloc[0]['time_in_2012']   
    col2 = df.ix[:,2:3].iloc[0]['time_in_2013']
    col3 = df.ix[:,3:4].iloc[0]['time_in_2014']
    col4 = df.ix[:,4:5].iloc[0]['time_in_2015']    
    Y    = df.ix[:,5:6].iloc[0]['timsec']
    #print(col1 + 2," ",col2," ",col3," ",col4," ")
    #print(df)
    
    total = col1 + col2 + col3 + col4    
    noOfNonZeroColumns = 4
    if col1 == 0:
        noOfNonZeroColumns -= 1
    if col2 == 0:
        noOfNonZeroColumns -= 1
    if col3 == 0:
        noOfNonZeroColumns -= 1
    if col4 == 0:
        noOfNonZeroColumns -= 1
    
    #if noOfNonZeroColumns == 0 or noOfNonZeroColumns == 4:
     #   continue
    
    #print(col1," ",col2," ",col3," ",col4," ",Y)
    
    ''' update the columns'''
    if col1 == 0:
        noise = (random.randrange(1,2500,1))
        col1 = total // noOfNonZeroColumns + noise * (-1 + (noise % 2) * 2)
        
        total += col1
        noOfNonZeroColumns += 1
    
    if col2 == 0:
        noise = (random.randrange(1,1500,2))
        col2 = total // noOfNonZeroColumns + noise * (-1 + (noise % 2) * 2)
        
        total += col2
        noOfNonZeroColumns += 1

    if col3 == 0:
        noise = (random.randrange(1,1500,2))
        col3 = total // noOfNonZeroColumns + noise * (-1 + (noise % 2) * 2)
        
        total += col3
        noOfNonZeroColumns += 1

    if col4 == 0:
        noise = (random.randrange(1,1500,2))
        col4 = total // noOfNonZeroColumns + noise * (-1 + (noise % 2) * 2)
        
        total += col4
        noOfNonZeroColumns += 1
        
    if  math.isnan(Y):
        Y = total // 4
        #print ("NAN")
    result.loc[i]=[id,col1,col2,col3,col4,Y]
    i = i + 1

result.to_csv('XYfilled.csv', index=False)
        
