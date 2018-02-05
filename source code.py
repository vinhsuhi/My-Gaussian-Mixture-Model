# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 08:16:26 2018

@author: vinh
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy as sc
from scipy import random, linalg, stats, special

random.seed(10)

NProperties = 1
NClasses = 4
NObjects = 200
symmetric_dirichlet = 1
distanceBTWclasses = 20
DiffBTWSpreadOFclasses = 2

# The mean vector
Mu = [np.random.random(NProperties)*distanceBTWclasses*i for i in range(1,NClasses+1)]
# the sd vector
Var = [np.random.random(NProperties)*DiffBTWSpreadOFclasses*i for i in range(1,NClasses+1)]

if symmetric_dirichlet==1:
    theta = np.repeat(1.0/NClasses,NClasses)
else:
    a = np.ones(NClasses)
    n = 1
    p = len(a)
    rd = np.random.gamma(np.repeat(a,n),n,p)
    rd = np.divide(rd,np.repeat(np.sum(rd),p))
    theta = rd

r = np.random.multinomial(NObjects,theta)

rAlln = [np.random.normal(Mu[i], Var[i], r[i]) for i in range(0,NClasses)]

plotsize = 8
bins = 10
text_size = 16
axis_font = {'fontname':'Arial', 'size':'24'}
Title_font = {'fontname':'Arial', 'size':'28'}
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
for i in range(0,NClasses):
    ax.hist(rAlln[i], 10, color=color[i])

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('Frequency', **axis_font)
ax.set_xlabel('Property 1', **axis_font)
ax.set_title('Classes', y=1.08, **Title_font)
ax.figure.set_size_inches(plotsize,plotsize)
plt.savefig('ahihi.png')

y = rAlln[0]
for i in range(NClasses-1):
    y = np.hstack((y,rAlln[i+1]))

# Getting the true classes of the points 
v_true = np.zeros((1)) 
for i,j in enumerate(r):
    v_true = np.hstack((v_true, np.repeat(i+1, j)))

v_true = np.array(v_true[1:])
y_true = np.vstack((y, v_true))

# random shuffle the data points
np.random.shuffle(y_true.T)

y = y_true[0,:]

plotsize = 8
bins = 40
text_size = 16
axis_font = {'fontname':'Arial', 'size':'24'}
Title_font = {'fontname':'Arial', 'size':'28'}
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.hist(y,bins)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('Frequency', **axis_font)
ax.set_xlabel('Property 1', **axis_font)
ax.set_title('Classes', y=1.08, **Title_font)
ax.figure.set_size_inches(plotsize,plotsize)
plt.savefig('ahihi2.png')

v = np.array([random.randint(1, NClasses+1) for i in range(y.shape[0])])

broadness = 15
initMu = np.random.random(NClasses)*max(y)
initVar = np.random.random(NClasses)+broadness
initW = theta #np.random.random(NClasses)

def EStep(y, w, Mu, Sigma):
    
    r_ij = np.zeros((y.shape[0], Mu.shape[0]))
    
    for Object in range(y.shape[0]):
        
        r_ij_Sumj = np.zeros(Mu.shape[0])
        
        for jClass in range(Mu.shape[0]):
            r_ij_Sumj[jClass] = w[jClass] * sc.stats.norm.pdf(y[Object], Mu[jClass], np.sqrt(Sigma[jClass]))
        
        for jClass in range(r_ij_Sumj.shape[0]):
            r_ij[Object,jClass] =   r_ij_Sumj[jClass] / np.sum(r_ij_Sumj)
        
    return r_ij

r_n = EStep(y, initW, initMu, initVar)

def MStep(r, y, Mu, Sigma):
    
    N = y.shape[0]
    
    mu_j = np.zeros((N, Mu.shape[0]))
    sigma_j = np.zeros((N, Mu.shape[0]))
    
    for Object in range(y.shape[0]):
        
        # mean
        mu_j[Object,:] = r[Object,:] * y[Object]
        
        # sd
        sigma_j[Object,:] = r[Object,:] * np.square(-Mu + y[Object])

    w_j = np.sum(r, axis=0) / N
    mu_j = (1/np.sum(r, axis=0)) * np.sum(mu_j, axis=0)
    sigma_j = (1/np.sum(r, axis=0)) * np.sum(sigma_j, axis=0)
    
    return w_j,mu_j,sigma_j

w_n,mu_n,sigma_n = MStep(r_n, y, initMu, initVar)

Inititeration = 10
EMiteration = 200
lookLH = 20

for init in range(Inititeration):
    
    # starting values
    initMu = np.random.random(NClasses)*max(y)
    r_n = EStep(y, initW, initMu, initVar)
    w_n,mu_n,sigma_n = MStep(r_n, y, initMu, initVar)
    
    if init == 0:
        logLH = -1000000000000
        
    for i in range(EMiteration):

        # E step
        r_n = EStep(y, w_n, mu_n, sigma_n)

        # M step
        w_n,mu_n,sigma_n = MStep(r_n, y, mu_n, sigma_n)

        # compute log likelihood
        logLall = np.zeros((y.shape[0]))

        for Object in range(y.shape[0]):

            LH = np.zeros(NClasses)

            for jClass in range(NClasses):
                LH[jClass] = w_n[jClass] * sc.stats.norm.pdf(y[Object], mu_n[jClass], np.sqrt(sigma_n[jClass]))

            logLall[Object] = np.log(np.sum(LH))

        logL = np.sum(logLall)

        if i > EMiteration - lookLH:
            print(logL)

    if logL > logLH:
        logLH = logL
        print('found larger ', logLH)
        w_p = w_n
        mu_p = mu_n
        sigma_p = sigma_n
        r_p = r_n
